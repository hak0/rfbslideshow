use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::codecs::gif::GifDecoder;
use image::{DynamicImage, GenericImageView, ImageFormat, Rgb, AnimationDecoder};
use rusttype::{Font, Scale};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, RwLock};
use parking_lot::Mutex;
use std::thread;
use std::time::Duration;
use jwalk::WalkDir;
use rayon::prelude::*;
use rouille::{Request, Response};
use std::fs::OpenOptions;
use memmap2::{MmapOptions, MmapMut};

const FB_NUM_BUFFERS: usize = 2; // 2 is the maximum on raspberrypi

// thread-safe version of config
struct Config {
    config_path: RwLock<String>,
    folder_path: RwLock<String>,
    interval: AtomicU32,
    status_bar_font_path: RwLock<String>,
    status_bar_height: AtomicU32,
    http_server_port: AtomicU16,
    max_fps: AtomicU32,
}

#[derive(Deserialize)]
struct ConfigPrimitive {
    folder_path: String,
    interval: u32,
    status_bar_font_path: String,
    status_bar_height: u32,
    http_server_port: u16,
    max_fps: u32,
}

impl Config {
    fn new(config_path: &str) -> Self {
        let mut file = File::open(&config_path).expect("Failed to open config file");
        let mut content = String::new();
        file.read_to_string(&mut content)
            .expect("Failed to read config file");
        let config_file: ConfigPrimitive = toml::from_str(&content).expect("Failed to parse config file");

        Config {
            config_path: RwLock::new(config_path.to_owned()),
            folder_path: RwLock::new(config_file.folder_path),
            interval: AtomicU32::new(config_file.interval),
            status_bar_font_path: RwLock::new(config_file.status_bar_font_path),
            status_bar_height: AtomicU32::new(config_file.status_bar_height),
            http_server_port: AtomicU16::new(config_file.http_server_port),
            max_fps: AtomicU32::new(config_file.max_fps),
        }
    }

    fn load(&self) {
        let config_path = self.config_path.read().unwrap().clone();
        let mut file = File::open(&config_path).expect("Failed to open config file");
        let mut content = String::new();
        file.read_to_string(&mut content)
            .expect("Failed to read config file");
        let config_file: ConfigPrimitive = toml::from_str(&content).expect("Failed to parse config file");

        *self.folder_path.write().unwrap() = config_file.folder_path;
        self.interval.store(config_file.interval, std::sync::atomic::Ordering::SeqCst);
        *self.status_bar_font_path.write().unwrap() = config_file.status_bar_font_path;
        self.status_bar_height.store(config_file.status_bar_height, std::sync::atomic::Ordering::SeqCst);
        self.http_server_port.store(config_file.http_server_port, std::sync::atomic::Ordering::SeqCst);
        self.max_fps.store(config_file.max_fps, std::sync::atomic::Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct SlideshowState {
    paused: Arc<AtomicBool>,
    current_index: Arc<AtomicUsize>,
    image_list_len: Arc<AtomicUsize>,
    need_rescan: Arc<AtomicBool>,
    need_query: Arc<AtomicBool>,
    slideshow_condvar: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

#[derive(Clone)]
struct FrameBufferInfo {
    width: u32,
    height: u32,
    bytes_per_line: u32,
    bits_per_pixel: u32,
}

#[derive(Clone)]
struct FrameBufferState {
    mmap: Arc<Mutex<MmapMut>>,
    fb_info: FrameBufferInfo,
    fb_idx: Arc<AtomicUsize>,
    fb_file: Arc<Mutex<File>>,
    var_screen_info: Arc<Mutex<framebuffer::VarScreeninfo>>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 || args[1] != "-c" {
        eprintln!("Usage: {} -c <config.toml>", args[0]);
        std::process::exit(1);
    }


    // initialize data containers
    // lock order: config -> image_list
    let config = Arc::new(Config::new(&args[2]));
    let fb_state = Arc::new(setup_framebuffer());
    let slideshow_state = Arc::new(SlideshowState {
        paused: Arc::new(AtomicBool::new(false)),
        current_index: Arc::new(AtomicUsize::new(0)),
        image_list_len: Arc::new(AtomicUsize::new(0)),
        need_rescan: Arc::new(AtomicBool::new(true)),
        need_query: Arc::new(AtomicBool::new(false)),
        slideshow_condvar: Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new())),
    });
    let (tx, rx) = std::sync::mpsc::channel::<String>();

    // Spawn a new thread to run the warp server
    let port = config.http_server_port.load(std::sync::atomic::Ordering::Relaxed);
    let slideshow_state_clone = slideshow_state.clone();
    let config_clone = config.clone();
    std::thread::spawn(move || {
        let rx_clone = Arc::new(Mutex::new(rx));
        rouille::start_server(format!("0.0.0.0:{}", port), move |request| {
            setup_routes(request, slideshow_state_clone.clone(), config_clone.clone(), rx_clone.clone())
        });
    });

    // Create a new thread to increase the index with configured interval
    let slideshow_state_clone = slideshow_state.clone();
    let config_clone = config.clone();
    std::thread::spawn(move || {
        loop {
            let sleep_interval = config_clone.interval.load(std::sync::atomic::Ordering::Relaxed) as u64;
            std::thread::sleep(std::time::Duration::from_secs(sleep_interval));
            maybe_increase_index(&slideshow_state_clone);
            println!("index: {}", slideshow_state_clone.current_index.load(std::sync::atomic::Ordering::Relaxed));
        }
    });

    // Run the slideshow loop
    let slideshow_state_clone = slideshow_state.clone();
    let config_clone = config.clone();
    std::thread::spawn(move || {
        run_slideshow(fb_state, slideshow_state_clone, config_clone, tx);
    });

    // Setup signal handler for SIGINT
    ctrlc::set_handler(move || {
        println!("SIGINT received");
        // Perform your cleanup here
        let tty_path = "/dev/tty0";
        let path = PathBuf::from(tty_path);
        // switch back to tty text mode, so after exiting, the user returned to the interactive shell
        framebuffer::Framebuffer::set_kd_mode_ex(&path, framebuffer::KdMode::Text).expect("Failed to set KD_TEXT mode");
        println!("Dropped FrameBufferState, setting /dev/tty0 back to text mode.");
        // normal exit
        std::process::exit(0);
    }).expect("Error setting SIGINT(ctrl+c) handler");


    // Keep the main function alive
    loop {
        std::thread::sleep(std::time::Duration::from_secs(60));
    }
}

fn setup_framebuffer() -> FrameBufferState {
    let framebuffer_path = "/dev/fb0";
    let path = PathBuf::from(framebuffer_path);
    let file = OpenOptions::new().read(true).write(true).create(true).open(&path).expect("Failed to open fb file");
    //let screeninfo =  framebuffer::VarScreeninfo { xres: 3840, yres: 2160, xres_virtual: 3840, yres_virtual: 2160, xoffset: 0, yoffset: 0, bits_per_pixel: 32, grayscale: 0, red: framebuffer::Bitfield { offset: 16, length: 8, msb_right: 0 }, green: framebuffer::Bitfield { offset: 8, length: 8, msb_right: 0 }, blue: framebuffer::Bitfield { offset: 0, length: 8, msb_right: 0 }, transp: framebuffer::Bitfield { offset: 24, length: 8, msb_right: 0 }, nonstd: 0, activate: 0, height: 0, width: 0, accel_flags: 1, pixclock: 0, left_margin: 0, right_margin: 0, upper_margin: 0, lower_margin: 0, hsync_len: 0, vsync_len: 0, sync: 0, vmode: 512, rotate: 0, colorspace: 0, reserved: [0, 0, 0, 0] };
    let mut screeninfo = framebuffer::Framebuffer::get_var_screeninfo(&file).expect("failed to get var_screen_info");
    screeninfo.yres_virtual = screeninfo.yres * FB_NUM_BUFFERS as u32;
    println!("screeninfo: {:?}", screeninfo);
    // write framebuffer info
    framebuffer::Framebuffer::put_var_screeninfo(&file, &screeninfo).expect("failed to put var_screen_info");
    let fbinfo = FrameBufferInfo {
        width: screeninfo.xres,
        height: screeninfo.yres,
        bytes_per_line: screeninfo.xres * screeninfo.bits_per_pixel / 8,
        bits_per_pixel: screeninfo.bits_per_pixel,
    };
    // setup mmap for framebuffer
    let framebuffer_size = fbinfo.bytes_per_line as usize * fbinfo.height as usize;
    // double buffer, so size should be FB_NUM_BUFFERS x screen pixel sizes
    let mmap = unsafe { MmapOptions::new().len(framebuffer_size * FB_NUM_BUFFERS).populate().map_mut(&file).expect("Failed to mmap fb") };
    #[cfg(unix)]
    {
        mmap.advise(memmap2::Advice::Sequential).expect("Failed to advise mmap");
        mmap.advise(memmap2::Advice::DontFork).expect("Failed to advise mmap");
    }
    // enter kd_mode graphics to avoid strange tty output like unervoltage or something
    let tty_path = "/dev/tty0";
    let path = PathBuf::from(tty_path);
    framebuffer::Framebuffer::set_kd_mode_ex(&path, framebuffer::KdMode::Graphics).expect("Failed to set KD_GRAPHICS mode");
    FrameBufferState {
        mmap: Arc::new(Mutex::new(mmap)),
        fb_info: fbinfo,
        fb_idx: Arc::new(AtomicUsize::new(0)),
        fb_file: Arc::new(Mutex::new(file)),
        var_screen_info: Arc::new(Mutex::new(screeninfo)),
    }
}

// currently there is no lock or mutex to protect the opreations
// the user needs to take care by themselves.
// e.g. don't rescan when uploading, or don't continue/prev/next/seek when rescnaning
fn setup_routes(request: &Request, slideshow_state: Arc<SlideshowState>, config: Arc<Config>, rx: Arc<Mutex<Receiver<String>>>) -> Response {
    rouille::router!(request,
        // usage: http://serverhost/pause -> pause the slideshow loop
        (GET) (/pause) => {
            slideshow_state.paused.store(true, Ordering::Relaxed);
            Response::json(&"Paused")
        },
        // usage: http://serverhost/resume -> resume the slideshow loop
        (GET) (/resume) => {
            slideshow_state.paused.store(false, Ordering::Relaxed);
            // awake the slideshow loop
            slideshow_state.slideshow_condvar.1.notify_one();
            Response::json(&"Resumed")
        },
        // usage: http://serverhost/prev -> move to the previous image
        (GET) (/prev) => {
            let image_list_len = slideshow_state.image_list_len.load(Ordering::Relaxed);
            if image_list_len > 0 {
                let current_index_value = slideshow_state.current_index.load(Ordering::Relaxed);
                let new_index_value = (current_index_value + image_list_len - 1) % image_list_len;
                slideshow_state.current_index.store(new_index_value, Ordering::Relaxed);
                // awake the slideshow loop
                slideshow_state.slideshow_condvar.1.notify_one();
            }
            Response::json(&"Previous")
        },
        // usage: http://serverhost/next -> move to the next image
        (GET) (/next) => {
            let image_list_len = slideshow_state.image_list_len.load(Ordering::Relaxed);
            if image_list_len > 0 {
                let current_index_value = slideshow_state.current_index.load(Ordering::Relaxed);
                let new_index_value = (current_index_value + 1) % image_list_len;
                slideshow_state.current_index.store(new_index_value, Ordering::Relaxed);
                // awake the slideshow loop
                slideshow_state.slideshow_condvar.1.notify_one();
            }
            Response::json(&"Next")
        },
        // usage: http://serverhost/rescan -> rescan the folder for new images
        (GET) (/rescan) => {
            slideshow_state.need_rescan.store(true, Ordering::Relaxed);
            // awake the slideshow loop
            slideshow_state.slideshow_condvar.1.notify_one();
            let start = std::time::Instant::now();
            while slideshow_state.need_rescan.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if start.elapsed().as_secs() > 180 {
                    slideshow_state.need_rescan.store(false, Ordering::Relaxed);
                    return Response::json(&"Rescan failed, 180s timeout");
                }
            }
            Response::json(&"Rescan")
        },
        // usage: http://serverhost/reload -> reload the config file
        (GET) (/reload) => {
            config.load();
            Response::json(&"Config Reloaded")
        },
        // usage: http://serverhost/seek/{index} -> seek to the specified index
        (GET) (/seek/{index: usize}) => {
            let image_list_len = slideshow_state.image_list_len.load(Ordering::Relaxed);
            if index >= image_list_len {
                return Response::json(&"Index out of range");
            }
            slideshow_state.current_index.store(index, Ordering::Relaxed);
            // awake the slideshow loop
            slideshow_state.slideshow_condvar.1.notify_one();
            Response::json(&"Seeked")
        },
        // usage: http://serverhost/query -> query the current image name
        (GET) (/query) => {
            slideshow_state.need_query.store(true, Ordering::Relaxed);
            // awake the slideshow loop
            slideshow_state.slideshow_condvar.1.notify_one();
            let image_name = rx.lock().recv().unwrap();
            Response::json(&format!("Image name: {}", image_name))
        },
        // usage: http://serverhost/clearfolder -> clear the folder
        (GET) (/clearfolder) => {
            slideshow_state.paused.store(true, Ordering::Relaxed);
            slideshow_state.image_list_len.store(0, Ordering::Relaxed);
            slideshow_state.current_index.store(0, Ordering::Relaxed);

            let folder_path = config.folder_path.read().unwrap().clone();
            // skip if folder doesn't exist
            if !Path::new(&folder_path).exists() {
                return Response::json(&"Folder doesn't exist");
            }
            for entry in fs::read_dir(&folder_path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_dir() {
                    fs::remove_dir_all(&path).unwrap();
                } else {
                    fs::remove_file(&path).unwrap();
                }
            }

            Response::json(&"Folder Cleared")
        },
        // usage: http://serverhost/upload/{image_name} -> upload an image
        (POST) (/upload/{image_name: String}) => {
            let data = request.data().expect("Failed to read request data");
            let mut body = Vec::new();
            data.take(10 * 1024 * 1024).read_to_end(&mut body).unwrap(); // limit to 10MB

            slideshow_state.paused.store(true, Ordering::Relaxed);
            let folder_path = config.folder_path.read().unwrap().clone();
            // create folder path if not existing
            if !Path::new(&folder_path).exists() {
                fs::create_dir_all(&folder_path).unwrap();
            }

            let image_path = Path::new(&folder_path).join(&image_name);

            if image_path.exists() {
                return Response::json(&"Image already exists");
            }

            if image_path.components().any(|comp| comp == std::path::Component::ParentDir) {
                return Response::json(&"Invalid image name, has parent path component");
            }

            fs::write(&image_path, body).unwrap();
            Response::json(&"Uploaded")
        },
        _ => Response::empty_404()
    )
}
fn maybe_increase_index(slideshow_state: &Arc<SlideshowState>) {
    let current_index = slideshow_state.current_index.load(std::sync::atomic::Ordering::Relaxed);
    let image_list_len = slideshow_state.image_list_len.load(std::sync::atomic::Ordering::Relaxed);
    let paused = slideshow_state.paused.load(std::sync::atomic::Ordering::Relaxed);
    if image_list_len != 0 && !paused {
        let new_index = (current_index + 1 + image_list_len) % image_list_len;
        slideshow_state.current_index.store(new_index, std::sync::atomic::Ordering::Relaxed);
    }
}

// main "event loop" for the slideshow
fn run_slideshow(
    fb_state: Arc<FrameBufferState>,
    slideshow_state: Arc<SlideshowState>,
    config: Arc<Config>, 
    tx: std::sync::mpsc::Sender<String>,
) {
    // private image list inside run_slideshow, won't share it
    let mut image_list: Vec<(PathBuf, ImageFormat)> = Vec::new();
    
    let mut current_index_last_time = None;
    let mut terminal_signal = Arc::new(AtomicBool::new(false));
    loop {
        let current_index = slideshow_state.current_index.load(std::sync::atomic::Ordering::Relaxed);
        let image_list_len = slideshow_state.image_list_len.load(std::sync::atomic::Ordering::Relaxed);
        let needs_rescan = slideshow_state.need_rescan.load(std::sync::atomic::Ordering::Relaxed);
        let needs_query = slideshow_state.need_query.load(std::sync::atomic::Ordering::Relaxed);
        let slideshow_condvar_pair = slideshow_state.slideshow_condvar.clone();

        // response to the query request
        if needs_query {
            slideshow_state.need_query.store(false, std::sync::atomic::Ordering::Relaxed);
            if image_list_len == 0 {
                tx.send(String::from("Image list is Empty")).unwrap();
            } else if current_index >= image_list_len {
                tx.send(String::from("Invalid")).unwrap();
            } else if let Some(path) = image_list.get(current_index).map(|(path, _)| path) {
                tx.send(path.to_str().unwrap().to_string()).unwrap();
            }
        }       

        // if needs rescan, re-init image_list
        if needs_rescan {
            let scan_folder_path = config.folder_path.read().unwrap().clone();
            slideshow_state.need_rescan.store(false, std::sync::atomic::Ordering::Relaxed);
            image_list = scan_folder(&scan_folder_path, slideshow_state.clone());
            continue;
        }

        // if image_list is empty, sleep for 1 second
        if image_list_len == 0 {
            sleep_slideshow(slideshow_condvar_pair, Duration::from_secs(1));
            continue;
        }

        // if the index is the same as the index last time, sleep for 1 second
        if current_index_last_time == Some(current_index) {
            sleep_slideshow(slideshow_condvar_pair, Duration::from_secs(1));
            continue;
        } else {
            current_index_last_time = Some(current_index);
        }

        // if the current index is greater than the length of the image list, reset the index to 0
        if current_index >= image_list_len {
            slideshow_state.current_index.store(0, std::sync::atomic::Ordering::Relaxed);
            continue;
        }

        // start displaying the new image
        let (path, format) = image_list[current_index].clone();

        // terminate the last running instance
        terminal_signal.store(true, std::sync::atomic::Ordering::Relaxed);
        terminal_signal = Arc::new(AtomicBool::new(false));

        if path.exists() && path.is_file() {
            let bar_text = format!(
                "[{}/{}] {}",
                current_index,
                image_list_len,
                path.file_name().unwrap().to_str().unwrap()
            );
            if Some(()) == match format {
                ImageFormat::Gif => {
                    let cloned_config = config.clone();
                    let cloned_fb_state = fb_state.clone();
                    let cloned_terminal_signal = terminal_signal.clone();
                    std::thread::spawn(move || {
                        display_gif(&path, cloned_config, cloned_fb_state, &bar_text, cloned_terminal_signal);
                    });
                    Some(())
                },
                ImageFormat::WebP => {
                    let cloned_config = config.clone();
                    let cloned_fb_state = fb_state.clone();
                    let cloned_terminal_signal = terminal_signal.clone();
                    std::thread::spawn(move || {
                        display_single_img(&path, format, cloned_config, cloned_fb_state, &bar_text, cloned_terminal_signal);
                    });
                    Some(())
                }
                _ => None,
            } {
                // success
                continue;
            }
        }

        // else the image is not found or not supported
        // increase the index immediately
        maybe_increase_index(&slideshow_state)
    }
}

// use condvar to sleep for a while, that can be awaken by condvar
// maybe shorter due to Spurious Wakeups, but it's ok
fn sleep_slideshow(slideshow_condvar_pair: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>, duration: Duration) {
    let (lock, condvar) = &*slideshow_condvar_pair;
    let mut _guard = lock.lock().unwrap();
    let _ = condvar.wait_timeout(_guard, duration);
}

fn scan_folder(scan_folder_path: &str, slideshow_state: Arc<SlideshowState>) -> Vec<(PathBuf, ImageFormat)> {
    // pause the slideshow first
    let pause_value = slideshow_state.paused.load(std::sync::atomic::Ordering::Relaxed);
    slideshow_state.paused.store(true, std::sync::atomic::Ordering::Relaxed);

    let image_list : Vec<(PathBuf, ImageFormat)> = WalkDir::new(scan_folder_path)
        .into_iter()
        .filter_map(|entry_result| entry_result.ok()) // Handle errors by filtering out invalid entries
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| {
            let path = entry.path();
            get_image_format(&path).map(|format| (path.to_path_buf(), format))
        })
        .collect();
    slideshow_state.image_list_len.store(image_list.len(), std::sync::atomic::Ordering::Relaxed);
    slideshow_state.current_index.store(0, std::sync::atomic::Ordering::Relaxed);
    slideshow_state.paused.store(pause_value, std::sync::atomic::Ordering::Relaxed);
    image_list
}

fn get_image_format(path: &Path) -> Option<ImageFormat> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    match ext.as_str() {
        "gif" => Some(ImageFormat::Gif),
        "webp" => Some(ImageFormat::WebP),
        "png" => Some(ImageFormat::Png),
        "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
        "bmp" => Some(ImageFormat::Bmp),
        _ => None,
    }
}

fn display_image(img: &DynamicImage, resize_algorithm : ResizeAlg , config : Arc<Config>, fb_state: Arc<FrameBufferState>, status_text: &str) {
    // read the image with the specified format
    let start = std::time::Instant::now();
    let fb_info = fb_state.fb_info.clone();
    if resize_algorithm != ResizeAlg::Nearest {
        let img = resize_image(&img, &fb_info, config.clone(), resize_algorithm);
        let elapsed = start.elapsed();
        println!("Resizing Elapsed: {:?}", elapsed);
        // Combine the resized image and the status bar
        write_to_framebuffer(&img, &status_text, fb_state, config.clone(), 1);
    } else {
        println!("Skip Resizing");
        let scale_x = fb_info.width as f32 / img.width() as f32;
        let scale_y = fb_info.height as f32 / img.height() as f32;
        let scale = scale_x.min(scale_y).floor() as u32;
        // skip resizing and do integer-factor resize in the fb write
        write_to_framebuffer(&img, &status_text, fb_state, config.clone(), scale);
    }
}

fn resize_image(img: &DynamicImage, fb_info: &FrameBufferInfo, config: Arc<Config>, algorithm: ResizeAlg) -> DynamicImage {
    let ratio = img.width() as f32 / img.height() as f32;
    let status_bar_height = config.status_bar_height.load(std::sync::atomic::Ordering::Relaxed);
    let new_height = fb_info.height - status_bar_height;
    if img.height() != new_height {
        // Create Resizer instance and resize source image
        // into buffer of destination image
        let mut resizer = Resizer::new();
        let new_width = (new_height as f32 * ratio) as u32;
        let mut dest_img = DynamicImage::ImageRgb8(image::RgbImage::new(new_width, new_height));
        resizer.resize(img, &mut dest_img, &ResizeOptions::new().resize_alg(algorithm)).unwrap();
        dest_img
    } else {
        img.clone()
    }
}

fn draw_statusbar_text(raw_mem: &mut [u8], fb_info: &FrameBufferInfo,config: Arc<Config>, status_text: &str) {
    // Create a new image for the status bar only
    let bytes_per_pixel = (fb_info.bits_per_pixel / 8) as i32;

    // Read the font data from the file specified in the config
    // check if font exists
    let font_path_str = config.status_bar_font_path.read().unwrap().trim().to_string();
    let status_bar_height = config.status_bar_height.load(std::sync::atomic::Ordering::Relaxed).clone();
    let font_path = Path::new(&font_path_str);
    println!("statusbar font: \"{}\"", font_path_str);
    if !Path::new(font_path).exists() {
        println!("font does not exist");
    }
    let font_data = fs::read(font_path_str).expect("Failed to read font file");
    let font = Font::try_from_bytes(&font_data).expect("Failed to load font");
    let font_size = (status_bar_height - 4) as f32;
    let scale = Scale { x: font_size, y: font_size };
    let bar_color = Rgb([255, 255, 255]); // White color for background
    let text_color = Rgb([0, 0, 0]); // Black color for text

    // Calculate the width of the text
    let v_metrics = font.v_metrics(scale);
    let glyphs: Vec<_> = font.layout(status_text, scale, rusttype::point(0.0, 0.0)).collect();
    let text_width = glyphs.iter().rev().filter_map(|g| g.pixel_bounding_box()).next().map_or(0, |bb| bb.max.x) as f32;

    // Calculate the starting position for right alignment
    let offset = rusttype::point(fb_info.width as f32 - text_width - 10.0, v_metrics.ascent);

    // Draw the text on the status bar, right aligned
    for glyph in font.layout(status_text, scale, offset) {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|x, y, v| {
                let x = x as i32 + bounding_box.min.x;
                let y = y as i32 + bounding_box.min.y;
                if x >= 0 && x < fb_info.width as i32 && y >= 0 && y < status_bar_height as i32 {
                    let x_offset = (x * bytes_per_pixel) as usize;
                    let y_offset = y as usize * fb_info.bytes_per_line as usize;
                    // reverse rgb to bgr
                    let r = (v * text_color[0] as f32 + (1.0 - v) * bar_color[0] as f32) as u8;
                    let g = (v * text_color[1] as f32 + (1.0 - v) * bar_color[1] as f32) as u8;
                    let b = (v * text_color[2] as f32 + (1.0 - v) * bar_color[2] as f32) as u8;
                    raw_mem[y_offset + x_offset + 0] = b;
                    raw_mem[y_offset + x_offset + 1] = g;
                    raw_mem[y_offset + x_offset + 2] = r;
                }
            });
        }
    }
}

fn convert_rgba_or_rgb_to_bgr(img: DynamicImage) -> DynamicImage {
    let start = std::time::Instant::now();
    let img = match img.color() {
        image::ColorType::Rgba8 => {
            // Convert to RGBA8
            let img = img.to_rgba8();
            let (img_width, img_height) = img.dimensions();
            let img_raw = img.into_raw();
            let img_pixels = img_raw.len() / 4; // Number of pixels

            // Create a new vector to store the BGR data
            let mut bgr_raw = vec![0u8; img_pixels * 3];

            // Process each pixel in parallel
            bgr_raw.par_chunks_mut(3).enumerate().for_each(|(i, bgr_pixel)| {
                let r = img_raw[i * 4] as u16;
                let g = img_raw[i * 4 + 1] as u16;
                let b = img_raw[i * 4 + 2] as u16;
                let alpha = img_raw[i * 4 + 3] as u16;
                let r_background = 255 as u16;
                let g_background = 255 as u16;
                let b_background = 255 as u16;
                let r = (r * alpha / 255 + r_background * (255 - alpha) / 255) as u8;
                let g = (g * alpha / 255 + g_background * (255 - alpha) / 255) as u8;
                let b = (b * alpha / 255 + b_background * (255 - alpha) / 255) as u8;
                bgr_pixel[0] = b;
                bgr_pixel[1] = g;
                bgr_pixel[2] = r;
            });

            // Create a new ImageBuffer from the BGR data
            let new_buffer = image::ImageBuffer::from_vec(img_width, img_height, bgr_raw);
            DynamicImage::ImageRgb8(new_buffer.unwrap())
        },
        image::ColorType::Rgb8 => {
            let img_buffer = img.to_rgb8();
            let (img_width, img_height) = img_buffer.dimensions();
            let mut img_raw = img_buffer.into_raw();
            // rgb -> bgr
            img_raw.par_chunks_mut(3).for_each(|pixel| {
                pixel.swap(0, 2);
            });
            let new_buffer = image::ImageBuffer::from_vec(img_width, img_height, img_raw);
            DynamicImage::ImageRgb8(new_buffer.unwrap())
        },
        _ => {
            println!("Not supported color type");
            img
        }
    };
    let elapsed = start.elapsed();
    println!("Color Converting Elapsed: {:?}", elapsed);
    img
}

fn display_single_img(path: &Path, format: ImageFormat, config: Arc<Config>, fb_state: Arc<FrameBufferState>, status_text: &str, terminal_signal: Arc<AtomicBool>) {
    if let Ok(img) = image::open(path) {
        println!("image format: {:?}", format);
        println!("image color: {:?}", img.color());
        // if terminal_signal is true, skip loading image and early return
        if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

        // convert the image to bgr format
        let img = convert_rgba_or_rgb_to_bgr(img);


        // if terminal_signal is true, skip loading image and early return
        if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

        display_image(&img, ResizeAlg::Convolution(FilterType::Mitchell), config, fb_state, status_text)
    }
}

fn display_gif(path: &Path, config: Arc<Config>, fb_state: Arc<FrameBufferState>, status_text: &str, terminal_signal: Arc<AtomicBool>) {
    if let Ok(file) = File::open(path) {
        let file_in = io::BufReader::new(file);
        let decoder = GifDecoder::new(file_in).unwrap();
        let mut frames = decoder.into_frames();
        let mut img_vec = Vec::new();
        let mut img_vec_idx = 0;
        let max_fps = config.max_fps.load(std::sync::atomic::Ordering::Relaxed) as f32;
        let fps_delay = Duration::from_secs_f32(1.0 / max_fps);

        loop {
            // if terminal_signal is true, skip loading image and early return
            if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

            // decode the next frame
            let start = std::time::Instant::now();
            let (img, gif_delay) = if let Some(Ok(frame)) = frames.next() {
                // the frames is not empty, retrieve a new frame and display it
                // cache it in the img_vec
                let delay = frame.delay();
                let img = DynamicImage::ImageRgba8(frame.into_buffer());
                let img = convert_rgba_or_rgb_to_bgr(img);
                let pair = (img, delay);
                img_vec.push(pair.clone());
                pair
            } else {
                // the frames is empty, play cached imgs
                let pair = img_vec[img_vec_idx].clone();
                img_vec_idx = (img_vec_idx + 1) % img_vec.len();
                pair
            };
            let elapsed = start.elapsed();
            println!("GIF Frame Decoding Elapsed: {:?}", elapsed);

            // if terminal_signal is true, skip loading image and early return
            if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

            display_image(&img, ResizeAlg::Nearest, Arc::clone(&config), fb_state.clone(), status_text);

            // if terminal_signal is true, skip loading image and early return
            if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

            // Sleep for the delay time
            let elapsed = start.elapsed();
            let delay = fps_delay.max(Duration::from(gif_delay));
            if Duration::from(delay) > elapsed {
                thread::sleep(Duration::from(delay) - elapsed);
            }
        }
    }
}

fn write_to_framebuffer(img: &DynamicImage, status_text: &str, fb_state: Arc<FrameBufferState>, config: Arc<Config>, scale: u32) {
    // Time it
    let fb_info = fb_state.fb_info.clone();
    let bytes_per_pixel = (fb_info.bits_per_pixel / 8) as usize;
    let bytes_per_line = fb_info.bytes_per_line as usize;
    let fb_height = fb_info.height as usize;
    let fb_width = fb_info.width as usize;

    let fb_size = bytes_per_line as usize * fb_height as usize;
    let fb_idx = fb_state.fb_idx.load(std::sync::atomic::Ordering::Relaxed);
    let mut mmap_unlock = fb_state.mmap.lock();
    let mmap = &mut mmap_unlock[fb_idx * fb_size..(fb_idx + 1) * fb_size];

    let status_bar_height = config.status_bar_height.load(std::sync::atomic::Ordering::Relaxed) as usize;

    let start = std::time::Instant::now();
    // Draw Image line-by-line
    let (img_width, img_height) = img.dimensions();
    let img_width = img_width as usize;
    let img_height = img_height as usize;
    let scale = scale as usize;
    // To draw the image at 2/3 of the screen width, calculate the starting position
    let x_offset_ratio = 2.0 / 3.0;
    let start_x = ((fb_width as f32 * x_offset_ratio) - (scale as f32 * img_width as f32 * x_offset_ratio)) as usize;
    let start_y = (fb_height - scale * img_height - status_bar_height) / 2;
    let end_x = (start_x + img_width * scale).min(fb_width);
    let end_y = (start_y + img_height * scale).min(fb_height - status_bar_height);

    // If image is not type Rgb8 (actually bgr8), warning
    if img.color() != image::ColorType::Rgb8 {
        println!("Warning: image color type is not Rgb8(BGR8)");
    }

    let img_raw = img.as_rgb8().unwrap().as_raw();

    // draw the white space first, parallelize the drawing
    mmap[0..start_y * bytes_per_line].fill(0xFFu8);
    // draw the image
    let image_buffer = &mut mmap[start_y * bytes_per_line..(start_y + img_height * scale) * bytes_per_line];
    // draw the image line by line, parallelize the drawing, combine several lines as a batch
    {
        image_buffer.par_chunks_mut(scale * bytes_per_line).enumerate().for_each(|(y, target_line_chunk)| {
            // locate the original image line
            let y_src = y;
            if y_src >= img_height {
                return;
            }
            // use local memory buffer to store the line data
            let mut line_buffer = vec![0u8; bytes_per_line];
            // draw the first target line
            // draw the white space at the beggining of the line
            line_buffer[0..start_x as usize * bytes_per_pixel].fill(0xFFu8);
            // calculate the offset in the image raw data
            let src_y_offset = y_src as usize * img_width as usize * 3;
            for x in 0..img_width {
                let src_x_offset = x as usize * 3;
                let pixel = &img_raw[src_y_offset + src_x_offset..src_y_offset + src_x_offset + 3];
                // repeat pixels scale times in the target line
                for j_repeat in 0..scale {
                    let dest_x_offset = (start_x + scale * x + j_repeat) as usize * bytes_per_pixel;
                    line_buffer[dest_x_offset..dest_x_offset + 3].copy_from_slice(pixel);
                }
            }
            // draw the white space at the end of the line
            line_buffer[end_x * bytes_per_pixel..].fill(0xFFu8);
        
            // copy the first target line to the rest of the lines in the chunk
            for j_repeat in 0..scale {
                let target_line = &mut target_line_chunk[j_repeat * bytes_per_line..(j_repeat + 1) * bytes_per_line];
                target_line.copy_from_slice(&line_buffer);
            }
        });
    }

    // draw the white space last, including the status bar area as background
    mmap[end_y * bytes_per_line..(fb_height - status_bar_height) * bytes_per_line].fill(0xFFu8);
    let elapsed = start.elapsed();
    println!("Copying to fb Elapsed: {:?}", elapsed);


    // Draw status bar
    let start_y = fb_height - status_bar_height;
    let y_offset = start_y  * bytes_per_line;
    // Prepare buffer according to framebuffer's pixel layout
    let start = std::time::Instant::now();
    let mut statusbar_buffer = vec![0xFFu8; status_bar_height * bytes_per_line];
    draw_statusbar_text(&mut statusbar_buffer, &fb_info, config, status_text);
    let statusbar_target = &mut mmap[y_offset..y_offset + status_bar_height * bytes_per_line];
    statusbar_target.copy_from_slice(&statusbar_buffer);

    let elapsed = start.elapsed();
    println!("Drawing status bar Elapsed: {:?}", elapsed);

    let start = std::time::Instant::now();
    println!("fb_idx: {}", fb_idx);
    // flush the data to the framebuffer
    mmap_unlock.flush_range(fb_idx * fb_size, fb_size).expect("Failed to flush data");
    // switch fb offset to the drawn frame
    {
        let fb_file = fb_state.fb_file.lock();
        let mut var_screen_info = fb_state.var_screen_info.lock();
        var_screen_info.yoffset = (fb_idx * fb_height) as u32;
        framebuffer::Framebuffer::pan_display(&fb_file, &var_screen_info).expect("Failed to pan display");
    }

    // switch fb idx
    fb_state.fb_idx.fetch_update(std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed, |x| Some((x + 1) % FB_NUM_BUFFERS)).unwrap();

    let elapsed = start.elapsed();
    println!("Flushing to fb Elapsed: {:?}", elapsed);
}