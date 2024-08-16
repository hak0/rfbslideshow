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
use memmap2::MmapOptions;



// thread-safe version of config
struct Config {
    config_path: RwLock<String>,
    folder_path: RwLock<String>,
    interval: AtomicU32,
    status_bar_font_path: RwLock<String>,
    status_bar_height: AtomicU32,
    http_server_port: AtomicU16,
}

#[derive(Deserialize)]
struct ConfigPrimitive {
    folder_path: String,
    interval: u32,
    status_bar_font_path: String,
    status_bar_height: u32,
    http_server_port: u16,
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
    }
}
#[derive(Clone)]
struct FramebufferInfo {
    width: u32,
    height: u32,
    bytes_per_line: u32,
    bits_per_pixel: u32,
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 || args[1] != "-c" {
        eprintln!("Usage: {} -c <config.toml>", args[0]);
        std::process::exit(1);
    }

    let fb_info = get_framebuffer_info().expect("Failed to get framebuffer info");

    // initialize data containers
    // lock order: config -> image_list
    let config = Arc::new(Config::new(&args[2]));
    let state = Arc::new(SlideshowState {
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
    let state_clone = state.clone();
    let config_clone = config.clone();
    std::thread::spawn(move || {
        let rx_clone = Arc::new(Mutex::new(rx));
        rouille::start_server(format!("0.0.0.0:{}", port), move |request| {
            setup_routes(request, state_clone.clone(), config_clone.clone(), rx_clone.clone())
        });
    });

    // Create a new thread to increase the index with configured interval
    let state_clone = state.clone();
    let config_clone = config.clone();
    std::thread::spawn(move || {
        loop {
            let sleep_interval = config_clone.interval.load(std::sync::atomic::Ordering::Relaxed) as u64;
            std::thread::sleep(std::time::Duration::from_secs(sleep_interval));
            maybe_increase_index(&state_clone);
            println!("index: {}", state_clone.current_index.load(std::sync::atomic::Ordering::Relaxed));
        }
    });

    // Run the slideshow loop
    let state_clone = state.clone();
    let config_clone = config.clone();
    std::thread::spawn(move || {
        run_slideshow(fb_info, state_clone, config_clone, tx);
    });

    // Keep the main function alive
    loop {
        std::thread::sleep(std::time::Duration::from_secs(60));
    }
}

// currently there is no lock or mutex to protect the opreations
// the user needs to take care by themselves.
// e.g. don't rescan when uploading, or don't continue/prev/next/seek when rescnaning
fn setup_routes(request: &Request, state: Arc<SlideshowState>, config: Arc<Config>, rx: Arc<Mutex<Receiver<String>>>) -> Response {
    rouille::router!(request,
        // usage: http://serverhost/pause -> pause the slideshow loop
        (GET) (/pause) => {
            state.paused.store(true, Ordering::Relaxed);
            Response::json(&"Paused")
        },
        // usage: http://serverhost/resume -> resume the slideshow loop
        (GET) (/resume) => {
            state.paused.store(false, Ordering::Relaxed);
            // awake the slideshow loop
            state.slideshow_condvar.1.notify_one();
            Response::json(&"Resumed")
        },
        // usage: http://serverhost/prev -> move to the previous image
        (GET) (/prev) => {
            let image_list_len = state.image_list_len.load(Ordering::Relaxed);
            if image_list_len > 0 {
                let current_index_value = state.current_index.load(Ordering::Relaxed);
                let new_index_value = (current_index_value + image_list_len - 1) % image_list_len;
                state.current_index.store(new_index_value, Ordering::Relaxed);
                // awake the slideshow loop
                state.slideshow_condvar.1.notify_one();
            }
            Response::json(&"Previous")
        },
        // usage: http://serverhost/next -> move to the next image
        (GET) (/next) => {
            let image_list_len = state.image_list_len.load(Ordering::Relaxed);
            if image_list_len > 0 {
                let current_index_value = state.current_index.load(Ordering::Relaxed);
                let new_index_value = (current_index_value + 1) % image_list_len;
                state.current_index.store(new_index_value, Ordering::Relaxed);
                // awake the slideshow loop
                state.slideshow_condvar.1.notify_one();
            }
            Response::json(&"Next")
        },
        // usage: http://serverhost/rescan -> rescan the folder for new images
        (GET) (/rescan) => {
            state.need_rescan.store(true, Ordering::Relaxed);
            // awake the slideshow loop
            state.slideshow_condvar.1.notify_one();
            let start = std::time::Instant::now();
            while state.need_rescan.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if start.elapsed().as_secs() > 180 {
                    state.need_rescan.store(false, Ordering::Relaxed);
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
            let image_list_len = state.image_list_len.load(Ordering::Relaxed);
            if index >= image_list_len {
                return Response::json(&"Index out of range");
            }
            state.current_index.store(index, Ordering::Relaxed);
            // awake the slideshow loop
            state.slideshow_condvar.1.notify_one();
            Response::json(&"Seeked")
        },
        // usage: http://serverhost/query -> query the current image name
        (GET) (/query) => {
            state.need_query.store(true, Ordering::Relaxed);
            // awake the slideshow loop
            state.slideshow_condvar.1.notify_one();
            let image_name = rx.lock().recv().unwrap();
            Response::json(&format!("Image name: {}", image_name))
        },
        // usage: http://serverhost/clearfolder -> clear the folder
        (GET) (/clearfolder) => {
            state.paused.store(true, Ordering::Relaxed);
            state.image_list_len.store(0, Ordering::Relaxed);
            state.current_index.store(0, Ordering::Relaxed);

            let folder_path = config.folder_path.read().unwrap().clone();
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

            state.paused.store(true, Ordering::Relaxed);
            let folder_path = config.folder_path.read().unwrap().clone();
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
fn maybe_increase_index(state: &Arc<SlideshowState>) {
    let current_index = state.current_index.load(std::sync::atomic::Ordering::Relaxed);
    let image_list_len = state.image_list_len.load(std::sync::atomic::Ordering::Relaxed);
    let paused = state.paused.load(std::sync::atomic::Ordering::Relaxed);
    if image_list_len != 0 && !paused {
        let new_index = (current_index + 1 + image_list_len) % image_list_len;
        state.current_index.store(new_index, std::sync::atomic::Ordering::Relaxed);
    }
}

// main "event loop" for the slideshow
fn run_slideshow(
    fb_info: FramebufferInfo,
    state: Arc<SlideshowState>,
    config: Arc<Config>, 
    tx: std::sync::mpsc::Sender<String>
) {
    // private image list inside run_slideshow, won't share it
    let mut image_list: Vec<(PathBuf, ImageFormat)> = Vec::new();
    
    let mut current_index_last_time = None;
    let mut terminal_signal = Arc::new(AtomicBool::new(false));
    loop {
        let current_index = state.current_index.load(std::sync::atomic::Ordering::Relaxed);
        let image_list_len = state.image_list_len.load(std::sync::atomic::Ordering::Relaxed);
        let needs_rescan = state.need_rescan.load(std::sync::atomic::Ordering::Relaxed);
        let needs_query = state.need_query.load(std::sync::atomic::Ordering::Relaxed);
        let slideshow_condvar_pair = state.slideshow_condvar.clone();

        // response to the query request
        if needs_query {
            state.need_query.store(false, std::sync::atomic::Ordering::Relaxed);
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
            state.need_rescan.store(false, std::sync::atomic::Ordering::Relaxed);
            image_list = scan_folder(&scan_folder_path, state.clone());
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
            state.current_index.store(0, std::sync::atomic::Ordering::Relaxed);
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
                    let cloned_fb_info = fb_info.clone();
                    let cloned_terminal_signal = terminal_signal.clone();
                    std::thread::spawn(move || {
                        display_gif(&path, cloned_config, &cloned_fb_info, &bar_text, cloned_terminal_signal);
                    });
                    Some(())
                },
                ImageFormat::WebP => {
                    let cloned_config = config.clone();
                    let cloned_fb_info = fb_info.clone();
                    let cloned_terminal_signal = terminal_signal.clone();
                    std::thread::spawn(move || {
                        display_single_img(&path, format, cloned_config, &cloned_fb_info, &bar_text, cloned_terminal_signal);
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
        maybe_increase_index(&state)
    }
}

// use condvar to sleep for a while, that can be awaken by condvar
// maybe shorter due to Spurious Wakeups, but it's ok
fn sleep_slideshow(slideshow_condvar_pair: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>, duration: Duration) {
    let (lock, condvar) = &*slideshow_condvar_pair;
    let mut _guard = lock.lock().unwrap();
    let _ = condvar.wait_timeout(_guard, duration);
}

fn scan_folder(scan_folder_path: &str, state: Arc<SlideshowState>) -> Vec<(PathBuf, ImageFormat)> {
    // pause the slideshow first
    let pause_value = state.paused.load(std::sync::atomic::Ordering::Relaxed);
    state.paused.store(true, std::sync::atomic::Ordering::Relaxed);

    let image_list : Vec<(PathBuf, ImageFormat)> = WalkDir::new(scan_folder_path)
        .into_iter()
        .filter_map(|entry_result| entry_result.ok()) // Handle errors by filtering out invalid entries
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| {
            let path = entry.path();
            get_image_format(&path).map(|format| (path.to_path_buf(), format))
        })
        .collect();
    state.image_list_len.store(image_list.len(), std::sync::atomic::Ordering::Relaxed);
    state.current_index.store(0, std::sync::atomic::Ordering::Relaxed);
    state.paused.store(pause_value, std::sync::atomic::Ordering::Relaxed);
    image_list
}

fn get_framebuffer_info() -> io::Result<FramebufferInfo> {
    // check framebuffer size
    // the file content looks like: 1024, 768
    let mut fb_var_screeninfo_size = File::open("/sys/class/graphics/fb0/virtual_size")?;
    let mut content = String::new();
    fb_var_screeninfo_size.read_to_string(&mut content)?;
    let dimensions: Vec<&str> = content.trim().split(',').collect();
    if dimensions.len() != 2 {
        eprintln!("Unexpected format in virtual_size file");
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid format"));
    }

    let width: u32 = match dimensions[0].trim().parse() {
        Ok(w) => w,
        Err(_) => {
            eprintln!("Failed to parse width");
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid width"));
        }
    };

    let height: u32 = match dimensions[1].trim().parse() {
        Ok(h) => h,
        Err(_) => {
            eprintln!("Failed to parse height");
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid height"));
        }
    };
    // file content looks like: 32
    let mut fb_var_screeninfo_bits_per_pixel = File::open("/sys/class/graphics/fb0/bits_per_pixel")?;
    let bits_per_pixel: u32 = {
        let mut content = String::new();
        fb_var_screeninfo_bits_per_pixel.read_to_string(&mut content)?;
        content.trim().parse().unwrap()
    };
    let mut fb_var_screeninfo_stride = File::open("/sys/class/graphics/fb0/stride")?;
    let stride: u32 = {
        let mut content = String::new();
        fb_var_screeninfo_stride.read_to_string(&mut content)?;
        content.trim().parse().unwrap()
    };


    Ok(FramebufferInfo {
        width: width,
        height: height,
        bytes_per_line: stride, 
        bits_per_pixel: bits_per_pixel,
    })
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

fn display_image(img: &DynamicImage, resize_algorithm : ResizeAlg , config : Arc<Config>, fb_info: &FramebufferInfo, status_text: &str) {
    // read the image with the specified format
    let start = std::time::Instant::now();
    if resize_algorithm != ResizeAlg::Nearest {
        let img = resize_image(&img, fb_info, config.clone(), resize_algorithm);
        let elapsed = start.elapsed();
        println!("Resizing Elapsed: {:?}", elapsed);
        // Combine the resized image and the status bar
        write_to_framebuffer(&img, &status_text, fb_info, config.clone(), 1);
    } else {
        println!("Skip Resizing");
        let scale_x = fb_info.width as f32 / img.width() as f32;
        let scale_y = fb_info.height as f32 / img.height() as f32;
        let scale = scale_x.min(scale_y).floor() as u32;
        // skip resizing and do integer-factor resize in the fb write
        write_to_framebuffer(&img, &status_text, fb_info, config.clone(), scale);
    }
}

fn resize_image(img: &DynamicImage, fb_info: &FramebufferInfo, config: Arc<Config>, algorithm: ResizeAlg) -> DynamicImage {
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

fn draw_statusbar_text(raw_mem: &mut [u8], fb_info: &FramebufferInfo,config: Arc<Config>, status_text: &str) {
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

fn display_single_img(path: &Path, format: ImageFormat, config: Arc<Config>, fb_info: &FramebufferInfo, status_text: &str, terminal_signal: Arc<AtomicBool>) {
    if let Ok(img) = image::open(path) {
        println!("image format: {:?}", format);
        println!("image color: {:?}", img.color());
        // if terminal_signal is true, skip loading image and early return
        if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

        // convert the image to bgr format
        let img = convert_rgba_or_rgb_to_bgr(img);


        // if terminal_signal is true, skip loading image and early return
        if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

        display_image(&img, ResizeAlg::Convolution(FilterType::Mitchell), config, fb_info, status_text)
    }
}

fn display_gif(path: &Path, config: Arc<Config>, fb_info: &FramebufferInfo, status_text: &str, terminal_signal: Arc<AtomicBool>) {
    if let Ok(file) = File::open(path) {
        let file_in = io::BufReader::new(file);
        let decoder = GifDecoder::new(file_in).unwrap();
        let mut frames = decoder.into_frames();
        let mut img_vec = Vec::new();
        let mut img_vec_idx = 0;

        loop {
            // if terminal_signal is true, skip loading image and early return
            if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

            // decode the next frame
            let start = std::time::Instant::now();
            let (img, delay) = if let Some(Ok(frame)) = frames.next() {
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

            display_image(&img, ResizeAlg::Nearest, Arc::clone(&config), fb_info, status_text);

            // if terminal_signal is true, skip loading image and early return
            if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

            // Sleep for the delay time
            let elapsed = start.elapsed();
            if Duration::from(delay) > elapsed {
                thread::sleep(Duration::from(delay) - elapsed);
            }
        }
    }
}

fn write_to_framebuffer(img: &DynamicImage, status_text: &str, fb_info: &FramebufferInfo, config: Arc<Config>, scale: u32) {
    // Time it
    let framebuffer_path = "/dev/fb0";
    let bytes_per_pixel = (fb_info.bits_per_pixel / 8) as usize;
    let bytes_per_line = fb_info.bytes_per_line as usize;
    let fb_height = fb_info.height as usize;
    let fb_width = fb_info.width as usize;

    let path = PathBuf::from(framebuffer_path);
    let file = OpenOptions::new().read(true).write(true).create(true).open(path).expect("Failed to open fb file");
    let framebuffer_size = bytes_per_line as usize * fb_height as usize;
    let mut mmap = unsafe { MmapOptions::new().len(framebuffer_size).map_mut(&file).expect("Failed to mmap fb") };

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
    let end_y = (start_y + img_height * scale).min(fb_height - status_bar_height - 16); // The raspberrypi have some error at the screen corner, move the status bar up a bit


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

    // draw the white space last, including the status bar area as background
    mmap[end_y * bytes_per_line..(fb_height - status_bar_height - 16) * bytes_per_line].fill(0xFFu8); // The raspberrypi have some error at the screen corner, move the status bar up a bit
    let elapsed = start.elapsed();
    println!("Copying to fb Elapsed: {:?}", elapsed);


    // Draw status bar
    let start_y = fb_height - status_bar_height - 16; // The raspberrypi have some error at the screen corner, move the status bar up a bit
    let y_offset = start_y  * bytes_per_line;
    // Prepare buffer according to framebuffer's pixel layout
    let start = std::time::Instant::now();
    let mut statusbar_buffer = vec![0xFFu8; status_bar_height * bytes_per_line];
    draw_statusbar_text(&mut statusbar_buffer, fb_info, config, status_text);
    let statusbar_target = &mut mmap[y_offset..y_offset + status_bar_height * bytes_per_line];
    statusbar_target.copy_from_slice(&statusbar_buffer);

    let elapsed = start.elapsed();
    println!("Drawing status bar Elapsed: {:?}", elapsed);
    // flush the data to the framebuffer
    //mmap.flush_async_range(0, framebuffer_size).expect("Failed to flush data");
    //mmap.flush().expect("Failed to flush data");
}