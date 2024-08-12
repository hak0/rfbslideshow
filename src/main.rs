use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::codecs::gif::GifDecoder;
use image::AnimationDecoder;
use image::{DynamicImage, GenericImageView, ImageFormat, Rgb};
use rusttype::{Font, Scale};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;
use jwalk::WalkDir;
use rayon::prelude::*;
use warp::Filter;



#[derive(Deserialize, Clone)]
struct Config {
    #[serde(skip)]
    config_path: String,
    folder_path: String,
    interval: u64,
    status_bar_font_path: String,
    status_bar_height: u32,
}

impl Config {
    fn new(config_path: &str) -> Self {
        let mut file = File::open(&config_path).expect("Failed to open config file");
        let mut content = String::new();
        file.read_to_string(&mut content)
            .expect("Failed to read config file");
        let mut new_config : Config = toml::from_str(&content).expect("Failed to parse config file");
        new_config.config_path = config_path.to_owned();
        new_config
    }

    fn load(&mut self) {
        let mut file = File::open(&self.config_path).expect("Failed to open config file");
        let mut content = String::new();
        file.read_to_string(&mut content)
            .expect("Failed to read config file");
        let new_config : Config = toml::from_str(&content).expect("Failed to parse config file");
        self.folder_path = new_config.folder_path;
        self.interval = new_config.interval;
        self.status_bar_font_path = new_config.status_bar_font_path;
        self.status_bar_height = new_config.status_bar_height;
    }
}


#[derive(Clone)]
struct FramebufferInfo {
    width: u32,
    height: u32,
    line_length: u32,     // number of bytes per line
    bits_per_pixel: u32,
}

#[derive(Clone)]
struct SlideshowState {
    paused: Arc<AtomicBool>,
    current_index: Arc<AtomicUsize>,
    image_list_len: Arc<AtomicUsize>,
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
    let config = Arc::new(RwLock::new(Config::new(&args[2])));
    let state = Arc::new(SlideshowState {
        paused: Arc::new(AtomicBool::new(false)),
        current_index: Arc::new(AtomicUsize::new(0)),
        image_list_len: Arc::new(AtomicUsize::new(0)),
    });
    let image_list = {
        let config_unlock = config.read().unwrap();
        Arc::new(RwLock::new(scan_folder(&config_unlock.folder_path, state.clone())))
    };

    let routes = setup_routes(state.clone(), image_list.clone(), config.clone());

    // Spawn a new thread to run the warp server
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            warp::serve(routes).run(([0, 0, 0, 0], 3030)).await;
        });
    });

    // Create a new thread to increase the index with configured interval
    let state_clone = state.clone();
    let config_clone = config.clone();

    std::thread::spawn(move || {
        loop {
            let sleep_interval = {
                let config = config_clone.read().unwrap();
                config.interval
            };
            std::thread::sleep(std::time::Duration::from_secs(sleep_interval));
            maybe_increase_index(&state_clone);
            println!("index: {}", state_clone.current_index.load(std::sync::atomic::Ordering::Relaxed));
        }
    });

    // Run the slideshow loop
    let state_clone = state.clone();
    let config_clone = config.clone();
    let image_list_clone = image_list.clone();
    std::thread::spawn(move || {
        run_slideshow(fb_info, state_clone, config_clone, image_list_clone);
    });

    // Keep the main function alive
    loop {
        std::thread::sleep(std::time::Duration::from_secs(60));
    }
}

fn setup_routes(state: Arc<SlideshowState>, image_list: Arc<RwLock<Vec<(PathBuf, ImageFormat)>>>, config: Arc<RwLock<Config>>) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    let state_filter = warp::any().map(move || Arc::clone(&state));
    let config_filter = warp::any().map(move || Arc::clone(&config));
    let image_list_filter = warp::any().map(move || Arc::clone(&image_list));

    // usage: http://serverhost/pause -> pause the slideshow
    let pause_route = warp::path("pause")
        .and(state_filter.clone())
        .map(|state: Arc<SlideshowState>| {
            state.paused.store(true, std::sync::atomic::Ordering::Relaxed);
            warp::reply::json(&"Paused")
        });

    // usage: http://serverhost/continue -> continue the slideshow
    let continue_route = warp::path("continue")
        .and(state_filter.clone())
        .map(|state: Arc<SlideshowState>| {
            state.paused.store(false, std::sync::atomic::Ordering::Relaxed);
            warp::reply::json(&"Continued")
        });

    // usage: http://serverhost/prev -> move to the previous image
    let prev_route = warp::path("prev")
        .and(state_filter.clone())
        .map(move |state: Arc<SlideshowState>| {
            let image_list_len = state.image_list_len.load(std::sync::atomic::Ordering::Relaxed);
            if image_list_len > 0  {
                let current_index_value = state.current_index.load(std::sync::atomic::Ordering::Relaxed);
                let new_index_value = (current_index_value + image_list_len- 1) % image_list_len;
                state.current_index.store(new_index_value, std::sync::atomic::Ordering::Relaxed);
            }
            warp::reply::json(&"Previous")
        });

    // usage: http://serverhost/next -> move to the next image
    let next_route = warp::path("next")
        .and(state_filter.clone())
        .map(move |state: Arc<SlideshowState>| {
            let image_list_len = state.image_list_len.load(std::sync::atomic::Ordering::Relaxed);
            if image_list_len > 0  {
                let current_index_value = state.current_index.load(std::sync::atomic::Ordering::Relaxed);
                let new_index_value = (current_index_value + image_list_len + 1) % image_list_len;
                state.current_index.store(new_index_value, std::sync::atomic::Ordering::Relaxed);
            }
            warp::reply::json(&"Next")
        });

    // usage: http://serverhost/rescan -> rescan the folder and play slideshow from the beginning
    let rescan_route = warp::path("rescan")
        .and(state_filter.clone())
        .and(config_filter.clone())
        .and(image_list_filter.clone())
        .map(move |state: Arc<SlideshowState>, config: Arc<RwLock<Config>>, image_list: Arc<RwLock<Vec<(PathBuf, ImageFormat)>>>| {
            // pause the slideshow first
            let pause_value = state.paused.load(std::sync::atomic::Ordering::Relaxed);
            state.paused.store(true, std::sync::atomic::Ordering::Relaxed);
            let config = config.read().unwrap();
            let mut image_list = image_list.write().unwrap();
            *image_list = scan_folder(&config.folder_path, state.clone());
            // restore pause state
            state.paused.store(pause_value, std::sync::atomic::Ordering::Relaxed);
            warp::reply::json(&"Rescan")
        });

    // usage: http://serverhost/reload -> reload config file
    let reload_route = warp::path("reload")
        .and(config_filter.clone())
        .map(move |config: Arc<RwLock<Config>>| {
            let mut config = config.write().unwrap();
            config.load();
            warp::reply::json(&"Config Reloaded")
        });

    // usage: http://serverhost/seek/5 -> move to 5th image
    let seek_route = warp::path("seek")
        .and(warp::path::param())
        .and(state_filter.clone())
        .map(move |index: i64, state: Arc<SlideshowState>| {
            // check if the index is within the range
            let image_list_len = state.image_list_len.load(std::sync::atomic::Ordering::Relaxed);
            if index >= image_list_len as i64|| index < 0{
                return warp::reply::json(&"Index out of range");
            }
            state.current_index.store(index as usize, std::sync::atomic::Ordering::Relaxed);
            warp::reply::json(&"Seeked")
        });

    // usage: http://serverhost/query -> get the current image name
    let query_route = warp::path("query")
        .and(state_filter.clone())
        .and(image_list_filter.clone())
        .map(move |state: Arc<SlideshowState>, image_list: Arc<RwLock<Vec<(PathBuf, ImageFormat)>>>| {
            let current_index = state.current_index.load(std::sync::atomic::Ordering::Relaxed);
            let image_list_unlocked = image_list.read().unwrap();
            let image_name = image_list_unlocked.get(current_index).unwrap().0.file_name().unwrap().to_str().unwrap();
            warp::reply::json(&image_name)
        });


    // usage: http://serverhost/query/5 -> get the 5th image name
    let query_with_id_route = warp::path!("query" / i64)
        .and(image_list_filter.clone())
        .map(move |index: i64, image_list: Arc<RwLock<Vec<(PathBuf, ImageFormat)>>>| {
            let image_list_unlocked = image_list.read().unwrap();
            // check if the index is within the range
            let image_list_len = image_list_unlocked.len();
            if index >= image_list_len as i64 || index < 0 {
                return warp::reply::json(&"Index out of range");
            }
            let image_name = image_list_unlocked.get(index as usize).unwrap().0.file_name().unwrap().to_str().unwrap();
            warp::reply::json(&image_name)
        });

    warp::get().and(
        pause_route
            .or(continue_route)
            .or(prev_route)
            .or(next_route)
            .or(rescan_route)
            .or(reload_route)
            .or(seek_route)
            .or(query_with_id_route)
            .or(query_route)
    )
}

fn maybe_increase_index(state_clone: &Arc<SlideshowState>) {
    let current_index = state_clone.current_index.load(std::sync::atomic::Ordering::Relaxed);
    let image_list_len = state_clone.image_list_len.load(std::sync::atomic::Ordering::Relaxed);
    let paused = state_clone.paused.load(std::sync::atomic::Ordering::Relaxed);
    if image_list_len != 0 && !paused {
        let new_index = (current_index + 1 + image_list_len) % image_list_len;
        state_clone.current_index.store(new_index, std::sync::atomic::Ordering::Relaxed);
    }
}

fn run_slideshow(
    fb_info: FramebufferInfo,
    state: Arc<SlideshowState>,
    config: Arc<RwLock<Config>>, 
    image_list: Arc<RwLock<Vec<(PathBuf, ImageFormat)>>>,
) {
    let mut current_index_last_time = None;
    let mut terminal_signal = Arc::new(AtomicBool::new(false));
    loop {
        let current_index = state.current_index.load(std::sync::atomic::Ordering::Relaxed);
        let image_list_len = state.image_list_len.load(std::sync::atomic::Ordering::Relaxed);

        // if image_list is empty, sleep for 1 second
        if image_list_len == 0 {
            std::thread::sleep(std::time::Duration::from_secs(1));
            continue;
        }

        // if the index is the same as the index last time, sleep for 1 second
        if current_index_last_time == Some(current_index) {
            std::thread::sleep(std::time::Duration::from_secs(1));
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
        let (path, format) = {
            let image_list_unlocked = image_list.read().unwrap();
            image_list_unlocked.get(current_index).unwrap().clone()
        };

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
        line_length: stride, 
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

fn display_image(img: &DynamicImage, resize_algorithm : ResizeAlg ,config : Arc<RwLock<Config>>, fb_info: &FramebufferInfo, status_text: &str) {
    // read the image with the specified format
    let start = std::time::Instant::now();
    let img = resize_image(&img, fb_info, config.clone(), resize_algorithm);
    let elapsed = start.elapsed();
    println!("Resizing Elapsed: {:?}", elapsed);


    // Combine the resized image and the status bar
    write_to_framebuffer(&img, &status_text, fb_info, config.clone());
}

fn resize_image(img: &DynamicImage, fb_info: &FramebufferInfo, config: Arc<RwLock<Config>>, algorithm: ResizeAlg) -> DynamicImage {
    let ratio = img.width() as f32 / img.height() as f32;
    let new_height = {
        let config_unlock = config.read().unwrap();
        fb_info.height - config_unlock.status_bar_height
    };
    let img = if img.height() != new_height {
        // Create Resizer instance and resize source image
        // into buffer of destination image
        let mut resizer = Resizer::new();
        let mut dest_img = if img.color() == image::ColorType::Rgba8 {
            DynamicImage::new_rgba8((new_height as f32 * ratio) as u32, new_height as u32)
        } else {
            DynamicImage::new_rgb8((new_height as f32 * ratio) as u32, new_height as u32)
        };
        resizer.resize(img, &mut dest_img, 
            //&ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))).unwrap();
            &ResizeOptions::new().resize_alg(algorithm)).unwrap();
        dest_img
    } else {
        img.clone()
    };
    img
}

fn draw_statusbar_text(raw_mem: &mut Vec<u8>, fb_info: &FramebufferInfo,config: Arc<RwLock<Config>>, status_text: &str) {
    // Create a new image for the status bar only
    let bytes_per_pixel = (fb_info.bits_per_pixel / 8) as i32;

    // Read the font data from the file specified in the config
    // check if font exists
    let config_unlock = config.read().unwrap();
    let font_path_str = config_unlock.status_bar_font_path.trim().to_owned();
    let status_bar_height = config_unlock.status_bar_height.clone();
    drop(config_unlock);
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
    println!("offset: {:?}", offset);

    // Draw the text on the status bar, right aligned
    for glyph in font.layout(status_text, scale, offset) {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|x, y, v| {
                let x = x as i32 + bounding_box.min.x;
                let y = y as i32 + bounding_box.min.y;
                if x >= 0 && x < fb_info.width as i32 && y >= 0 && y < status_bar_height as i32 {
                    let x_offset = (x * bytes_per_pixel) as usize;
                    let y_offset = y as usize * fb_info.line_length as usize;
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

fn display_single_img(path: &Path, format: ImageFormat, config: Arc<RwLock<Config>>, fb_info: &FramebufferInfo, status_text: &str, terminal_signal: Arc<AtomicBool>) {
    if let Ok(img) = image::open(path) {
        println!("image format: {:?}", format);
        println!("image color: {:?}", img.color());
        // if terminal_signal is true, skip loading image and early return
        if terminal_signal.load(std::sync::atomic::Ordering::Relaxed) { return; }

        display_image(&img, ResizeAlg::Convolution(FilterType::Lanczos3), config, fb_info, status_text)
    }
}

fn display_gif(path: &Path, config: Arc<RwLock<Config>>, fb_info: &FramebufferInfo, status_text: &str, terminal_signal: Arc<AtomicBool>) {
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

fn write_to_framebuffer(img: &DynamicImage, status_text: &str, fb_info: &FramebufferInfo, config: Arc<RwLock<Config>>) {
    //time it
    let framebuffer_path = "/dev/fb0";
    let bytes_per_pixel = (fb_info.bits_per_pixel / 8) as usize;

    let framebuffer = Arc::new(RwLock::new(io::BufWriter::new(File::create(framebuffer_path).expect("Failed to open framebuffer"))));

    let status_bar_height = {
        let config_unlock = config.read().unwrap();
        config_unlock.status_bar_height.clone()
    };

    let start = std::time::Instant::now();
    // Draw Image line-by-line
    let (img_width, img_height) = img.dimensions();
    // To draw the image at 2/3 of the screen width, calculate the starting position
    let x_offset_ratio = 2.0 / 3.0;
    let start_x = ((fb_info.width as f32 * x_offset_ratio) - (img_width as f32 * x_offset_ratio)) as u32;
    let start_y = (fb_info.height - img_height - status_bar_height) / 2;

    (0..img_height.min(fb_info.height)).into_par_iter().for_each(|y| {
        let y_offset = (start_y + y) as usize * fb_info.line_length as usize;
        let mut line_buffer = vec![0xFFu8; fb_info.line_length as usize];
        for x in 0..img_width.min(fb_info.width) {
            let pixel = img.get_pixel(x, y);
            let x_offset = (start_x + x) as usize * bytes_per_pixel;
            // reverse rgb to bgr
            line_buffer[x_offset + 0] = pixel.0[2];
            line_buffer[x_offset + 1] = pixel.0[1];
            line_buffer[x_offset + 2] = pixel.0[0];
        }
        let mut framebuffer_unlocked = framebuffer.write().unwrap();
        framebuffer_unlocked.seek(SeekFrom::Start(y_offset as u64)).expect("Failed to seek in framebuffer");
        framebuffer_unlocked.write(&line_buffer).expect("Failed to write to framebuffer");
    });
    let elapsed = start.elapsed();
    println!("Copying to fb Elapsed: {:?}", elapsed);

    // Draw status bar
    // Prepare buffer according to framebuffer's pixel layout
    let start = std::time::Instant::now();
    let mut statusbar_buffer = vec![0xFFu8; (status_bar_height * fb_info.line_length) as usize];
    draw_statusbar_text(&mut statusbar_buffer, fb_info, config, status_text);
    let start_y = fb_info.height - status_bar_height - 16; // the raspberrypi have some error at the screen corner, move the status bar up a bit
    let y_offset = start_y as usize * fb_info.line_length as usize;
    {
        let mut framebuffer_unlocked = framebuffer.write().unwrap();
        framebuffer_unlocked.seek(SeekFrom::Start(y_offset as u64)).expect("Failed to seek in framebuffer");
        framebuffer_unlocked.write(&statusbar_buffer).expect("Failed to write to framebuffer");
    }
    let elapsed = start.elapsed();
    println!("Drawing status bar Elapsed: {:?}", elapsed);

}