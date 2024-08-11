use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::codecs::gif::GifDecoder;
use image::{AnimationDecoder, Pixel};
use image::{DynamicImage, GenericImageView, ImageFormat, Rgb};
use rusttype::{Font, Scale};
use serde::Deserialize;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use jwalk::WalkDir;
use rayon::prelude::*;

#[derive(Deserialize)]
struct Config {
    scan_folder: String,
    interval: u64,
    status_bar_font_path: String,
    status_bar_height: u32,
}

struct FramebufferInfo {
    width: u32,
    height: u32,
    line_length: u32,     // number of bytes per line
    bits_per_pixel: u32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 || args[1] != "-c" {
        eprintln!("Usage: {} -c <config.toml>", args[0]);
        std::process::exit(1);
    }

    let config = load_config(&args[2]);
    let fb_info = get_framebuffer_info().expect("Failed to get framebuffer info");

    // Get a list of images with valid formats
    loop {
        let mut image_queue = scan_folder(&config.scan_folder)
        .into_iter().map(|(path, format)| (path, format, true)).collect::<Vec<_>>();
        while !image_queue.is_empty() {
            let total = image_queue.len();
            let mut failed = 0;
            for i in 0..total {
                let (path, format, is_valid) = &mut image_queue[i];
                println!("{:?}", path);
                if *is_valid && path.exists() && path.is_file() {
                    let bar_text = format!("[{}/{}] {}", i, total, path.file_name().unwrap().to_str().unwrap());
                    if Some(()) == match format {
                        ImageFormat::Gif => Some(display_gif(path, &config, &fb_info, &bar_text)),
                        ImageFormat::WebP => Some(display_single_img(path, *format, &config, &fb_info, &bar_text)),
                        _ => None,
                    } {
                        // success
                        continue
                    }
                }
                // failed
                *is_valid = false;
                failed += 1
            }
            if failed == total {
                break;
            }
        }
    }
}

fn load_config(filename: &str) -> Config {
    let mut file = File::open(filename).expect("Failed to open config file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Failed to read config file");
    toml::from_str(&content).expect("Failed to parse config file")
}

fn scan_folder(scan_folder: &str) -> Vec<(PathBuf, ImageFormat)> {
    WalkDir::new(scan_folder)
        .into_iter()
        .filter_map(|entry_result| entry_result.ok()) // Handle errors by filtering out invalid entries
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| {
            let path = entry.path();
            get_image_format(&path).map(|format| (path.to_path_buf(), format))
        })
        .collect()
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

fn display_image(img: &DynamicImage, resize_algorithm : ResizeAlg ,config: &Config, fb_info: &FramebufferInfo, status_text: &str) {
    // read the image with the specified format
    let start = std::time::Instant::now();
    let img = resize_image(&img, fb_info, config, resize_algorithm);
    let elapsed = start.elapsed();
    println!("Resizing Elapsed: {:?}", elapsed);


    // Combine the resized image and the status bar
    write_to_framebuffer(&img, &status_text, fb_info, config);
}

fn resize_image(img: &DynamicImage, fb_info: &FramebufferInfo, config: &Config, algorithm: ResizeAlg) -> DynamicImage {
    let ratio = img.width() as f32 / img.height() as f32;
    let new_height = fb_info.height - config.status_bar_height;
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

fn draw_statusbar_text(raw_mem: &mut Vec<u8>, fb_info: &FramebufferInfo,config: &Config, status_text: &str) {
    // Create a new image for the status bar only
    //let mut bar_image = image::RgbImage::from_pixel(fb_info.width, config.status_bar_height, bar_color);
    let bytes_per_pixel = (fb_info.bits_per_pixel / 8) as i32;

    // Read the font data from the file specified in the config
    // check if font exists
    let font_path_str = config.status_bar_font_path.trim();
    let font_path = Path::new(&font_path_str);
    println!("statusbar font: \"{}\"", font_path_str);
    if !Path::new(font_path).exists() {
        println!("font does not exist");
    }
    let font_data = fs::read(&config.status_bar_font_path).expect("Failed to read font file");
    let font = Font::try_from_bytes(&font_data).expect("Failed to load font");
    let font_size = (config.status_bar_height - 4) as f32;
    let scale = Scale { x: font_size, y: font_size };
    let bar_color = Rgb([255, 255, 255]); // White color for background
    let text_color = Rgb([0, 0, 0]); // Black color for text

    // Draw the text on the status bar
    let v_metrics = font.v_metrics(scale);
    let offset = rusttype::point(10.0, v_metrics.ascent);

    for glyph in font.layout(status_text, scale, offset) {
        if let Some(bounding_box) = glyph.pixel_bounding_box() {
            glyph.draw(|x, y, v| {
                let x = x as i32 + bounding_box.min.x;
                let y = y as i32 + bounding_box.min.y;
                if x >= 0 && x < fb_info.width as i32 && y >= 0 && y < config.status_bar_height as i32 {
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

fn display_single_img(path: &Path, format: ImageFormat, config: &Config, fb_info: &FramebufferInfo, status_text: &str) {
    let start = std::time::Instant::now();
    if let Ok(img) = image::open(path) {
        println!("image format: {:?}", format);
        println!("image color: {:?}", img.color());
        display_image(&img, ResizeAlg::Convolution(FilterType::Lanczos3), config, fb_info, status_text)
    }
    let elapsed = start.elapsed();
    if Duration::from_secs(config.interval) > elapsed {
        // wait for a duration before displaying the next image
        thread::sleep(Duration::from_secs(config.interval) - elapsed);
    }
}

fn display_gif(path: &Path, config: &Config, fb_info: &FramebufferInfo, status_text: &str) {
    // read the image with the specified format
    if let Ok(file) = File::open(path) {
        let file_in = io::BufReader::new(file);
        // Configure the decoder such that it will expand the image to RGB.
        let decoder = GifDecoder::new(file_in).unwrap();
        let frames = decoder.into_frames();
        let start_total = std::time::Instant::now();
        let mut img_vec = Vec::new();
        for frame in frames {
            let start = std::time::Instant::now();
            let frame = frame.unwrap();
            let delay = frame.delay();
            let img = DynamicImage::ImageRgba8(frame.into_buffer());
            let elapsed = start.elapsed();
            println!("GIF Frame Decoding Elapsed: {:?}", elapsed);
            display_image(&img, ResizeAlg::Nearest, config, fb_info, status_text);
            // cache the image
            img_vec.push((img, delay));
            // check if the gif should be exited and move to next image
            let total_elapsed = start_total.elapsed();
            if total_elapsed > Duration::from_secs(config.interval) {
                return;
            }
            // check if need to delay between frames
            let elapsed = start.elapsed();
            if Duration::from(delay) > elapsed{
                thread::sleep(Duration::from(delay) - elapsed);
            }
        }
        loop {
            for (img, delay) in &img_vec {
                let start = std::time::Instant::now();
                display_image(&img, ResizeAlg::Nearest, config, fb_info, status_text);
                // check if the gif should be exited and move to next image
                let total_elapsed = start_total.elapsed();
                if total_elapsed > Duration::from_secs(config.interval) {
                    return;
                }
                // check if the gif should be exited and move to next image
                let total_elapsed = start_total.elapsed();
                if total_elapsed > Duration::from_secs(config.interval) {
                    return;
                }
                // check if need to delay between frames
                let elapsed = start.elapsed();
                if Duration::from(*delay) > elapsed{
                    thread::sleep(Duration::from(*delay) - elapsed);
                }
            }
        }
    }
}

fn write_to_framebuffer(img: &DynamicImage, status_text: &str, fb_info: &FramebufferInfo, config: &Config) {
    //time it
    let framebuffer_path = "/dev/fb0";
    let bytes_per_pixel = (fb_info.bits_per_pixel / 8) as usize;

    let framebuffer = Arc::new(Mutex::new(io::BufWriter::new(File::create(framebuffer_path).expect("Failed to open framebuffer"))));

    // Draw status bar
    // Prepare buffer according to framebuffer's pixel layout
    let start = std::time::Instant::now();
    let mut statusbar_buffer = vec![0xFFu8; (config.status_bar_height * fb_info.line_length) as usize];
    draw_statusbar_text(&mut statusbar_buffer, fb_info, config, status_text);
    framebuffer.lock().unwrap().write(&statusbar_buffer).expect("Failed to write to framebuffer");
    let elapsed = start.elapsed();
    println!("Drawing status bar Elapsed: {:?}", elapsed);

    let start = std::time::Instant::now();
    // Draw Image line-by-line
    let (img_width, img_height) = img.dimensions();
    // To draw the image at 2/3 of the screen width, calculate the starting position
    let x_offset_ratio = 2.0 / 3.0;
    let start_x = ((fb_info.width as f32 * x_offset_ratio) - (img_width as f32 * x_offset_ratio)) as u32;
    let start_y = (fb_info.height - img_height - config.status_bar_height) / 2 + config.status_bar_height;

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
        let mut framebuffer_unlocked = framebuffer.lock().unwrap();
        framebuffer_unlocked.seek(SeekFrom::Start(y_offset as u64)).expect("Failed to seek in framebuffer");
        framebuffer_unlocked.write(&line_buffer).expect("Failed to write to framebuffer");
    });
    //

    let elapsed = start.elapsed();
    println!("Copying to fb Elapsed: {:?}", elapsed);
}