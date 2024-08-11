# RFbSlideshow

A lightweight Rust slideshow program based on frame buffer. No need for x11 or wayland.

## TODO

- [ ] prefetch buffer in memory(size can be configured). Only save resized image and status bar text. Maybe useful for gif
- [x] a simple http server to receive remote control command. Maybe in the main thread, while the image display is spawned in a worker thread. Commands include rescan/prev/next/pause/continue