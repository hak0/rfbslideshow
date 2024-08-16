# RFbSlideshow

A lightweight Rust slideshow program based on frame buffer. No need for x11 or wayland.

## Capabilities

During the start, the program rfbslideshow will need to set kd_mode into KD_GRAPHICS, just like any other x11 programs. After that the tty text output will be shadowed.

This operation require `CAP_SYS_TTY_CONFIG` capability. To run program without `sudo`, we need to add this capability to the program.

```bash
# set the capabliity
sudo setcap 'CAP_SYS_TTY_CONFIG=eip' ./rfbslideshow
# check the capability
getcap ./rfbslideshow
# the result should looks like: `./rfbslideshow cap_sys_tty_config=eip`
```

if still not working, try adding user into the tty group

```bash
sudo usermod -a -G tty $USER
sudo usermod -a -G dialout $USER
```

## TODO

- [x] ~~prefetch buffer in memory(size can be configured). Only save resized image and status bar text. Maybe useful for gif~~the performance is good enough without prefetching
- [x] a simple http server to receive remote control command. Maybe in the main thread, while the image display is spawned in a worker thread. Commands include rescan/prev/next/pause/continue