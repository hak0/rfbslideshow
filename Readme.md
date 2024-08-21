# RFbSlideshow

A lightweight Rust slideshow program based on frame buffer. No need for x11 or wayland. I use it on my Raspberrypi.

* R = Rust
* Fb = framebuffer
* slideshow = this is a slideshow program

## Usage

```
rfbslideshow -c config.toml
```

## Features

* minimum dependency and low memory usage
* start from tty text mode, rather than in an X11/wayland environment
* doesn't require a GPU driver(no need for /dev/dri/card)
* an HTTP server for remote control
* double buffering to prevent screen tearing

## Limitations

So far it is customized to fit my own needs, so it may look strange and rough. If you are interested to use it, and want to add features, please feel free to submit a PR or create a new issue. I will check in my free time.

* no "random order" feature
* supported formats: webp, gif, png, jpeg, bmp
* no authentication for web server
* the image is not centered horizontally; instead, it is aligned at the 2/3 position horizontally. 🤓
* Gif images only support integer scaling for speed

## Story

I use a raspberrypi 3b to play slideshow, and managed to get 4k output according to this [stack exchange](https://raspberrypi.stackexchange.com/questions/44089/can-raspberry-pi-3-do-4k-video). However, since the raspberrypi 3b's graphic driver doesn't support resolution over 1080p, there will be no hardware acceleration and no `/dev/dri/card` graphic driver. The slideshow programs like `sxiv` and `imvr` are slow in this situation, especially when handling gif animations.

So I decided to start from the bare minimum, that is, writing to the framebuffer directly, and control it from the LAN using http requests.

## Capabilitiy Requirements

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

## Remote Control

A web server will be started to accept remote commands. By default the URL is `http://ip:3030`. Users can use cURL or other http clients to control the slideshow.

* http://serverhost/pause -> pause the slideshow loop
* http://serverhost/resume -> resume the slideshow loop
* http://serverhost/prev -> move to the previous image
* http://serverhost/next -> move to the next image
* http://serverhost/rescan -> rescan the folder and restart the slideshow
* http://serverhost/reload -> reload the config file
* http://serverhost/seek/{index} -> move to the specified index
* http://serverhost/query -> query the current image name
* http://serverhost/clearfolder -> clear the folder
* http://serverhost/upload/{image_name} -> upload an image to the folder
