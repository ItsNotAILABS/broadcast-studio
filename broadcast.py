#!/usr/bin/env python3
"""GPU webcam stack aimed at the NVIDIA Broadcast effect list.

Video: recurrent alpha matting, blur / replace / remove, auto frame,
video noise removal, eye contact, virtual key light, vignette.
Audio: noise removal, room echo reduction, studio voice, routed to a
virtual cable so meeting apps can select it as the microphone.

RVM is GPL-3.0. This program is GPL-3.0.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from effects.audio import AudioEngine
from effects.matting import Matter, auto_downsample, pick_device, soften_alpha
from effects.video import (
    AutoFrame,
    EyeContact,
    FaceTracker,
    KeyLight,
    VideoDenoise,
    composite,
    fast_blur,
    studio_backdrop,
    vignette,
)

ROOT = Path(__file__).resolve().parent
SNAPSHOT_DIR = ROOT / "snapshots"
SETTINGS = ROOT / "settings.json"


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    system = platform.system()
    if system == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif system == "Darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Backgrounds:
    def __init__(self, image_path: str, video_path: str) -> None:
        self.image = None
        self.video = None
        if image_path:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f"Could not read background image: {image_path}")
            self.image = img
        if video_path:
            self.video = cv2.VideoCapture(video_path)
            if not self.video.isOpened():
                raise SystemExit(f"Could not read background video: {video_path}")

    def frame(self, mode: str, width: int, height: int, source: np.ndarray, blur: int, color: tuple[int, int, int], anchor_x: float) -> np.ndarray:
        if mode == "blur":
            return fast_blur(source, blur)
        if mode == "color":
            canvas = np.empty((height, width, 3), np.uint8)
            canvas[:] = color
            return canvas
        if mode == "green":
            canvas = np.empty((height, width, 3), np.uint8)
            canvas[:] = (0, 177, 64)
            return canvas
        if mode == "remove":
            return np.zeros((height, width, 3), np.uint8)
        if mode == "studio":
            return studio_backdrop(width, height, anchor_x)
        if mode == "image" and self.image is not None:
            return cv2.resize(self.image, (width, height), interpolation=cv2.INTER_AREA)
        if mode == "video" and self.video is not None:
            ok, frame = self.video.read()
            if not ok:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.video.read()
            if ok and frame is not None:
                return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return fast_blur(source, blur)


class VirtualCam:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.cam = None
        self.error = ""
        try:
            import pyvirtualcam
        except ImportError:
            self.error = "pyvirtualcam is not installed"
            return
        try:
            self.cam = pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.RGB)
            print(f"Virtual camera: {self.cam.device}")
        except Exception as exc:
            self.error = str(exc)
            print(f"Virtual camera unavailable ({exc}). Preview still runs.")

    @property
    def label(self) -> str:
        if self.cam is not None:
            return self.cam.device
        return self.error or "unavailable"

    def send(self, bgr: np.ndarray) -> None:
        if self.cam is None:
            return
        self.cam.send(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.cam.sleep_until_next_frame()

    def close(self) -> None:
        if self.cam is not None:
            self.cam.close()
            self.cam = None


def parse_color(text: str) -> tuple[int, int, int]:
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be R,G,B")
    r, g, b = parts
    return (b, g, r)


def draw_hud(frame, lines: list[str]) -> None:
    y = 22
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
        y += 20


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GPU Broadcast-style webcam and microphone")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--variant", choices=("mobilenetv3", "resnet50"), default="mobilenetv3")
    p.add_argument("--mode", choices=("blur", "color", "image", "video", "green", "remove", "studio", "mask"), default="blur")
    p.add_argument("--blur", type=int, default=28)
    p.add_argument("--color", type=parse_color, default="20,24,28")
    p.add_argument("--background", default="")
    p.add_argument("--background-video", default="")
    p.add_argument("--downsample", type=float, default=0.0)
    p.add_argument("--quality", action="store_true")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--virtual-cam", action="store_true")
    p.add_argument("--autoframe", action="store_true")
    p.add_argument("--eye-contact", action="store_true")
    p.add_argument("--keylight", type=float, default=0.35)
    p.add_argument("--video-denoise", type=float, default=0.35)
    p.add_argument("--vignette", type=float, default=0.0)
    p.add_argument("--audio", action="store_true", help="start the microphone effect chain")
    p.add_argument("--deepfilter", action="store_true", help="use DeepFilterNet if installed")
    p.add_argument("--mic-in", type=int, default=None)
    p.add_argument("--mic-out", type=int, default=None, help="virtual cable / BlackHole / VB-Cable device index")
    p.add_argument("--list-audio", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    audio = AudioEngine()
    if args.list_audio:
        if not audio.available():
            raise SystemExit("Install sounddevice to list audio devices: pip install sounddevice")
        print("\n".join(audio.list_devices()))
        return 0

    device = pick_device(args.device)
    cap = open_camera(args.camera, args.width, args.height, args.fps)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit("Camera opened but returned no frames")
    height, width = frame.shape[:2]
    matter = Matter(args.variant, device)
    backs = Backgrounds(args.background, args.background_video)
    tracker = FaceTracker()
    eye = EyeContact()
    framer = AutoFrame()
    denoiser = VideoDenoise()
    lighter = KeyLight()
    vcam = VirtualCam(width, height, args.fps) if args.virtual_cam else None
    if args.audio:
        if not audio.available():
            print("Audio effects need sounddevice. Video will still run.")
        else:
            audio.noise_on = True
            audio.echo_on = True
            audio.studio_on = True
            audio.start(args.mic_in, args.mic_out, deepfilter=args.deepfilter)

    state = {
        "mode": args.mode,
        "blur": args.blur,
        "color": args.color,
        "downsample": args.downsample,
        "quality": args.quality,
        "autoframe": args.autoframe,
        "eye": args.eye_contact,
        "eye_strength": 0.7,
        "keylight": args.keylight,
        "key_on": True,
        "video_denoise": args.video_denoise,
        "denoise_on": args.video_denoise > 0,
        "vignette": args.vignette,
        "hud": True,
        "tightness": 0.55,
    }
    ds = state["downsample"] or auto_downsample(height, width, state["quality"])
    print(f"Camera {width}x{height} on {device}  downsample {ds:.3f}  face {tracker.backend}")
    matter.matte(frame, ds)
    matter.reset()
    window = "Broadcast stack"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(width, 1280), min(height, 720))
    last = time.perf_counter()
    fps_smooth = 0.0
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            if state["denoise_on"]:
                frame = denoiser.apply(frame, state["video_denoise"])
            if state["eye"]:
                frame = eye.apply(frame, tracker, state["eye_strength"])
            frame = framer.apply(frame, tracker, state["autoframe"], state["tightness"])
            ds = state["downsample"] or auto_downsample(height, width, state["quality"])
            fgr, pha = matter.matte(frame, ds)
            pha = soften_alpha(pha)
            anchor = float(np.clip((pha * np.linspace(0, 1, width)).sum() / max(pha.sum(), 1e-6), 0.2, 0.8))
            if state["mode"] == "mask":
                out = cv2.cvtColor((pha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                if state["key_on"]:
                    pts = tracker.landmarks(frame)
                    lit = lighter.apply(cv2.cvtColor((fgr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), pha, pts, state["keylight"])
                    fgr = cv2.cvtColor(lit, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                background = backs.frame(state["mode"], width, height, frame, state["blur"], state["color"], anchor)
                out = composite(fgr, pha, background)
            out = vignette(out, state["vignette"])
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            fps_smooth = inst if fps_smooth == 0 else fps_smooth * 0.9 + inst * 0.1
            preview = out.copy()
            if state["hud"]:
                draw_hud(preview, [
                    f"{args.variant}  {device.type}  {fps_smooth:4.1f} fps  ds {ds:.2f}  face {tracker.backend}",
                    f"bg {state['mode']}  blur {state['blur']}  frame {state['autoframe']}  eye {state['eye']}  key {state['key_on']}  denoise {state['denoise_on']}",
                    f"cam {vcam.label if vcam else 'off'}   mic {audio.engine_name} {audio.device_label}",
                    "b blur c color i image o video g green x remove u studio k mask",
                    "f frame  e eye  l key  d denoise  v vignette  n vcam  a audio  r reset  q quit",
                ])
            cv2.imshow(window, preview)
            if vcam is not None:
                vcam.send(out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("b"):
                state["mode"] = "blur"
            elif key == ord("c"):
                state["mode"] = "color"
            elif key == ord("i"):
                state["mode"] = "image"
            elif key == ord("o"):
                state["mode"] = "video"
            elif key == ord("g"):
                state["mode"] = "green"
            elif key == ord("x"):
                state["mode"] = "remove"
            elif key == ord("u"):
                state["mode"] = "studio"
            elif key == ord("k"):
                state["mode"] = "mask"
            elif key == ord("["):
                state["blur"] = max(1, state["blur"] - 4)
            elif key == ord("]"):
                state["blur"] = min(80, state["blur"] + 4)
            elif key == ord("-"):
                state["downsample"] = max(0.15, (state["downsample"] or ds) - 0.05)
            elif key == ord("="):
                state["downsample"] = min(1.0, (state["downsample"] or ds) + 0.05)
            elif key == ord("f"):
                state["autoframe"] = not state["autoframe"]
            elif key == ord("e"):
                state["eye"] = not state["eye"]
            elif key == ord("l"):
                state["key_on"] = not state["key_on"]
            elif key == ord("d"):
                state["denoise_on"] = not state["denoise_on"]
            elif key == ord("v"):
                state["vignette"] = 0.0 if state["vignette"] > 0 else 0.45
            elif key == ord("h"):
                state["hud"] = not state["hud"]
            elif key == ord("r"):
                matter.reset()
            elif key == ord("s"):
                path = SNAPSHOT_DIR / f"snap-{time.strftime('%Y%m%d-%H%M%S')}.png"
                cv2.imwrite(str(path), out)
                print(f"Saved {path}")
            elif key == ord("n"):
                if vcam is None:
                    vcam = VirtualCam(width, height, args.fps)
                else:
                    vcam.close()
                    vcam = None
            elif key == ord("a"):
                if audio.stream is None and audio.available():
                    audio.start(args.mic_in, args.mic_out, deepfilter=args.deepfilter)
                    audio.noise_on = True
                    audio.echo_on = True
                    audio.studio_on = True
                else:
                    audio.stop()
    finally:
        SETTINGS.write_text(json.dumps({k: state[k] for k in state if k != "color"}, indent=2))
        cap.release()
        tracker.close()
        audio.stop()
        if vcam is not None:
            vcam.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
