"""Look edits shared by the live PC studio and the file exporter."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class EditState:
    mode: str = "blur"
    blur: int = 28
    color_r: int = 20
    color_g: int = 24
    color_b: int = 28
    bg_scale: float = 1.0
    bg_x: float = 0.0
    bg_y: float = 0.0
    mirror: bool = True
    exposure: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    temperature: float = 0.0
    sharpness: float = 0.15
    beauty: float = 0.0
    eye: bool = False
    eye_strength: float = 0.7
    autoframe: bool = False
    tightness: float = 0.55
    key_on: bool = True
    keylight: float = 0.35
    denoise_on: bool = True
    video_denoise: float = 0.3
    vignette: float = 0.0
    lower_on: bool = False
    lower_name: str = ""
    lower_title: str = ""
    logo_on: bool = False
    logo_scale: float = 0.16
    logo_x: float = 0.04
    logo_y: float = 0.04
    quality: bool = False
    downsample: float = 0.0

    def color_bgr(self) -> tuple[int, int, int]:
        return (self.color_b, self.color_g, self.color_r)

    def to_json(self) -> dict:
        return asdict(self)


def grade(frame: np.ndarray, state: EditState) -> np.ndarray:
    img = frame.astype(np.float32)
    img *= 2.0 ** float(np.clip(state.exposure, -1.5, 1.5))
    img = (img - 128.0) * float(np.clip(state.contrast, 0.5, 1.8)) + 128.0
    img[:, :, 0] += state.temperature * -14.0
    img[:, :, 2] += state.temperature * 14.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    if abs(state.saturation - 1.0) > 0.01:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * state.saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if state.sharpness > 0.01:
        soft = cv2.GaussianBlur(img, (0, 0), 1.1)
        img = np.clip(img.astype(np.float32) + state.sharpness * 1.4 * (img.astype(np.float32) - soft.astype(np.float32)), 0, 255).astype(np.uint8)
    return img


def beauty(frame: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.01:
        return frame
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(2, w // 2), max(2, h // 2)), interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, 5, 28, 28)
    smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    m = np.clip(mask * amount, 0, 1)[..., None]
    return (smooth.astype(np.float32) * m + frame.astype(np.float32) * (1.0 - m)).astype(np.uint8)


def place_still(image: np.ndarray, width: int, height: int, scale: float, ox: float, oy: float) -> np.ndarray:
    ih, iw = image.shape[:2]
    cover = max(width / iw, height / ih) * max(0.4, scale)
    nw, nh = max(2, int(iw * cover)), max(2, int(ih * cover))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    x = int((width - nw) / 2 + ox * width)
    y = int((height - nh) / 2 + oy * height)
    canvas = np.zeros((height, width, 3), np.uint8)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + nw), min(height, y + nh)
    if x1 <= x0 or y1 <= y0:
        return canvas
    canvas[y0:y1, x0:x1] = resized[y0 - y : y1 - y, x0 - x : x1 - x]
    return canvas


def lower_third(frame: np.ndarray, name: str, title: str) -> np.ndarray:
    if not name and not title:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = 48, h - 148, min(w - 48, 640), h - 48
    plate = out.copy()
    cv2.rectangle(plate, (x0 + 3, y0 + 4), (x1 + 3, y1 + 4), (0, 0, 0), -1)
    cv2.rectangle(plate, (x0, y0), (x1, y1), (16, 18, 22), -1)
    cv2.rectangle(plate, (x0, y0), (x0 + 6, y1), (196, 132, 42), -1)
    out = cv2.addWeighted(plate, 0.88, out, 0.12, 0)
    cv2.putText(out, name, (x0 + 22, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (244, 244, 244), 2, cv2.LINE_AA)
    cv2.putText(out, title.upper(), (x0 + 22, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (176, 186, 196), 1, cv2.LINE_AA)
    return out


def overlay_logo(frame: np.ndarray, logo: np.ndarray, scale: float, ox: float, oy: float) -> np.ndarray:
    h, w = frame.shape[:2]
    target = max(24, int(w * scale))
    lh, lw = logo.shape[:2]
    nh = max(8, int(target * lh / lw))
    resized = cv2.resize(logo, (target, nh), interpolation=cv2.INTER_AREA)
    x, y = int(ox * w), int(oy * h)
    x1, y1 = min(w, x + target), min(h, y + nh)
    if x >= w or y >= h or x1 <= x or y1 <= y:
        return frame
    patch = resized[: y1 - y, : x1 - x]
    if patch.shape[2] == 4:
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        bgr = patch[:, :, :3].astype(np.float32)
        dest = frame[y:y1, x:x1].astype(np.float32)
        frame[y:y1, x:x1] = (bgr * alpha + dest * (1.0 - alpha)).astype(np.uint8)
    else:
        frame[y:y1, x:x1] = patch
    return frame


PRESETS = {
    "Meeting": dict(mode="blur", blur=26, beauty=0.25, key_on=True, keylight=0.32, denoise_on=True, video_denoise=0.35, vignette=0.0, eye=False, autoframe=False, sharpness=0.12),
    "Stream": dict(mode="studio", blur=20, beauty=0.15, key_on=True, keylight=0.4, vignette=0.35, lower_on=True, sharpness=0.22, eye=False, autoframe=True),
    "Podcast": dict(mode="image", beauty=0.3, key_on=True, keylight=0.45, denoise_on=True, video_denoise=0.4, vignette=0.2, sharpness=0.1),
    "Clean plate": dict(mode="remove", beauty=0.0, key_on=False, vignette=0.0, lower_on=False, logo_on=False, sharpness=0.05),
}
