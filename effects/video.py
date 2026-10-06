"""Camera effects that correspond to Broadcast's video stack.

Eye contact here is geometric iris redirection from face landmarks. It is a
real gaze shift, not NVIDIA's closed Maxine network, and it will not invent
a plausible eye at extreme angles. Strength is clamped for that reason.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

LEFT_OUTER, LEFT_INNER, LEFT_IRIS = 33, 133, 468
RIGHT_OUTER, RIGHT_INNER, RIGHT_IRIS = 263, 362, 473
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)


class FaceTracker:
    def __init__(self) -> None:
        self.mesh = None
        self.backend = "haar"
        self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        try:
            import mediapipe as mp

            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.backend = "mediapipe"
            print("Face tracker: MediaPipe Face Mesh with iris landmarks")
        except Exception as exc:
            print(f"Face tracker: OpenCV Haar ({exc})")

    def landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.mesh is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        pts = result.multi_face_landmarks[0].landmark
        out = np.empty((len(pts), 2), dtype=np.float32)
        for i, p in enumerate(pts):
            out[i, 0] = p.x * w
            out[i, 1] = p.y * h
        return out

    def face_box(self, frame_bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (320, max(1, int(320 * h / w))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.15, 4, minSize=(36, 36))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        sx, sy = w / 320.0, h / small.shape[0]
        return int(x * sx), int(y * sy), int(fw * sx), int(fh * sy)

    def close(self) -> None:
        if self.mesh is not None:
            self.mesh.close()


def _shift_patch(frame: np.ndarray, center: np.ndarray, delta: np.ndarray, radius: float) -> None:
    h, w = frame.shape[:2]
    r = int(max(8, radius))
    cx, cy = int(center[0]), int(center[1])
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r), min(h, cy + r)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    patch = frame[y0:y1, x0:x1]
    matrix = np.float32([[1, 0, delta[0]], [0, 1, delta[1]]])
    warped = cv2.warpAffine(patch, matrix, (patch.shape[1], patch.shape[0]), borderMode=cv2.BORDER_REFLECT)
    yy, xx = np.ogrid[: patch.shape[0], : patch.shape[1]]
    mask = ((xx - (cx - x0)) ** 2 + (yy - (cy - y0)) ** 2) <= (r * 0.72) ** 2
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), r * 0.18)[..., None]
    frame[y0:y1, x0:x1] = (warped * mask + patch * (1.0 - mask)).astype(np.uint8)


class EyeContact:
    def apply(self, frame: np.ndarray, tracker: FaceTracker, strength: float) -> np.ndarray:
        if strength <= 0.01:
            return frame
        pts = tracker.landmarks(frame)
        if pts is not None and len(pts) > LEFT_IRIS:
            out = frame.copy()
            for outer, inner, iris in (
                (LEFT_OUTER, LEFT_INNER, LEFT_IRIS),
                (RIGHT_OUTER, RIGHT_INNER, RIGHT_IRIS),
            ):
                eye_w = float(np.linalg.norm(pts[inner] - pts[outer]))
                desired = (pts[outer] + pts[inner]) * 0.5
                delta = (desired - pts[iris]) * (0.35 + 0.65 * strength)
                limit = 0.32 * eye_w
                delta = np.clip(delta, -limit, limit)
                if float(np.linalg.norm(delta)) < 0.6:
                    continue
                _shift_patch(out, pts[iris], delta, radius=eye_w * 0.55)
            return out
        return self._haar_fallback(frame, tracker, strength)

    def _haar_fallback(self, frame: np.ndarray, tracker: FaceTracker, strength: float) -> np.ndarray:
        box = tracker.face_box(frame)
        if box is None:
            return frame
        x, y, w, h = box
        roi = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        eyes = tracker._eye.detectMultiScale(gray, 1.12, 6, minSize=(16, 12))
        if len(eyes) == 0:
            return frame
        out = frame.copy()
        for ex, ey, ew, eh in eyes[:2]:
            eye = gray[ey : ey + eh, ex : ex + ew]
            _, _, _, min_loc = cv2.minMaxLoc(cv2.GaussianBlur(eye, (0, 0), 1.2))
            pupil = np.array([x + ex + min_loc[0], y + ey + min_loc[1]], dtype=np.float32)
            center = np.array([x + ex + ew * 0.5, y + ey + eh * 0.48], dtype=np.float32)
            delta = (center - pupil) * strength
            _shift_patch(out, pupil, delta, radius=max(ew, eh) * 0.45)
        return out


class AutoFrame:
    def __init__(self) -> None:
        self.cx, self.cy, self.zoom = 0.5, 0.42, 1.0
        self.ready = False

    def apply(self, frame: np.ndarray, tracker: FaceTracker, enabled: bool, tightness: float) -> np.ndarray:
        if not enabled:
            return frame
        h, w = frame.shape[:2]
        target = None
        pts = tracker.landmarks(frame)
        if pts is not None:
            xs, ys = pts[:, 0], pts[:, 1]
            target = (float(xs.mean() / w), float(np.percentile(ys, 18) / h), float(np.clip((ys.max() - ys.min()) / h, 0.05, 0.9)))
        else:
            box = tracker.face_box(frame)
            if box is not None:
                x, y, fw, fh = box
                target = ((x + fw * 0.5) / w, (y + fh * 0.35) / h, fh / h)
        if target is None and not self.ready:
            return frame
        if target is not None:
            tcx, tcy, face_h = target
            span = max(0.18, face_h * (2.3 - tightness))
            tzoom = float(np.clip(0.62 / span, 1.0, 1.7))
            a = 0.08
            self.cx = self.cx * (1 - a) + tcx * a
            self.cy = self.cy * (1 - a) + tcy * a
            self.zoom = self.zoom * (1 - a) + tzoom * a
            self.ready = True
        zw, zh = w / self.zoom, h / self.zoom
        x0 = float(np.clip(self.cx * w - zw / 2, 0, w - zw))
        y0 = float(np.clip(self.cy * h - zh * 0.42, 0, h - zh))
        crop = frame[int(y0) : int(y0 + zh), int(x0) : int(x0 + zw)]
        if crop.size == 0:
            return frame
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


class VideoDenoise:
    def __init__(self) -> None:
        self.prev: Optional[np.ndarray] = None

    def apply(self, frame: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0.01:
            self.prev = frame
            return frame
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        sigma = 0.6 + strength * 1.8
        ycc[:, :, 1] = cv2.GaussianBlur(ycc[:, :, 1], (0, 0), sigma)
        ycc[:, :, 2] = cv2.GaussianBlur(ycc[:, :, 2], (0, 0), sigma)
        spatial = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        if self.prev is None or self.prev.shape != frame.shape:
            self.prev = spatial
            return spatial
        diff = cv2.absdiff(frame, self.prev).mean(axis=2)
        still = (diff < 14).astype(np.float32)[..., None]
        mix = still * (0.25 + 0.45 * strength)
        out = np.clip(spatial * (1.0 - mix) + self.prev.astype(np.float32) * mix, 0, 255).astype(np.uint8)
        self.prev = out
        return out


class KeyLight:
    def apply(self, frame_bgr: np.ndarray, pha: np.ndarray, pts: Optional[np.ndarray], amount: float) -> np.ndarray:
        if amount <= 0.01:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        image = frame_bgr.astype(np.float32) / 255.0
        if pts is not None:
            mask = np.zeros((h, w), np.float32)
            oval = pts[list(FACE_OVAL)].astype(np.int32)
            cv2.fillConvexPoly(mask, oval, 1.0)
            mask = cv2.GaussianBlur(mask, (0, 0), 9)
        else:
            mask = pha
        person = image[mask > 0.2]
        if person.size == 0:
            return frame_bgr
        luma = person.mean()
        gain = 1.0 + amount * float(np.clip(0.58 - luma, 0.0, 0.5))
        lifted = np.clip(image * gain, 0, 1)
        # Front-top key: brighter forehead, less under the chin.
        yy = np.linspace(1.08, 0.94, h, dtype=np.float32)[:, None]
        lifted = np.clip(lifted * yy[..., None], 0, 1)
        m = mask[..., None]
        out = image * (1.0 - m) + lifted * m
        return (out * 255.0).astype(np.uint8)


def vignette(frame: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.01:
        return frame
    h, w = frame.shape[:2]
    y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    radius = np.sqrt(x * x * 1.15 + y * y)
    mask = np.clip(1.0 - amount * np.clip(radius - 0.55, 0, 1) ** 1.4, 0.45, 1.0)
    return np.clip(frame.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def fast_blur(image: np.ndarray, strength: int) -> np.ndarray:
    strength = max(1, int(strength))
    h, w = image.shape[:2]
    scale = 8 if strength < 40 else 6
    small = cv2.resize(image, (max(2, w // scale), max(2, h // scale)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, strength / 3.5))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def studio_backdrop(width: int, height: int, anchor_x: float) -> np.ndarray:
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    pool = np.exp(-((x - anchor_x) ** 2) / 0.08 - ((y - 0.42) ** 2) / 0.18)
    shade = 18 + 28 * (1 - y) + 70 * pool
    canvas = np.dstack([shade * 0.85, shade * 0.9, shade]).astype(np.uint8)
    return canvas


def portrait_bokeh(frame: np.ndarray, pha: np.ndarray, strength: float) -> np.ndarray:
    """Blur falls off with distance from the person. The matte is the depth cue."""
    if strength <= 0.01:
        return frame
    person = cv2.GaussianBlur((pha > 0.45).astype(np.float32), (0, 0), 18)
    far = np.clip(1.0 - person, 0, 1)
    near = np.clip(person * 1.35, 0, 1)[..., None]
    mid = fast_blur(frame, int(10 + strength * 18))
    deep = fast_blur(frame, int(24 + strength * 48))
    field = mid.astype(np.float32) * (1.0 - far[..., None]) + deep.astype(np.float32) * far[..., None]
    return np.clip(frame.astype(np.float32) * near + field * (1.0 - near), 0, 255).astype(np.uint8)


def belong(fgr: np.ndarray, pha: np.ndarray, background: np.ndarray, amount: float) -> np.ndarray:
    """Nudge the person toward the scene's color so the cut does not look pasted on."""
    if amount <= 0.01:
        return fgr
    outside = pha < 0.18
    inside = pha > 0.62
    if int(outside.sum()) < 40 or int(inside.sum()) < 40:
        return fgr
    bg = cv2.cvtColor(background, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    scene = bg[outside].mean(axis=0)
    person = fgr[inside].mean(axis=0)
    gain = np.clip((scene + 0.04) / (person + 0.04), 0.86, 1.16)
    mixed = fgr * ((1.0 - amount) + amount * gain)
    return np.clip(mixed, 0.0, 1.0)


def composite(fgr: np.ndarray, pha: np.ndarray, background: np.ndarray) -> np.ndarray:
    alpha = pha[..., None]
    fg = fgr * 255.0 if fgr.dtype != np.uint8 else fgr.astype(np.float32)
    if fgr.dtype == np.uint8:
        fg = fgr.astype(np.float32)
    out = fg * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def despill(fgr: np.ndarray, pha: np.ndarray, amount: float) -> np.ndarray:
    """Pull green and blue fringe off the hair edge. fgr is RGB float 0-1."""
    if amount <= 0.01:
        return fgr
    edge = (pha > 0.04) & (pha < 0.92)
    if not np.any(edge):
        return fgr
    rgb = fgr.copy()
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cap = np.maximum(r, b)
    green = np.clip(g - cap, 0, 1)
    blue = np.clip(b - np.maximum(r, g * 0.85), 0, 1)
    mix = edge.astype(np.float32) * amount
    rgb[..., 1] = g - green * mix
    rgb[..., 2] = b - blue * mix * 0.65
    return np.clip(rgb, 0, 1)
