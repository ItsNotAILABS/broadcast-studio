#!/usr/bin/env python3
"""PC studio. Embedded preview, inspector, and one edit path for
preview, virtual camera, recording, and file export.
"""

from __future__ import annotations

import json
import platform
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np

from effects.audio import AudioEngine
from effects.edit import PRESETS, EditState, beauty, grade, lower_third, overlay_logo, place_still
from effects.matting import Matter, auto_downsample, guide_alpha, pick_device, soften_alpha
from effects.video import AutoFrame, EyeContact, FaceTracker, KeyLight, VideoDenoise, belong, composite, despill, fast_blur, portrait_bokeh, studio_backdrop, vignette

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "exports"
SETTINGS = ROOT / "studio-settings.json"
BG = "#101114"
PANEL = "#181b21"
INK = "#f2f4f8"
MUTED = "#8d95a3"
LINE = "#2a2e37"
ACCENT = "#e0b15a"
REC = "#d4534a"


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class Session:
    def __init__(self) -> None:
        self.state = EditState()
        self.lock = threading.Lock()
        self.running = False
        self.writer = None
        self.vcam = None
        self.background = None
        self.background_video = None
        self.logo = None
        self.source_path = ""
        self.device = pick_device("auto")
        self.matter = None
        self.tracker = FaceTracker()
        self.eye = EyeContact()
        self.framer = AutoFrame()
        self.denoiser = VideoDenoise()
        self.lighter = KeyLight()
        self.variant = "mobilenetv3"
        self.latest: np.ndarray | None = None
        self.fps = 0.0
        self.width = 1280
        self.height = 720
        self.audio = AudioEngine()

    def ensure_model(self) -> None:
        if self.matter is None or self.matter.variant != self.variant:
            self.matter = Matter(self.variant, self.device)
            self.matter.reset()

    def render(self, frame: np.ndarray) -> np.ndarray:
        with self.lock:
            state = EditState(**self.state.to_json())
            logo = None if self.logo is None else self.logo.copy()
        if state.mirror:
            frame = cv2.flip(frame, 1)
        if state.denoise_on:
            frame = self.denoiser.apply(frame, state.video_denoise)
        if state.eye:
            frame = self.eye.apply(frame, self.tracker, state.eye_strength)
        frame = self.framer.apply(frame, self.tracker, state.autoframe, state.tightness)
        h, w = frame.shape[:2]
        ds = state.downsample or auto_downsample(h, w, state.quality)
        fgr, pha = self.matter.matte(frame, ds)
        pha = soften_alpha(pha)
        pha = guide_alpha(pha, frame)
        fgr = despill(fgr, pha, state.spill)
        if state.mode == "blur":
            plate = portrait_bokeh(frame, pha, state.bokeh if state.bokeh > 0.01 else state.blur / 80.0)
        else:
            plate = self._background(state, frame, 0.5)
        if state.belong_on:
            fgr = belong(fgr, pha, plate, state.belong)
        if state.beauty > 0.01:
            person = cv2.cvtColor((fgr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            person = beauty(person, pha, state.beauty)
            fgr = cv2.cvtColor(person, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if state.key_on:
            pts = self.tracker.landmarks(frame)
            lit = self.lighter.apply(cv2.cvtColor((fgr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), pha, pts, state.keylight)
            fgr = cv2.cvtColor(lit, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if state.mode == "mask":
            out = cv2.cvtColor((pha * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            anchor = float(np.clip((pha * np.linspace(0, 1, w)).sum() / max(pha.sum(), 1e-6), 0.2, 0.8))
            if state.mode != "blur":
                plate = self._background(state, frame, anchor)
            out = composite(fgr, pha, plate)
        out = grade(out, state)
        out = vignette(out, state.vignette)
        if state.lower_on:
            out = lower_third(out, state.lower_name, state.lower_title)
        if state.logo_on and logo is not None:
            out = overlay_logo(out, logo, state.logo_scale, state.logo_x, state.logo_y)
        return out

    def _background(self, state: EditState, frame: np.ndarray, anchor: float) -> np.ndarray:
        h, w = frame.shape[:2]
        if state.mode == "blur":
            return fast_blur(frame, state.blur)
        if state.mode == "color":
            canvas = np.empty((h, w, 3), np.uint8)
            canvas[:] = state.color_bgr()
            return canvas
        if state.mode == "green":
            canvas = np.empty((h, w, 3), np.uint8)
            canvas[:] = (0, 177, 64)
            return canvas
        if state.mode == "remove":
            return np.zeros((h, w, 3), np.uint8)
        if state.mode == "studio":
            return studio_backdrop(w, h, anchor)
        if state.mode == "image" and self.background is not None:
            return place_still(self.background, w, h, state.bg_scale, state.bg_x, state.bg_y)
        if state.mode == "video":
            still = self._next_video(w, h)
            if still is not None:
                return place_still(still, w, h, state.bg_scale, state.bg_x, state.bg_y)
        return fast_blur(frame, state.blur)

    def _next_video(self, width: int, height: int):
        cap = self.background_video
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame


class Studio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Broadcast Studio")
        self.geometry("1280x800")
        self.minsize(1100, 680)
        self.configure(bg=BG)
        self.session = Session()
        self.photo = None
        self.worker = None
        self.export_thread = None
        self._style()
        self._build()
        self._load()
        self._bind_keys()
        self.after(40, self._paint)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=INK, fieldbackground=PANEL, bordercolor=LINE)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=INK)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("TButton", background="#232833", foreground=INK, padding=(10, 7), borderwidth=0)
        style.configure("Accent.TButton", background=ACCENT, foreground="#1a1408", padding=(12, 7))
        style.configure("Rec.TButton", background="#3a2424", foreground="#f0c2be", padding=(10, 7))
        style.map("TButton", background=[("active", "#2e3440")])
        style.map("Accent.TButton", background=[("active", "#edd08a")])
        style.configure("TNotebook", background=PANEL, borderwidth=0)
        style.configure("TNotebook.Tab", background="#14171c", foreground=MUTED, padding=(14, 8))
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        style.configure("TCheckbutton", background=PANEL, foreground=INK)
        style.configure("TRadiobutton", background=PANEL, foreground=INK)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("Card.TLabel", background=PANEL, foreground=INK)
        style.configure("CardMuted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Horizontal.TScale", background=PANEL)

    def _build(self) -> None:
        bar = tk.Frame(self, bg=BG, height=58)
        bar.pack(fill="x", padx=18, pady=(12, 6))
        mark = tk.Frame(bar, bg=BG)
        mark.pack(side="left")
        tk.Label(mark, text="STUDIO", bg=BG, fg=ACCENT, font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(mark, text="PROGRAM", bg=BG, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
        actions = tk.Frame(bar, bg=BG)
        actions.pack(side="right")
        ttk.Button(actions, text="Start", style="Accent.TButton", command=self.start_live).pack(side="left", padx=3)
        ttk.Button(actions, text="Stop", command=self.stop).pack(side="left", padx=3)
        ttk.Button(actions, text="Virtual cam", command=self.toggle_vcam).pack(side="left", padx=3)
        self.rec_button = ttk.Button(actions, text="Record", style="Rec.TButton", command=self.toggle_record)
        self.rec_button.pack(side="left", padx=3)
        ttk.Button(actions, text="Snap", command=self.snapshot).pack(side="left", padx=3)
        ttk.Button(actions, text="Open", command=self.open_video).pack(side="left", padx=3)
        ttk.Button(actions, text="Export", command=self.export_video).pack(side="left", padx=3)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=4)
        monitor = tk.Frame(body, bg="#07080b", highlightbackground=LINE, highlightthickness=1)
        monitor.pack(side="left", fill="both", expand=True)
        self.stage = tk.Canvas(monitor, bg="#07080b", highlightthickness=0)
        self.stage.pack(fill="both", expand=True, padx=10, pady=10)
        side = tk.Frame(body, bg=PANEL, width=360, highlightbackground=LINE, highlightthickness=1)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)
        tk.Label(side, text="INSPECTOR", bg=PANEL, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=(10, 0))
        book = ttk.Notebook(side)
        book.pack(fill="both", expand=True, padx=8, pady=8)
        self._background_tab(book)
        self._look_tab(book)
        self._face_tab(book)
        self._overlay_tab(book)
        self._audio_tab(book)

        foot = tk.Frame(self, bg="#0c0e12", height=36)
        foot.pack(fill="x", padx=18, pady=(6, 12))
        self.status = tk.StringVar(value=f"{self.session.device.type.upper()} READY")
        tk.Label(foot, textvariable=self.status, bg="#0c0e12", fg=MUTED, font=("Segoe UI", 9)).pack(side="left", padx=10, pady=8)
        cam = tk.Frame(foot, bg="#0c0e12")
        cam.pack(side="right", padx=10)
        tk.Label(cam, text="CAMERA", bg="#0c0e12", fg=MUTED, font=("Segoe UI", 8)).pack(side="left")
        self.camera = tk.IntVar(value=0)
        ttk.Spinbox(cam, from_=0, to=8, textvariable=self.camera, width=4).pack(side="left", padx=6)
        tk.Label(cam, text="FORMAT", bg="#0c0e12", fg=MUTED, font=("Segoe UI", 8)).pack(side="left", padx=(10, 0))
        self.format = tk.StringVar(value="1280x720")
        ttk.Combobox(cam, textvariable=self.format, values=("1280x720", "1920x1080"), width=11, state="readonly").pack(side="left", padx=6)

    def _background_tab(self, book: ttk.Notebook) -> None:
        tab = ttk.Frame(book)
        book.add(tab, text="Scene")
        self.mode = tk.StringVar(value="blur")
        modes = ttk.Frame(tab)
        modes.pack(fill="x", pady=6)
        for label, value in (("Blur", "blur"), ("Color", "color"), ("Image", "image"), ("Video", "video"), ("Studio", "studio"), ("Green", "green"), ("Remove", "remove")):
            ttk.Radiobutton(modes, text=label, value=value, variable=self.mode, command=self._pull).pack(anchor="w")
        self.blur = self._scale(tab, "Blur", 1, 80, 28)
        self.bg_scale = self._scale(tab, "Backdrop scale", 40, 220, 100)
        self.bg_x = self._scale(tab, "Backdrop X", -100, 100, 0)
        self.bg_y = self._scale(tab, "Backdrop Y", -100, 100, 0)
        ttk.Button(tab, text="Load backdrop", command=self.load_background).pack(anchor="w", pady=6)
        ttk.Button(tab, text="Load backdrop video", command=self.load_background_video).pack(anchor="w")
        self.color_r = self._scale(tab, "Red", 0, 255, 20)
        self.color_g = self._scale(tab, "Green", 0, 255, 24)
        self.color_b = self._scale(tab, "Blue", 0, 255, 28)
        self.spill = self._scale(tab, "Edge spill", 0, 100, 55)
        self.bokeh = self._scale(tab, "Portrait falloff", 0, 100, 40)
        self.belong = self._scale(tab, "Belong", 0, 100, 35)
        self.belong_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Match scene light", variable=self.belong_on, command=self._pull).pack(anchor="w", pady=4)
        ttk.Button(tab, text="Scan cameras", command=self.scan_cameras).pack(anchor="w", pady=4)

    def _look_tab(self, book: ttk.Notebook) -> None:
        tab = ttk.Frame(book)
        book.add(tab, text="Grade")
        self.exposure = self._scale(tab, "Exposure", -100, 100, 0)
        self.contrast = self._scale(tab, "Contrast", 50, 180, 100)
        self.saturation = self._scale(tab, "Saturation", 0, 200, 100)
        self.temperature = self._scale(tab, "Temperature", -100, 100, 0)
        self.sharpness = self._scale(tab, "Sharpness", 0, 100, 15)
        self.vignette = self._scale(tab, "Vignette", 0, 100, 0)
        self.beauty = self._scale(tab, "Skin smooth", 0, 100, 0)
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=8)
        for name in PRESETS:
            ttk.Button(row, text=name, command=lambda n=name: self.apply_preset(n)).pack(fill="x", pady=2)

    def _face_tab(self, book: ttk.Notebook) -> None:
        tab = ttk.Frame(book)
        book.add(tab, text="Subject")
        self.eye = tk.BooleanVar(value=False)
        self.autoframe = tk.BooleanVar(value=False)
        self.key_on = tk.BooleanVar(value=True)
        self.denoise_on = tk.BooleanVar(value=True)
        self.mirror = tk.BooleanVar(value=True)
        self.quality = tk.BooleanVar(value=False)
        for text, var in (("Mirror", self.mirror), ("Eye contact", self.eye), ("Auto frame", self.autoframe), ("Key light", self.key_on), ("Low-light denoise", self.denoise_on), ("Quality matte", self.quality)):
            ttk.Checkbutton(tab, text=text, variable=var, command=self._pull).pack(anchor="w", pady=1)
        self.eye_strength = self._scale(tab, "Eye strength", 0, 100, 70)
        self.keylight = self._scale(tab, "Key light", 0, 100, 35)
        self.video_denoise = self._scale(tab, "Denoise", 0, 100, 30)
        self.variant = tk.StringVar(value="mobilenetv3")
        ttk.Radiobutton(tab, text="Fast matte", value="mobilenetv3", variable=self.variant, command=self._pull).pack(anchor="w")
        ttk.Radiobutton(tab, text="Quality matte", value="resnet50", variable=self.variant, command=self._pull).pack(anchor="w")

    def _overlay_tab(self, book: ttk.Notebook) -> None:
        tab = ttk.Frame(book)
        book.add(tab, text="Graphics")
        self.lower_on = tk.BooleanVar(value=False)
        self.logo_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Lower third", variable=self.lower_on, command=self._pull).pack(anchor="w")
        ttk.Label(tab, text="Name", style="Muted.TLabel").pack(anchor="w")
        self.lower_name = tk.StringVar()
        ttk.Entry(tab, textvariable=self.lower_name).pack(fill="x", pady=2)
        ttk.Label(tab, text="Title", style="Muted.TLabel").pack(anchor="w")
        self.lower_title = tk.StringVar()
        ttk.Entry(tab, textvariable=self.lower_title).pack(fill="x", pady=2)
        self.lower_name.trace_add("write", lambda *_: self._pull())
        self.lower_title.trace_add("write", lambda *_: self._pull())
        ttk.Checkbutton(tab, text="Logo", variable=self.logo_on, command=self._pull).pack(anchor="w", pady=(8, 0))
        ttk.Button(tab, text="Load logo", command=self.load_logo).pack(anchor="w", pady=4)
        self.logo_scale = self._scale(tab, "Logo size", 6, 40, 16)
        self.logo_x = self._scale(tab, "Logo X", 0, 90, 4)
        self.logo_y = self._scale(tab, "Logo Y", 0, 90, 4)

    def _audio_tab(self, book: ttk.Notebook) -> None:
        tab = ttk.Frame(book)
        book.add(tab, text="Audio")
        ttk.Label(tab, text="Route the processed mic to a virtual cable, then select that cable in Zoom.", style="Muted.TLabel", wraplength=300).pack(anchor="w", pady=4)
        self.noise_on = tk.BooleanVar(value=True)
        self.echo_on = tk.BooleanVar(value=True)
        self.studio_on = tk.BooleanVar(value=False)
        self.deepfilter = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Noise removal", variable=self.noise_on, command=self._pull_audio).pack(anchor="w")
        ttk.Checkbutton(tab, text="Room echo", variable=self.echo_on, command=self._pull_audio).pack(anchor="w")
        ttk.Checkbutton(tab, text="Studio voice", variable=self.studio_on, command=self._pull_audio).pack(anchor="w")
        ttk.Checkbutton(tab, text="DeepFilterNet if installed", variable=self.deepfilter).pack(anchor="w")
        self.noise = self._scale(tab, "Noise", 0, 100, 75)
        self.echo = self._scale(tab, "Echo", 0, 100, 35)
        self.voice = self._scale(tab, "Studio", 0, 100, 40)
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="Mic in", style="Muted.TLabel").pack(side="left")
        self.mic_in = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.mic_in, width=6).pack(side="left", padx=4)
        ttk.Label(row, text="Cable out", style="Muted.TLabel").pack(side="left")
        self.mic_out = tk.StringVar(value="")
        ttk.Entry(row, textvariable=self.mic_out, width=6).pack(side="left", padx=4)
        ttk.Button(tab, text="Start mic chain", command=self.toggle_audio).pack(anchor="w", pady=4)
        ttk.Button(tab, text="List devices", command=self.list_audio).pack(anchor="w")

    def _device_index(self, text: str):
        text = text.strip()
        return int(text) if text else None

    def _pull_audio(self) -> None:
        audio = self.session.audio
        audio.noise_on = self.noise_on.get()
        audio.echo_on = self.echo_on.get()
        audio.studio_on = self.studio_on.get()
        audio.noise = self.noise.get() / 100
        audio.echo = self.echo.get() / 100
        audio.studio = self.voice.get() / 100

    def toggle_audio(self) -> None:
        audio = self.session.audio
        if audio.stream is not None:
            audio.stop()
            self.status.set("Mic chain off")
            return
        if not audio.available():
            self.status.set("Install sounddevice for the mic chain")
            return
        self._pull_audio()
        try:
            audio.start(self._device_index(self.mic_in.get()), self._device_index(self.mic_out.get()), deepfilter=self.deepfilter.get())
        except Exception as exc:
            self.status.set(str(exc))
            return
        self.status.set(f"MIC  {audio.engine_name}  {audio.device_label}")

    def list_audio(self) -> None:
        audio = self.session.audio
        if not audio.available():
            self.status.set("sounddevice is not installed")
            return
        print("\n".join(audio.list_devices()))
        self.status.set("Audio devices printed in the console")

    def snapshot(self) -> None:
        frame = self.session.latest
        if frame is None:
            self.status.set("Start the camera before snapping")
            return
        OUT_DIR.mkdir(exist_ok=True)
        path = OUT_DIR / f"snap-{time.strftime('%Y%m%d-%H%M%S')}.png"
        cv2.imwrite(str(path), frame)
        self.status.set(f"Saved {path.name}")

    def _scale(self, parent, label: str, lo: int, hi: int, value: int) -> tk.IntVar:
        var = tk.IntVar(value=value)
        ttk.Label(parent, text=label, style="Muted.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Scale(parent, from_=lo, to=hi, variable=var, command=lambda _v: self._pull()).pack(fill="x")
        return var

    def _pull(self) -> None:
        s = self.session.state
        s.mode = self.mode.get()
        s.blur = int(self.blur.get())
        s.bg_scale = self.bg_scale.get() / 100
        s.bg_x = self.bg_x.get() / 100
        s.bg_y = self.bg_y.get() / 100
        s.color_r, s.color_g, s.color_b = int(self.color_r.get()), int(self.color_g.get()), int(self.color_b.get())
        s.spill = self.spill.get() / 100
        s.bokeh = self.bokeh.get() / 100
        s.belong = self.belong.get() / 100
        s.belong_on = self.belong_on.get()
        s.exposure = self.exposure.get() / 100
        s.contrast = self.contrast.get() / 100
        s.saturation = self.saturation.get() / 100
        s.temperature = self.temperature.get() / 100
        s.sharpness = self.sharpness.get() / 100
        s.vignette = self.vignette.get() / 100
        s.beauty = self.beauty.get() / 100
        s.eye = self.eye.get()
        s.eye_strength = self.eye_strength.get() / 100
        s.autoframe = self.autoframe.get()
        s.key_on = self.key_on.get()
        s.keylight = self.keylight.get() / 100
        s.denoise_on = self.denoise_on.get()
        s.video_denoise = self.video_denoise.get() / 100
        s.mirror = self.mirror.get()
        s.quality = self.quality.get()
        s.lower_on = self.lower_on.get()
        s.lower_name = self.lower_name.get()
        s.lower_title = self.lower_title.get()
        s.logo_on = self.logo_on.get()
        s.logo_scale = self.logo_scale.get() / 100
        s.logo_x = self.logo_x.get() / 100
        s.logo_y = self.logo_y.get() / 100
        self.session.variant = self.variant.get()

    def _load(self) -> None:
        if not SETTINGS.exists():
            return
        try:
            data = json.loads(SETTINGS.read_text())
        except json.JSONDecodeError:
            return
        mapping = {
            "mode": self.mode, "blur": self.blur, "bg_scale": self.bg_scale, "bg_x": self.bg_x, "bg_y": self.bg_y,
            "color_r": self.color_r, "color_g": self.color_g, "color_b": self.color_b, "spill": self.spill,
            "bokeh": self.bokeh, "belong": self.belong,
            "exposure": self.exposure, "contrast": self.contrast, "saturation": self.saturation,
            "temperature": self.temperature, "sharpness": self.sharpness, "vignette": self.vignette, "beauty": self.beauty,
            "eye_strength": self.eye_strength, "keylight": self.keylight, "video_denoise": self.video_denoise,
            "logo_scale": self.logo_scale, "logo_x": self.logo_x, "logo_y": self.logo_y,
        }
        checks = {"eye": self.eye, "autoframe": self.autoframe, "key_on": self.key_on, "denoise_on": self.denoise_on, "mirror": self.mirror, "quality": self.quality, "lower_on": self.lower_on, "logo_on": self.logo_on, "belong_on": self.belong_on}
        for key, var in mapping.items():
            if key in data:
                var.set(data[key])
        for key, var in checks.items():
            if key in data:
                var.set(bool(data[key]))
        if "variant" in data:
            self.variant.set(data["variant"])
        if "lower_name" in data:
            self.lower_name.set(data["lower_name"])
        if "lower_title" in data:
            self.lower_title.set(data["lower_title"])
        self._pull()

    def _save(self) -> None:
        self._pull()
        payload = self.session.state.to_json()
        payload["variant"] = self.session.variant
        SETTINGS.write_text(json.dumps(payload, indent=2))

    def apply_preset(self, name: str) -> None:
        values = PRESETS[name]
        if "mode" in values:
            self.mode.set(values["mode"])
        if "blur" in values:
            self.blur.set(values["blur"])
        for key, widget in (("beauty", self.beauty), ("keylight", self.keylight), ("video_denoise", self.video_denoise), ("vignette", self.vignette), ("sharpness", self.sharpness), ("bokeh", self.bokeh), ("belong", self.belong), ("spill", self.spill)):
            if key in values:
                widget.set(int(float(values[key]) * 100) if float(values[key]) <= 1 else int(values[key]))
        self.belong_on.set(values.get("belong_on", self.belong_on.get()))
        self.eye.set(values.get("eye", self.eye.get()))
        self.autoframe.set(values.get("autoframe", self.autoframe.get()))
        self.key_on.set(values.get("key_on", self.key_on.get()))
        self.denoise_on.set(values.get("denoise_on", self.denoise_on.get()))
        self.lower_on.set(values.get("lower_on", self.lower_on.get()))
        self.logo_on.set(values.get("logo_on", self.logo_on.get()))
        self._pull()
        self.status.set(name)

    def scan_cameras(self) -> None:
        found = []
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        for index in range(6):
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                found.append(str(index))
                cap.release()
        self.status.set("Cameras " + (", ".join(found) if found else "none"))

    def load_background(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if not path:
            return
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            self.status.set("Could not read that image")
            return
        self.session.background = image
        self.mode.set("image")
        self._pull()
        self.status.set(Path(path).name)

    def load_background_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi")])
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self.status.set("Could not read that video")
            return
        if self.session.background_video is not None:
            self.session.background_video.release()
        self.session.background_video = cap
        self.mode.set("video")
        self._pull()
        self.status.set(Path(path).name)

    def load_logo(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if not path:
            return
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            self.status.set("Could not read that logo")
            return
        self.session.logo = image
        self.logo_on.set(True)
        self._pull()

    def start_live(self) -> None:
        if self.session.running:
            return
        self._pull()
        w, h = self.format.get().split("x")
        self.session.width, self.session.height = int(w), int(h)
        self.session.running = True
        self.worker = threading.Thread(target=self._live_loop, daemon=True)
        self.worker.start()

    def _live_loop(self) -> None:
        try:
            self.session.ensure_model()
        except Exception as exc:
            self.session.running = False
            self.status.set(str(exc))
            return
        cap = open_camera(int(self.camera.get()), self.session.width, self.session.height, 30)
        if not cap.isOpened():
            self.session.running = False
            self.status.set("Camera did not open. Check the camera index.")
            return
        self.status.set(f"LIVE  ·  {self.session.device.type.upper()}  ·  {self.session.width}x{self.session.height}")
        last = time.perf_counter()
        while self.session.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.resize(frame, (self.session.width, self.session.height))
            out = self.session.render(frame)
            now = time.perf_counter()
            self.session.fps = 1.0 / max(now - last, 1e-6)
            last = now
            self.session.latest = out
            if self.session.vcam is not None:
                self.session.vcam.send(out, pace=False)
            if self.session.writer is not None:
                self.session.writer.write(out)
        cap.release()

    def _paint(self) -> None:
        frame = self.session.latest
        if frame is not None:
            self._draw(frame)
        self.after(33, self._paint)

    def _draw(self, frame: np.ndarray) -> None:
        try:
            from PIL import Image, ImageTk
        except ImportError:
            return
        width = max(320, self.stage.winfo_width())
        height = max(180, self.stage.winfo_height())
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.stage.delete("all")
        self.stage.create_image(width // 2, height // 2, image=self.photo)
        self.stage.create_text(18, 22, anchor="w", fill=ACCENT, text="PGM", font=("Segoe UI", 9, "bold"))
        self.stage.create_text(18, height - 18, anchor="w", fill="#d7dbe3", text=f"{self.session.fps:4.1f}  FPS", font=("Segoe UI", 9))
        if self.session.writer is not None:
            elapsed = int(time.perf_counter() - getattr(self.session, "rec_started", time.perf_counter()))
            self.stage.create_oval(width - 28, 16, width - 16, 28, fill=REC, outline=REC)
            self.stage.create_text(width - 36, 22, anchor="e", fill=REC, text=f"REC  {elapsed // 60:02d}:{elapsed % 60:02d}", font=("Segoe UI", 9, "bold"))

    def stop(self) -> None:
        self.session.running = False
        self._close_writer()
        self.status.set("Stopped")

    def toggle_vcam(self) -> None:
        if self.session.vcam is not None:
            self.session.vcam.close()
            self.session.vcam = None
            self.status.set("Virtual camera off")
            return
        from broadcast import VirtualCam
        cam = VirtualCam(self.session.width, self.session.height, 30)
        self.session.vcam = cam if cam.cam is not None else None
        self.status.set("Virtual camera " + cam.label)

    def toggle_record(self) -> None:
        if self.session.writer is not None:
            self._close_writer()
            self.rec_button.configure(text="Record")
            self.status.set("Recording saved to exports")
            return
        OUT_DIR.mkdir(exist_ok=True)
        path = OUT_DIR / f"live-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (self.session.width, self.session.height))
        if not writer.isOpened():
            self.status.set("Could not open a writer")
            return
        self.session.writer = writer
        self.session.rec_started = time.perf_counter()
        self.rec_button.configure(text="Stop rec")
        self.status.set(f"REC  {path.name}")

    def _close_writer(self) -> None:
        if self.session.writer is not None:
            self.session.writer.release()
            self.session.writer = None

    def open_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi")])
        if path:
            self.session.source_path = path
            self.status.set(f"{Path(path).name}  ·  Export video to render the current grade")

    def export_video(self) -> None:
        if not self.session.source_path:
            self.open_video()
        if not self.session.source_path or (self.export_thread and self.export_thread.is_alive()):
            return
        self._pull()
        self.export_thread = threading.Thread(target=self._export, daemon=True)
        self.export_thread.start()

    def _export(self) -> None:
        self.session.ensure_model()
        self.session.matter.reset()
        cap = cv2.VideoCapture(self.session.source_path)
        if not cap.isOpened():
            self.status.set("Could not open the source video")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        OUT_DIR.mkdir(exist_ok=True)
        dest = OUT_DIR / f"edit-{Path(self.session.source_path).stem}.mp4"
        writer = None
        done = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = cv2.resize(frame, (1280, 720))
            out = self.session.render(frame)
            if writer is None:
                writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0]))
            writer.write(out)
            done += 1
            if done % 15 == 0:
                self.status.set(f"Exporting {done} / {total or '?'}")
        cap.release()
        if writer is not None:
            writer.release()
        self.session.matter.reset()
        self.status.set(f"Exported {dest.name}")

    def close(self) -> None:
        self._save()
        self.session.running = False
        self._close_writer()
        if self.session.vcam is not None:
            self.session.vcam.close()
        self.session.audio.stop()
        self.session.tracker.close()
        if self.session.background_video is not None:
            self.session.background_video.release()
        self.destroy()

    def _bind_keys(self) -> None:
        self.bind("<space>", lambda _: self.start_live() if not self.session.running else self.stop())
        self.bind("<Key-r>", lambda _: self.toggle_record())
        self.bind("<Key-n>", lambda _: self.toggle_vcam())
        self.bind("<Key-s>", lambda _: self.snapshot())
        self.bind("<Escape>", lambda _: self.stop())


if __name__ == "__main__":
    Studio().mainloop()
