"""Microphone effects: noise removal, room-echo reduction, studio voice.

DeepFilterNet is used when installed. The always-on path is a real
minimum-statistics spectral denoiser plus a decay-tail echo suppressor and a
biquad/compressor studio chain. Output goes to whatever playback device you
select — BlackHole, VB-Cable, or a virtual cable — so other apps can use it
as their microphone.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np


def _biquad(samples: np.ndarray, b: np.ndarray, a: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.empty_like(samples)
    x1, x2, y1, y2 = state
    for i, x0 in enumerate(samples):
        y0 = (b[0] * x0 + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2) / a[0]
        out[i] = y0
        x2, x1, y2, y1 = x1, x0, y1, y0
    return out, np.array([x1, x2, y1, y2], dtype=np.float32)


def _highpass(fs: int) -> tuple[np.ndarray, np.ndarray]:
    # 80 Hz Butterworth, RBJ cookbook.
    f0 = 80.0 / fs
    w = 2 * np.pi * f0
    cos_w, sin_w = np.cos(w), np.sin(w)
    alpha = sin_w / (2 * 0.707)
    b = np.array([(1 + cos_w) / 2, -(1 + cos_w), (1 + cos_w) / 2], dtype=np.float32)
    a = np.array([1 + alpha, -2 * cos_w, 1 - alpha], dtype=np.float32)
    return b, a


def _presence(fs: int) -> tuple[np.ndarray, np.ndarray]:
    f0, q, gain_db = 3500.0, 0.9, 3.5
    a = 10 ** (gain_db / 40)
    w = 2 * np.pi * (f0 / fs)
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2 * q)
    b = np.array([1 + alpha * a, -2 * cw, 1 - alpha * a], dtype=np.float32)
    aa = np.array([1 + alpha / a, -2 * cw, 1 - alpha / a], dtype=np.float32)
    return b, aa


class SpectralVoice:
    def __init__(self, sample_rate: int = 48000) -> None:
        self.sr = sample_rate
        self.n_fft = 1024
        self.hop = 256
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.noise = np.ones(self.n_fft // 2 + 1, dtype=np.float32) * 1e-4
        self.tail = np.zeros(self.n_fft // 2 + 1, dtype=np.float32)
        self.seen = 0
        self.hp_b, self.hp_a = _highpass(sample_rate)
        self.pr_b, self.pr_a = _presence(sample_rate)
        self.hp_state = np.zeros(4, dtype=np.float32)
        self.pr_state = np.zeros(4, dtype=np.float32)
        self.env = 0.0
        self.pending = np.zeros(0, dtype=np.float32)
        self.ola = np.zeros(self.n_fft, dtype=np.float32)

    def process(self, block: np.ndarray, noise: float, echo: float, studio: float) -> np.ndarray:
        audio = np.concatenate([self.pending, block.astype(np.float32)])
        out = []
        pos = 0
        while pos + self.n_fft <= len(audio):
            frame = audio[pos : pos + self.n_fft] * self.window
            spec = np.fft.rfft(frame)
            mag = np.abs(spec) + 1e-8
            self.seen += 1
            if self.seen < 8:
                self.noise = mag
            else:
                self.noise = np.minimum(self.noise * 1.0008, np.minimum(self.noise, mag) * 0.95 + mag * 0.05)
                quiet = mag < self.noise * 1.8
                self.noise = np.where(quiet, self.noise * 0.98 + mag * 0.02, self.noise)
            gain = 1.0
            if noise > 0.01:
                floor = self.noise * (1.0 + 8.0 * noise)
                wiener = np.clip(1.0 - (floor / mag) ** 1.2, 0.04, 1.0)
                gain = gain * (1.0 - noise + noise * wiener)
            if echo > 0.01:
                self.tail = self.tail * (0.82 - 0.15 * echo) + mag * (0.18 + 0.15 * echo)
                late = np.clip(self.tail / mag, 0, 1)
                gain = gain * (1.0 - echo * 0.65 * late)
            spec = spec * gain
            chunk = np.fft.irfft(spec).astype(np.float32) * self.window
            self.ola[: self.n_fft - self.hop] = self.ola[self.hop :]
            self.ola[self.n_fft - self.hop :] = 0
            self.ola += chunk
            out.append(self.ola[: self.hop].copy())
            pos += self.hop
        self.pending = audio[pos:]
        if not out:
            return np.zeros_like(block, dtype=np.float32)
        voice = np.concatenate(out)
        if studio > 0.01:
            voice, self.hp_state = _biquad(voice, self.hp_b, self.hp_a, self.hp_state)
            voice, self.pr_state = _biquad(voice, self.pr_b, self.pr_a, self.pr_state)
            mixed = []
            attack, release = 0.08, 0.01
            for sample in voice:
                level = abs(float(sample))
                self.env = self.env + (attack if level > self.env else release) * (level - self.env)
                comp = 1.0 if self.env < 0.12 else (0.12 / max(self.env, 1e-4)) ** (0.55 * studio)
                mixed.append(sample * comp * (1.0 + 0.4 * studio))
            voice = np.asarray(mixed, dtype=np.float32)
        if len(voice) < len(block):
            voice = np.pad(voice, (0, len(block) - len(voice)))
        return np.clip(voice[: len(block)], -1.0, 1.0)


class AudioEngine:
    def __init__(self) -> None:
        self.noise = 0.75
        self.echo = 0.35
        self.studio = 0.4
        self.noise_on = True
        self.echo_on = False
        self.studio_on = False
        self.use_deepfilter = False
        self.df = None
        self.stream = None
        self.error = ""
        self.device_label = "off"
        self.engine_name = "off"
        self._lock = threading.Lock()
        self._voice: Optional[SpectralVoice] = None
        self._sd = None
        self._df_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self._out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self._df_pending = np.zeros(0, dtype=np.float32)
        self._df_ready = np.zeros(0, dtype=np.float32)

    def available(self) -> bool:
        try:
            import sounddevice  # noqa: F401
            return True
        except ImportError:
            return False

    def list_devices(self) -> list[str]:
        import sounddevice as sd
        lines = []
        for index, dev in enumerate(sd.query_devices()):
            lines.append(f"{index:2d}  in {dev['max_input_channels']} out {dev['max_output_channels']}  {dev['name']}")
        return lines

    def start(self, input_device: Optional[int], output_device: Optional[int], sample_rate: int = 48000, deepfilter: bool = False) -> None:
        import sounddevice as sd

        self._sd = sd
        self._voice = SpectralVoice(sample_rate)
        self.use_deepfilter = deepfilter
        self.engine_name = "spectral"
        if deepfilter:
            try:
                from df.enhance import enhance, init_df
                import torch

                model, state, _ = init_df()
                self.df = (model, state, enhance, torch)
                self.engine_name = "DeepFilterNet"
                self._stop.clear()
                self._df_thread = threading.Thread(target=self._deepfilter_loop, daemon=True)
                self._df_thread.start()
                print("Mic noise model: DeepFilterNet (about 400 ms latency)")
            except Exception as exc:
                self.df = None
                self.engine_name = "spectral"
                print(f"DeepFilterNet unavailable ({exc}). Using spectral noise removal.")
        else:
            print("Mic noise model: spectral minimum-statistics")

        def callback(indata, outdata, frames, time_info, status):
            del time_info, status
            block = indata[:, 0].copy()
            with self._lock:
                noise = self.noise if self.noise_on else 0.0
                echo = self.echo if self.echo_on else 0.0
                studio = self.studio if self.studio_on else 0.0
                voice = self._voice
                use_df = self.df is not None and self.noise_on
            if voice is None:
                outdata[:, 0] = block
                return
            if use_df:
                cleaned = self._push_deepfilter(block)
                if echo > 0.01 or studio > 0.01:
                    cleaned = voice.process(cleaned, 0.0, echo, studio)
                outdata[:, 0] = cleaned
                return
            outdata[:, 0] = voice.process(block, noise, echo, studio)

        self.stream = sd.Stream(
            device=(input_device, output_device),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=480,
            callback=callback,
        )
        self.stream.start()
        in_name = sd.query_devices(input_device)["name"] if input_device is not None else "default input"
        out_name = sd.query_devices(output_device)["name"] if output_device is not None else "default output"
        self.device_label = f"{in_name} -> {out_name}"
        print(f"Audio chain: {self.device_label}")

    def _deepfilter_loop(self) -> None:
        model, state, enhance, torch = self.df
        hop = 48000 // 2
        while not self._stop.is_set():
            try:
                chunk = self._in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            with torch.inference_mode():
                enhanced = enhance(model, state, tensor, pad=True)
            audio = enhanced.squeeze(0).detach().cpu().numpy().astype(np.float32)
            try:
                self._out_q.put(audio[:hop], timeout=0.2)
            except queue.Full:
                pass

    def _push_deepfilter(self, block: np.ndarray) -> np.ndarray:
        self._df_pending = np.concatenate([self._df_pending, block])
        hop = 24000
        while len(self._df_pending) >= hop:
            chunk = self._df_pending[:hop]
            self._df_pending = self._df_pending[hop:]
            try:
                self._in_q.put_nowait(chunk.copy())
            except queue.Full:
                pass
        while True:
            try:
                self._df_ready = np.concatenate([self._df_ready, self._out_q.get_nowait()])
            except queue.Empty:
                break
        if len(self._df_ready) >= len(block):
            out = self._df_ready[: len(block)]
            self._df_ready = self._df_ready[len(block) :]
            return out
        return block

    def stop(self) -> None:
        self._stop.set()
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.device_label = "off"

