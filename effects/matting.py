"""Recurrent video matting. RVM is GPL-3.0."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but this PyTorch build cannot see a GPU.")
        return torch.device("cuda")
    if requested == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise SystemExit("MPS was requested but it is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Matter:
    def __init__(self, variant: str, device: torch.device) -> None:
        self.device = device
        self.variant = variant
        self.fp16 = device.type == "cuda"
        print(f"Loading RVM {variant} on {device} ({'fp16' if self.fp16 else 'fp32'})...")
        model = torch.hub.load("PeterL1n/RobustVideoMatting", variant, trust_repo=True)
        model = model.eval().to(device)
        if self.fp16:
            model = model.half()
        self.model = model
        self.rec: list[Optional[torch.Tensor]] = [None, None, None, None]
        self.prev_pha: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.rec = [None, None, None, None]
        self.prev_pha = None

    @torch.inference_mode()
    def matte(self, frame_bgr: np.ndarray, downsample: float) -> tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        src = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0)
        src = src.float().div_(255.0)
        if self.fp16:
            src = src.half()
        fgr, pha, *self.rec = self.model(src, *self.rec, downsample_ratio=float(downsample))
        fgr = fgr[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        pha = pha[0, 0].float().clamp(0, 1).cpu().numpy()
        if self.prev_pha is not None and self.prev_pha.shape == pha.shape:
            pha = 0.65 * pha + 0.35 * self.prev_pha
        self.prev_pha = pha
        return fgr, pha


def auto_downsample(height: int, width: int, quality: bool) -> float:
    target = 768.0 if quality else 512.0
    return float(min(1.0, max(0.15, target / float(max(height, width)))))


def soften_alpha(pha: np.ndarray) -> np.ndarray:
    edge = cv2.GaussianBlur(pha, (0, 0), 1.2)
    refined = np.where(np.abs(pha - 0.5) < 0.18, edge, pha)
    return np.clip((refined - 0.03) / 0.97, 0.0, 1.0).astype(np.float32)
