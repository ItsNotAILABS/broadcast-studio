# Broadcast Studio

Local GPU webcam and microphone studio. No cloud, no account. ItsNotAI Labs.

NVIDIA Maxine weights are closed. This is the open equivalent: recurrent alpha matting, grade, key light, eye contact, virtual camera, and a mic chain.

## Install on Windows

Run `Setup.exe` in the release folder, or:

```bat
install-windows.bat
start-studio.bat
```

`Setup.exe` finds Python, creates the venv, installs the CUDA PyTorch wheel, installs dependencies, and adds desktop and Start Menu shortcuts. The first studio launch downloads the RVM weights. OBS Virtual Camera must be installed once so Zoom or Discord can select the output.

## What it matches

| Broadcast effect | This app | Engine |
| --- | --- | --- |
| Background blur / replace / remove | yes | Robust Video Matting, recurrent alpha, decontaminated foreground |
| Auto Frame | yes | Face Mesh, Haar fallback |
| Video noise removal | yes | Chroma denoise plus temporal blend on still pixels |
| Eye contact | geometric | Iris shift from Face Mesh landmarks, clamped |
| Virtual key light | yes | Face-masked exposure lift |
| Vignette | yes | Lens falloff |
| Noise removal | yes | Spectral denoiser, DeepFilterNet optional |
| Room echo | yes | Late-energy spectral suppressor |
| Studio voice | yes | Highpass, presence, compressor |
| Virtual camera | yes | pyvirtualcam / OBS Virtual Camera |

Eye contact will not invent an iris at extreme angles. Studio Voice is an EQ and dynamics chain, not NVIDIA's speech network.

RVM is GPL-3.0, so this program is GPL-3.0.
