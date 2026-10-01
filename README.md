# GPU Broadcast stack

Local webcam and microphone effects covering the NVIDIA Broadcast list. No cloud, no account. RVM weights download once through torch.hub, then the video path is offline.

NVIDIA's Maxine networks are closed. This is the open equivalent, effect for effect:

| Broadcast effect | This app | Engine |
| --- | --- | --- |
| Background blur / replace / remove | yes | Robust Video Matting, recurrent alpha, decontaminated foreground |
| Auto Frame | yes | Face Mesh when MediaPipe is installed, Haar otherwise, spring-smoothed crop |
| Video Noise Removal | yes | Chroma denoise plus temporal blend on still pixels |
| Eye Contact | yes, geometric | Iris shift toward the camera from Face Mesh landmarks. Not Maxine's generative eye model. Clamped so a bad detect cannot drag the eye out of the socket |
| Virtual Key Light | yes | Face-masked exposure lift and a top-front key. It relights the person, it does not light the room |
| Vignette | yes | Lens falloff |
| Noise Removal | yes | DeepFilterNet when installed and `--deepfilter` is set, otherwise a minimum-statistics spectral denoiser in the audio callback |
| Room Echo Removal | yes | Late-energy spectral decay suppressor on the mic |
| Studio Voice | yes | 80 Hz highpass, 3.5 kHz presence, compressor. This is an EQ/dynamics chain, not NVIDIA's studio-voice network |
| Virtual camera | yes | pyvirtualcam: OBS Virtual Camera on Windows/macOS, v4l2loopback on Linux |
| Virtual microphone | yes, via a cable | Route `--mic-out` at BlackHole, VB-Cable, or a virtual cable, then select that device in Zoom |

## Windows PC studio

`studio.py` is the desktop app. The preview sits in the window. The inspector is on the right. Settings save when you close it. The virtual camera and the recording never get the frame-rate readout.

```bat
install-windows.bat
.venv\Scripts\python studio.py
```

Scene, grade, subject, and graphics are separate tabs. Presets are Meeting, Stream, Podcast, and Clean plate. Open a video and export it with the same sliders used live. Files land in `exports/`.

The Audio tab starts the mic chain. Put the virtual-cable index in Cable out, then select that cable as the microphone in Zoom. Snap saves a clean PNG to `exports/`.



```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision    # Apple Silicon
# NVIDIA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install deepfilternet        # optional, better noise removal
```

Virtual camera: start OBS Virtual Camera once on Windows or macOS. On Linux, `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Broadcast" exclusive_caps=1`.

Virtual microphone: install BlackHole (macOS) or VB-Cable (Windows). `--list-audio` prints device indexes. Pass the cable's **input** index as `--mic-out`. Meeting apps then select that cable as their microphone.

## Run

```bash
python broadcast.py --list-audio
python broadcast.py --width 1280 --height 720 --virtual-cam --audio --mic-out 2 --eye-contact --autoframe
python broadcast.py --variant resnet50 --quality --mode studio --deepfilter --audio --mic-out 2
```

Keys: `b` blur, `c` color, `i` image, `o` video, `g` green, `x` remove, `u` studio, `k` mask, `f` auto frame, `e` eye contact, `l` key light, `d` video denoise, `v` vignette, `n` virtual cam, `a` audio chain, `[` `]` blur, `-` `=` matting detail, `r` reset temporal memory, `q` quit.

The virtual camera gets the clean frame. The HUD is preview-only.

## What still is not NVIDIA

Eye contact will not synthesize a new iris when you look far off camera. Studio Voice will not match their speech-enhancement network. Video denoise is a temporal/chroma cleaner, not their low-light generative model. Quality of the cut itself, on a GPU, is the part that sits next to Broadcast, because it is alpha matting with memory rather than a binary person mask.

RVM is GPL-3.0, so this program is GPL-3.0.
