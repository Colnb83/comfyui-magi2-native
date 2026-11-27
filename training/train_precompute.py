#!/usr/bin/env python3
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""Turn clips + captions into cached training samples.

    python training/train_precompute.py --data <dir> --out <dir> [--size 448x256] [--size 896x512]

``<dir>`` holds pairs ``name.<media>`` and ``name.json`` / ``name.txt`` (the caption:
MAGI-2's structured JSON, or plain text). Media can be

* a video (.mp4/.mov/.mkv/.webm/.avi), with or without sound. Up to 10 s is used
  (``--start`` seconds in). A shorter clip is used at its own length: the frame count
  is rounded down to 4k+1 at 12.5 fps (at least 5 frames, 0.4 s) and the audio is cut
  to the same span, 2 audio latents per frame;
* a still image (.png/.jpg/.jpeg/.webp): one frame, no audio. It teaches appearance only.

For every clip and size one file ``<out>/<name>_<W>x<H>.pt`` is written with

* ``video``  ``[48, T, H/16, W/16]`` fp32: Wan2.2 VAE latent at 12.5 fps, normalised (the
  space the denoiser works in); T = 32 for 10 s, 1 for a still;
* ``audio``  ``[L, 64]`` fp32: Stable Audio Open latent of the waveform stretched by
  512/441 (the inverse of the stretch the pipeline applies after decoding), L = 250 for
  10 s; zeros with ``has_audio = False`` when the clip is silent or has no audio stream,
  and ``[0, 64]`` for a still;
* ``text``   ``[tokens, 5120]`` bf16: Qwen3.5 features, hidden_states[-3], as at inference;
* ``caption``, ``source``, ``size`` and the conventions used.

Text features are the slow part on the CPU (15 to 40 s each); ``--text-device cuda``
puts the 52 GB encoder on the GPU instead. Existing outputs are skipped.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

REPO = os.environ.get("MAGI2_FORK_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "magi2")
sys.path.insert(0, REPO)
DEFAULT_CONFIG = os.path.join(REPO, "configs", "local", "1gpu_preview.json")

SECONDS = 10.0
VIDEO_FPS = 12.5
FRAMES = 125
SAMPLE_RATE = 44100
AUDIO_LATENTS = 250
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", path],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    kinds = [s.get("codec_type") for s in info.get("streams", [])]
    return float(info["format"]["duration"]), "audio" in kinds


def usable_frames(duration, start):
    """Frames to use at 12.5 fps: 4k+1, at most 125."""
    available = int((duration - start) * VIDEO_FPS + 1e-6)
    return min(FRAMES, (available - 1) // 4 * 4 + 1) if available >= 1 else 0


def read_video(path, width, height, start, frames_wanted, still=False):
    """RGB frames at 12.5 fps (or the one image), scaled to cover width x height then centre-cropped."""
    scale = f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,crop={width}:{height}"
    if still:
        cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", scale, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    else:
        cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(SECONDS + 0.5), "-i", path, "-an", "-vf", f"fps={VIDEO_FPS}," + scale,
               "-frames:v", str(frames_wanted), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    if frames.shape[0] < frames_wanted:
        raise ValueError(f"only {frames.shape[0]} frames at {VIDEO_FPS} fps; expected {frames_wanted}")
    video = torch.from_numpy(frames[:frames_wanted].copy()).permute(3, 0, 1, 2).float() / 127.5 - 1.0  # [3, T, H, W]
    return video.unsqueeze(0)


def read_audio(path, start, seconds):
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-t", str(seconds), "-i", path, "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE),
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    wave = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
    target = int(round(seconds * SAMPLE_RATE))
    if wave.shape[0] < target:
        wave = np.concatenate([wave, np.zeros((target - wave.shape[0], 2), dtype=np.float32)])
    return wave[:target]


def stretch_for_encoder(wave, latents):
    """The pipeline decodes 250 latents to 512,000 samples and resamples them to 441,000
    (10 s at 44.1 kHz). Training goes the other way: stretch by 512/441, then encode."""
    from scipy.signal import resample

    return resample(wave, latents * 2048, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", action="append", default=None, help="WxH, multiples of 16; repeatable (default 448x256)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--ckpt-root", default=os.environ.get("MAGI2_CKPT_ROOT", ""))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audio-latent", default="sample", choices=["sample", "mean"],
                    help="audio VAE posterior: a sample (what the VAE's own encode returns) or its mean")
    args = ap.parse_args()
    sizes = [tuple(int(v) for v in s.lower().split("x")) for s in (args.size or ["448x256"])]
    for w, h in sizes:
        if w % 16 or h % 16:
            sys.exit(f"size {w}x{h} is not a multiple of 16")
    os.makedirs(args.out, exist_ok=True)

    clips = []
    for name in sorted(os.listdir(args.data)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in VIDEO_EXT + IMAGE_EXT:
            continue
        caption = next((os.path.join(args.data, stem + e) for e in (".json", ".txt") if os.path.isfile(os.path.join(args.data, stem + e))), None)
        if caption is None:
            print(f"[data] SKIP {name}: no {stem}.json or {stem}.txt caption", flush=True)
            continue
        clips.append((stem, os.path.join(args.data, name), caption))
    if not clips:
        sys.exit(f"no captioned clips in {args.data}")
    print(f"[data] {len(clips)} captioned clips, sizes {sizes}", flush=True)

    from inference.model.qwen35 import Qwen35TextEncoder
    from inference.model.vae2_2 import get_vae2_2
    from inference.pipeline.audio_decoder import SAAudioFeatureExtractor

    vae = get_vae2_2(os.path.join(args.ckpt_root, "vae", "Wan2.2_VAE.pth"), args.device, weight_dtype=torch.float32)
    audio_vae = SAAudioFeatureExtractor(os.path.join(args.ckpt_root, "stable-audio-open-1.0"))
    text_encoder = None
    torch.manual_seed(args.seed)

    done = skipped = failed = 0
    for stem, path, caption_path in clips:
        todo = [(w, h) for w, h in sizes if not os.path.isfile(os.path.join(args.out, f"{stem}_{w}x{h}.pt"))]
        if not todo:
            skipped += 1
            continue
        try:
            t0 = time.time()
            still = os.path.splitext(path)[1].lower() in IMAGE_EXT
            caption = open(caption_path, encoding="utf-8").read().strip()
            if not caption:
                raise ValueError("caption file is empty")
            if still:
                n_frames, n_audio, has_audio, rms = 1, 0, False, 0.0
                audio = torch.zeros(0, 64)
            else:
                duration, has_stream = probe(path)
                n_frames = usable_frames(duration, args.start)
                if n_frames < 5:
                    raise ValueError(f"clip is {duration:.2f} s; need at least 0.4 s from {args.start:.1f} s")
                n_audio = 2 * n_frames  # 250 latents for 125 frames
                seconds = n_frames / VIDEO_FPS
                wave = read_audio(path, args.start, seconds) if has_stream else np.zeros((int(round(seconds * SAMPLE_RATE)), 2), dtype=np.float32)
                rms = float(np.sqrt(np.mean(wave**2)))
                has_audio = bool(has_stream and rms > 1e-5)
                if has_audio:
                    stretched = torch.from_numpy(stretch_for_encoder(wave, n_audio)).T.unsqueeze(0).to(audio_vae.device)
                    if args.audio_latent == "mean":
                        with torch.no_grad():
                            pre = audio_vae.vae_model.encode(stretched, skip_bottleneck=True)
                        mean = pre.chunk(2, dim=1)[0]
                        audio = ((mean - audio_vae.rescale_mean) / audio_vae.rescale_std).squeeze(0).T.float().cpu()
                    else:
                        audio = audio_vae.encode(stretched).squeeze(0).T.float().cpu()  # [L, 64]
                    if audio.shape[0] != n_audio:
                        raise ValueError(f"audio encoder returned {audio.shape[0]} latents, expected {n_audio}")
                else:
                    audio = torch.zeros(n_audio, 64)

            if text_encoder is None:
                tl = time.time()
                text_encoder = Qwen35TextEncoder(model_path=os.path.join(args.ckpt_root, "text_encoder"), device=args.text_device,
                                                 precision=torch.bfloat16, skip_layer=2)
                print(f"[data] text encoder on {args.text_device} in {time.time() - tl:.1f}s", flush=True)
            text = text_encoder.encode(caption).squeeze(0).to(torch.bfloat16).cpu().clone()

            for w, h in todo:
                video = read_video(path, w, h, args.start, n_frames, still=still).to(args.device)
                with torch.no_grad():
                    latent = vae.encode(video).squeeze(0).float().cpu()
                expect = (48, (n_frames - 1) // 4 + 1, h // 16, w // 16)
                if tuple(latent.shape) != expect:
                    raise ValueError(f"video latent {tuple(latent.shape)}, expected {expect}")
                torch.save(
                    {"video": latent, "audio": audio, "has_audio": has_audio, "text": text, "caption": caption,
                     "source": os.path.basename(path), "size": (w, h), "audio_rms": rms, "frames": n_frames, "still": still,
                     "conventions": {"video_fps": VIDEO_FPS, "frames": FRAMES, "audio_stretch": "512/441", "text_layer": "hidden_states[-3]",
                                     "video_space": "wan2.2 normalised"}},
                    os.path.join(args.out, f"{stem}_{w}x{h}.pt"),
                )
                del video
            done += 1
            print(f"[data] {stem}: {'still' if still else f'{n_frames} frames'} text {tuple(text.shape)} audio {tuple(audio.shape)} rms {rms:.4f} has_audio={has_audio} sizes {todo} in {time.time() - t0:.1f}s", flush=True)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the batch
            failed += 1
            print(f"[data] FAILED {stem}: {type(exc).__name__}: {exc}", flush=True)
    print(f"[data] done: {done} written, {skipped} already present, {failed} failed -> {args.out}", flush=True)
    sys.exit(1 if failed and not done else 0)


if __name__ == "__main__":
    main()
