# Copyright (c) 2026 SandAI. All Rights Reserved.
# Apache-2.0.
"""One training example -> model call, shared by the probe and the trainer.

Conventions read from the sampler and scheduler (inference/pipeline/sampler.py):

    x_t = (1 - sigma) * x0 + sigma * noise        velocity target = noise - x0

Video latents are in the model's normalised Wan2.2 space ``[1, 48, T, H, W]``;
audio latents are ``[1, L, 64]`` at 25 latents per second; text features are
``[1, tokens, 5120]``. The model is told ``sigma`` through the per-token time
channels (inference feeds trunc(sigma * 1000) / 1000).
"""

from __future__ import annotations

import torch

from inference.pipeline.preview_data_proxy import ModelInput


def build_model_input(video: torch.Tensor, audio: torch.Tensor, context: torch.Tensor, sigma: torch.Tensor) -> ModelInput:
    """A single-item ModelInput (the pipeline's prepare_model_input for one branch)."""
    _, _, frames, height, width = video.shape
    length = audio.shape[1]
    device = video.device
    t = sigma.reshape(1).to(device=device, dtype=video.dtype)
    return ModelInput(
        x_t=video,
        audio_x_t=audio,
        audio_feat_len=torch.tensor([length], device=device),
        txt_feat=context,
        txt_feat_len=torch.tensor([context.shape[1]], device=device),
        t=t,
        per_token_video_t=t.view(1, 1, 1, 1, 1).expand(1, 1, frames, height, width).clone(),
        per_token_audio_t=t.view(1, 1, 1).expand(1, length, 1).clone(),
        ref_audio_feat=torch.zeros(1, 0, audio.shape[2], dtype=audio.dtype, device=device),
        ref_audio_feat_len=torch.tensor([0], device=device),
        ref_video_feat=torch.zeros_like(video),
        ref_video_feat_len=torch.tensor([0], device=device),
        ref_image_feat=None,
        ref_image_feat_len=None,
        ref_image_special_token_embedding=None,
    )


def predict_velocity(model, data_proxy, video_t, audio_t, context, sigma):
    """Model velocity for one noised example. Returns (video ``[1,48,T,H,W]``, audio ``[1,L,64]``) in fp32."""
    packed = data_proxy.process_input(build_model_input(video_t, audio_t, context, sigma))
    out = model(*packed)
    video, audio = data_proxy.process_output(out)
    return video.float(), audio.float()


def flow_matching_loss(model, data_proxy, video0, audio0, context, sigma, noise_video=None, noise_audio=None, audio_weight: float = 1.0):
    """Mean squared error on the velocity, video and audio averaged separately.
    Returns (loss, video_mse, audio_mse)."""
    noise_video = torch.randn_like(video0) if noise_video is None else noise_video
    noise_audio = torch.randn_like(audio0) if noise_audio is None else noise_audio
    s = sigma.to(video0.device, torch.float32)
    video_t = (1 - s) * video0 + s * noise_video
    audio_t = (1 - s) * audio0 + s * noise_audio
    pred_video, pred_audio = predict_velocity(model, data_proxy, video_t, audio_t, context, s)
    video_mse = torch.mean((pred_video - (noise_video - video0)) ** 2)
    if audio0.numel() == 0 or audio_weight == 0:
        # A still, or a clip without sound: nothing to learn from the audio tokens.
        return video_mse, video_mse.detach(), torch.full((), float("nan"), device=video_mse.device)
    audio_mse = torch.mean((pred_audio - (noise_audio - audio0)) ** 2)
    return video_mse + audio_weight * audio_mse, video_mse.detach(), audio_mse.detach()
