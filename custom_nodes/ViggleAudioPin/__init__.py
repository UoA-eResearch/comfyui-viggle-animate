"""Lip-sync for Viggle-Animate in ComfyUI: pin a real soundtrack in the audio rows of the AV latent.

The Viggle nodes (ComfyUI-Viggle-Animate-H3) leave the target audio rows to the model, and the frozen prompt
asks for silence, so the mouth has nothing to follow. The model card's fix (inference/sample.py --audio)
encodes the soundtrack with the H3 audio VAE and holds it in those rows as a CLEAN latent for the whole
denoise: the rows are labelled clean (t = 1) and never stepped. ComfyUI's MiniMax-H3 already does exactly
that for latent rows whose noise mask is 0 (comfy/model_base.py MiniMaxH3: audio_denoise_mask pins them at
the audio cond timestep and re-injects them clean every step), so these nodes only have to put the encoded
soundtrack into the audio half of the latent and mask it.

  Viggle Audio Pin (H3)                 after Viggle-Animate Conditioning (H3), before the sampler
  Viggle Chunked Sampler (audio pinned) the windowed sampler, with the whole clip's soundtrack pinned
  Viggle Chunked Sampler (masked)       the same, replacing only what a mask covers (one person, say)

As in sample.py the soundtrack is cut to the rendered length plus one frame and padded with silence; it
starts at the first rendered frame (`start_seconds` skips into it). Mux the original soundtrack into the
saved video as usual: the model's audio is the pinned one back through the VAE.

Masked: Viggle-Animate takes no mask, but the same mechanism holds video rows too. With `keep_background` on, each
window's video rows are the driving clip's own latent and only the rows the mask covers are generated; the rest are
clean conditioning, as the carried overlap between windows already is (the approach of the MiniMax-H3 inpainting
pipeline, diffusers-modular/minimax-h3-inpainting). Viggle was not trained this way: for a change as large as a
person into a very different one it tends to keep the original, half replaced. Off (the default), the whole frame is
generated as without a mask, and the workflow composites the replaced person back through the mask.

Licence: GPL-3.0 (this repository). ViggleChunkedSamplerAudioPin._sample_window is adapted from the window sampler
of ComfyUI-Viggle-Animate-H3 (https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3, Copyright Saganaki22,
Apache License 2.0); changed to mask the audio rows as clean instead of generated, and the video rows outside a mask.
"""
import copy
import hashlib
import sys
import types

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import latent_preview
import nodes
from comfy_extras import nodes_minimax_h3 as core_h3

FPS = 24
LATENTS_PER_SECOND = 40          # 32 kHz audio VAE, hop 800
FRAMES_PER_LATENT = (1, 4, 4, 4, 4)    # MiniMax-H3 video VAE: 17-frame chunks -> 5 latent frames


def mask_rows(mask, first_frame, latn, ch, cw):
    """MASK [N, H, W] (1 = replace) -> video-row mask [1, 1, latn, ch/16, cw/16] for the window starting at pixel
    frame `first_frame` (windows start on a 17-frame chunk boundary). A latent row is generated if the mask touches
    any of its pixels in any of its frames; frames past the end of the mask repeat its last frame."""
    m = mask.float()
    if m.dim() == 2:
        m = m[None]
    rows, f = [], first_frame
    for k in range(latn):
        n = FRAMES_PER_LATENT[k % 5]
        idx = torch.arange(f, f + n).clamp(max=m.shape[0] - 1)
        rows.append(m[idx].amax(dim=0))
        f += n
    m = torch.stack(rows)[:, None]                                            # [latn, 1, H, W]
    m = torch.nn.functional.interpolate(m, size=(ch, cw), mode="bilinear", align_corners=False)
    m = torch.nn.functional.max_pool2d(m, 16)                                 # [latn, 1, ch/16, cw/16]
    return (m > 0.5).float().permute(1, 0, 2, 3)[None]


def encode_soundtrack(audio_vae, audio, n_latents, start_seconds=0.0):
    """AUDIO -> normalised audio latent [1, 32, 2, n_latents] covering n_latents / 40 s from start_seconds."""
    wav = audio["waveform"][:1].float()                          # [1, C, L]
    sr = int(audio["sample_rate"])
    if wav.shape[1] == 1:
        wav = wav.repeat(1, 2, 1)
    wav = wav[:, :2]
    start = int(round(start_seconds * sr))
    need = int(round((n_latents / LATENTS_PER_SECOND + 1.0 / FPS) * sr))     # one frame spare, as sample.py
    wav = wav[..., start:start + need]
    wav = torch.nn.functional.pad(wav, (0, max(0, need - wav.shape[-1])))     # the tail is real silence
    z, _ = core_h3._encode_ref_audio(audio_vae, {"waveform": wav, "sample_rate": sr})
    if z.shape[-1] < n_latents:
        raise RuntimeError(f"soundtrack encoded to {z.shape[-1]} latents, short of {n_latents}")
    return z[..., :n_latents].float()


class ViggleAudioPin:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT", {"tooltip": "The AV latent from Viggle-Animate Conditioning (H3)."}),
            "audio_vae": ("VAE", {"tooltip": "MiniMax-H3 audio VAE (minimax_h3_audio_vae_fp32)."}),
            "audio": ("AUDIO", {"tooltip": "The soundtrack the mouth should follow, usually the driving clip's own "
                                           "(VHS Load Video's audio output)."}),
            "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01,
                                        "tooltip": "Where in the audio the first rendered frame is. 0 when the audio "
                                                   "comes from the same loader as the frames."}),
            "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off: the model makes its own (silent) soundtrack "
                                                                "and the mouth is not synced, as without --audio."}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "pin"
    CATEGORY = "sampling/viggle"
    DESCRIPTION = ("Lip-sync: hold the soundtrack in the latent's audio rows as clean conditioning for the whole "
                   "denoise (Viggle-Animate sample.py --audio), so the mouth follows the speech.")

    def pin(self, latent, audio_vae, audio, start_seconds, enabled=True):
        if not enabled:
            return (latent,)
        try:
            audio = {"waveform": audio["waveform"], "sample_rate": audio["sample_rate"]}
        except Exception as e:      # a clip without a soundtrack: nothing to sync to
            print(f"Viggle Audio Pin: no soundtrack ({str(e).splitlines()[0]}); not pinned")
            return (latent,)
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError("Viggle Audio Pin needs the joint video+audio latent from Viggle-Animate Conditioning")
        video, aud = samples.unbind()[:2]
        z = encode_soundtrack(audio_vae, audio, aud.shape[-1], start_seconds).to(aud.device, aud.dtype)
        mask_v = torch.ones([1, 1] + list(video.shape[2:]), device=video.device)
        mask_a = torch.zeros([1, 1, 1, aud.shape[-1]], device=aud.device)
        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, z))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((mask_v, mask_a))
        return (out,)


class ViggleChunkedSamplerAudioPin:
    """The Viggle Chunked Sampler with the whole clip's soundtrack pinned in every chunk's audio rows."""

    @classmethod
    def _base(cls):
        return nodes.NODE_CLASS_MAPPINGS["ViggleChunkedSampler"]

    @classmethod
    def INPUT_TYPES(cls):
        spec = copy.deepcopy(cls._base().INPUT_TYPES())
        spec["required"]["audio_vae"] = ("VAE", {"tooltip": "MiniMax-H3 audio VAE (minimax_h3_audio_vae_fp32)."})
        spec["required"]["audio"] = ("AUDIO", {"tooltip": "The driving clip's soundtrack (same start as its frames)."})
        spec["required"]["start_seconds"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01})
        return spec

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "chunk_map")
    FUNCTION = "sample"
    CATEGORY = "sampling/viggle"
    DESCRIPTION = "Chunked Viggle-Animate sampler with lip-sync: the soundtrack is pinned in every chunk."

    def sample(self, guider, sampler, sigmas, cond_set, vae, seed, rerender_chunk, rerender_seed,
               audio_vae, audio, start_seconds, mask=None, keep_background=True, dynprompt=None, unique_id=None):
        base_cls = self._base()
        base_mod = sys.modules[base_cls.__module__]
        inst = base_cls()
        total_a = round(int(cond_set["total_frames"]) / FPS * LATENTS_PER_SECOND)
        pinned = encode_soundtrack(audio_vae, audio, total_a, start_seconds)
        h = hashlib.sha256(pinned.cpu().numpy().tobytes())
        if mask is not None:                      # chunks rendered with another mask must not be reused either
            h.update(b"keep" if keep_background else b"whole")
            h.update((mask > 0.5).cpu().numpy().tobytes())
        digest = h.digest()
        parent_render, parent_key = inst._render_chunk, inst._chunk_key

        def _render_chunk(self, noise, guider, sampler, sigmas, cond, seed_i, ch, cw, a, b, lat0, latn,
                          carry, master_v, master_a):
            master_a.copy_(pinned[..., :master_a.shape[-1]].to(master_a.device, master_a.dtype))
            return parent_render(noise, guider, sampler, sigmas, cond, seed_i, ch, cw, a, b, lat0, latn,
                                 carry, master_v, master_a)

        def _sample_window(self, noise, guider, sampler, sigmas, cond, seed_i, ch, cw, a, b, carry, v, au):
            # the stock window sampler, with the audio rows masked 0 (pinned) instead of 1 (generated)
            latn = v.shape[2]
            mask_v = torch.ones([1, 1, latn, ch // 16, cw // 16], device=v.device)
            if mask is not None and keep_background:
                # the video rows outside the mask: the driving clip's own latent, held like the carried overlap
                footage = cond[0][1]["minimax_refs"][0]["latent"]
                if tuple(footage.shape[2:]) != tuple(v.shape[2:]):
                    raise ValueError("Viggle Chunked Sampler (masked): the driving clip and the canvas differ in size "
                                     f"({tuple(footage.shape[2:])} vs {tuple(v.shape[2:])}); set Viggle-Animate "
                                     "Conditioning's width and height to 0")
                v = v.clone()
                v[:, :, carry:] = footage[:, :, carry:].to(v.device, v.dtype)
                mask_v = mask_rows(mask, a, latn, ch, cw).to(v.device)
            mask_v[:, :, :carry] = 0.0
            samples = comfy.sample.fix_empty_latent_channels(guider.model_patcher,
                                                             comfy.nested_tensor.NestedTensor((v, au)))
            mask_a = torch.zeros([1, 1, 1, au.shape[-1]], device=au.device)
            chunk_noise = copy.copy(noise)
            chunk_noise.seed = seed_i
            callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
            out = base_mod._chunk_guider(guider, cond).sample(
                chunk_noise.generate_noise({"samples": samples}), samples, sampler, sigmas,
                denoise_mask=comfy.nested_tensor.NestedTensor((mask_v, mask_a)), callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed_i)
            out_v, out_a = out.unbind() if out.is_nested else (out, None)
            if not torch.isfinite(out_v).all():
                raise RuntimeError(f"Viggle Chunked Sampler (audio pinned): frames {a}-{b} produced NaN/Inf latents")
            return out_v, out_a

        def _chunk_key(self, *args):
            key = parent_key(*args)          # chunks rendered without this soundtrack must not be reused
            return None if key is None else hashlib.sha256(key + digest).digest()

        inst._render_chunk = types.MethodType(_render_chunk, inst)
        inst._sample_window = types.MethodType(_sample_window, inst)
        inst._chunk_key = types.MethodType(_chunk_key, inst)
        return inst.sample(guider, sampler, sigmas, cond_set, vae, seed, rerender_chunk, rerender_seed,
                           dynprompt=dynprompt, unique_id=unique_id)


class ViggleChunkedSamplerMasked(ViggleChunkedSamplerAudioPin):
    """The lip-synced chunked sampler, replacing only what a mask covers."""

    @classmethod
    def INPUT_TYPES(cls):
        spec = super().INPUT_TYPES()
        spec["required"]["mask"] = ("MASK", {"tooltip": "1 = replace (the person; from SAM 3.1, grown well past the "
                                                        "outline). One mask per frame of the driving clip, at its "
                                                        "size."})
        spec["required"]["keep_background"] = (
            "BOOLEAN", {"default": False,
                        "tooltip": "Off (default): the whole frame is generated, as without a mask; composite it back "
                                   "through the mask. On: only the masked area is generated and the rest of every "
                                   "frame is held as the driving clip, which gives seamless edges when the new "
                                   "character is like the person (similar build and clothing), but for a very "
                                   "different one (another gender, a knight in armour) Viggle-Animate tends to keep "
                                   "the original person, half replaced."})
        return spec

    DESCRIPTION = ("Chunked Viggle-Animate sampler with lip-sync that replaces only what the mask covers: the rest "
                   "of the frame stays the driving clip.")


NODE_CLASS_MAPPINGS = {"ViggleAudioPin": ViggleAudioPin,
                       "ViggleChunkedSamplerAudioPin": ViggleChunkedSamplerAudioPin,
                       "ViggleChunkedSamplerMasked": ViggleChunkedSamplerMasked}
NODE_DISPLAY_NAME_MAPPINGS = {"ViggleAudioPin": "Viggle Audio Pin (H3, lip-sync)",
                              "ViggleChunkedSamplerAudioPin": "Viggle Chunked Sampler (audio pinned, lip-sync)",
                              "ViggleChunkedSamplerMasked": "Viggle Chunked Sampler (masked, lip-sync)"}
