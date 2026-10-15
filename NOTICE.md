# Notices

## This repository

The code in this repository (Dockerfile, scripts, workflows, the `ViggleAudioPin` nodes) is licensed under the
GNU General Public License v3.0 (`LICENSE`). The example media in `examples/` are AI-generated and released under
CC0 1.0 (`examples/README.md`).

`custom_nodes/ViggleAudioPin/__init__.py` contains a window sampler adapted from
[ComfyUI-Viggle-Animate-H3](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3) (Copyright Saganaki22,
Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0): the audio rows of the latent are masked as
clean instead of generated. Modified by this repository's authors, 2026.

The design of the container (models downloaded into a volume on first start, workflows bundled) follows the
general approach of community RunPod templates such as HearmemanAI's; no code from those templates is used.

## Software in the image

Installed from upstream at pinned commits (`Dockerfile`, `docker/nodes.txt`); each keeps its own licence:

| Component | Licence |
|---|---|
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | GPL-3.0 |
| [ComfyUI-Viggle-Animate-H3](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3) | Apache-2.0 |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | GPL-3.0 |
| [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | GPL-3.0 |
| [ComfyUI-Crystools](https://github.com/crystian/ComfyUI-Crystools) | MIT |
| [ComfyUI-KikoStats](https://github.com/ComfyAssets/ComfyUI-KikoStats) | MIT |
| [PyTorch](https://pytorch.org) (with NVIDIA CUDA libraries from PyPI) | BSD-3-Clause; NVIDIA libraries under NVIDIA's licences |
| Ubuntu 24.04, FFmpeg and other system packages | their respective licences |

Downloaded at start, not part of the image:

| Component | Licence |
|---|---|
| `viggle_turbo.py` nodes from [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) | Qwen Research License |
| [ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo) (only with LTX enabled) | LTX-2 Community License |

## Models (downloaded by you, from Hugging Face, on first start)

The image contains no model weights. On first start the container downloads them from Hugging Face into your
`/workspace` volume; by doing so you obtain them from their publishers under their licences, and you are
responsible for complying with them.

| Model | Publisher | Licence |
|---|---|---|
| Viggle-Animate (ComfyUI conversion [drbaph/Viggle-Animate-ComfyUI](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI) of [Viggle/Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate)) | Viggle | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) |
| MiniMax-H3 video and audio VAEs ([Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)) | MiniMax | MiniMax H3 Community License |
| Qwen-Image-2.1, Qwen3-VL 8B text encoder, VAE ([Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)) | Qwen / Alibaba | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) (non-commercial) |
| Viggle turbo LoRA ([Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)) | Viggle | Qwen Research License |
| LTX-2.5 (optional, gated) ([Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)) | Lightricks | [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x) |
| LTX Ripple IC-LoRA (optional) ([WepeNerd/LTX-Ripple](https://huggingface.co/WepeNerd/LTX-Ripple)) | WepeNerd | LTX-2 Community License |
| SAM 3.1 (with LTX) ([Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1)) | Meta | [SAM License](https://huggingface.co/Comfy-Org/sam3.1/blob/main/LICENSE) |

Points of these licences that users most often miss (read the licences themselves; this is not legal advice):

- **MiniMax H3 Community License** (covers Viggle-Animate): the licence applies worldwide **except the European
  Union, the United Kingdom, the United States and South Korea**; using, distributing or displaying the models
  *or their outputs* in those territories is not licensed (MiniMax invites requests at api@minimax.io).
  Published AI-generated content must be clearly disclosed as machine-generated. If you let other people generate
  with it (for example a shared server), you must have reasonable safeguards against misuse and a way to report it.
  Commercial products over US$20 M revenue need MiniMax's authorisation and must show "MiniMax H3" in the UI.
  "MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights
  Reserved."
- **Qwen Research License**: research and evaluation only; commercial use needs a licence from Qwen.
  "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory
  Technology Co., Ltd. All Rights Reserved."
- **LTX-2 Community License**: free for individuals and entities under US$10 M annual revenue; larger entities
  need a paid licence except for non-commercial research and evaluation. Keep the AI-transparency features
  (disclosures, metadata, watermarks) of outputs intact.
