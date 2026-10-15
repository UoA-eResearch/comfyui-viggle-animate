# syntax=docker/dockerfile:1.7
#
# ComfyUI with Viggle-Animate (video character replacement with lip-sync), Qwen-Image-2.1 + the Viggle turbo LoRA
# (to paint the new character into the first frame) and, optionally, LTX-2.5 + LTX Ripple.
#
# The image holds code only: ComfyUI and the node packs, pinned. Model weights are downloaded from Hugging Face
# into the /workspace volume on first start (docker/download.py), so each user accepts the model licences
# themselves (see NOTICE.md).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_HIGH_PERFORMANCE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev build-essential git \
      ffmpeg ca-certificates curl tini libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && python3.12 -m venv /opt/venv \
 && pip install --upgrade pip wheel setuptools

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
# PyTorch for CUDA 13.0 (NVIDIA driver 580 or newer; Ampere to Blackwell), which ComfyUI's optimised kernels
# (comfy-kitchen: int8 and fp8 models) need. The constraint file keeps every later
# install from replacing it.
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
      --index-url ${TORCH_INDEX} \
 && pip freeze | grep -E '^(torch|torchvision|torchaudio)==' > /opt/torch-constraints.txt
ENV PIP_CONSTRAINT=/opt/torch-constraints.txt

# ComfyUI 0.36.0 + the MiniMax-H3 core nodes, the base the Viggle-Animate node pack is tested on
ARG COMFYUI_REF=3dd559d81f745747cab884a3b9f5fd8867d79efe
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
 && git -C /opt/ComfyUI checkout -q ${COMFYUI_REF} \
 && pip install -r /opt/ComfyUI/requirements.txt "huggingface_hub[hf_xet]==1.33.0"

# Node packs, pinned: Viggle-Animate (Apache-2.0), VideoHelperSuite (GPL-3.0, video in/out), KJNodes (GPL-3.0,
# image utilities and the VRAM unload step), Crystools and KikoStats (MIT, GPU monitors in the menu bar).
COPY docker/nodes.txt /opt/viggle/nodes.txt
RUN set -e; cd /opt/ComfyUI/custom_nodes; \
    grep -v '^#' /opt/viggle/nodes.txt | while read -r dir url ref; do \
      [ -n "$dir" ] || continue; \
      git clone -q "$url" "$dir"; git -C "$dir" checkout -q "$ref"; \
      [ ! -f "$dir/requirements.txt" ] || pip install -r "$dir/requirements.txt"; \
    done

# Lip-sync nodes (this repository)
COPY custom_nodes/ViggleAudioPin /opt/ComfyUI/custom_nodes/ViggleAudioPin

COPY docker/start.sh docker/setup_workspace.py docker/models.json docker/extra_model_paths.yaml /opt/viggle/
COPY workflows /opt/viggle/workflows
COPY examples /opt/viggle/examples
COPY LICENSE NOTICE.md /opt/viggle/
RUN chmod +x /opt/viggle/start.sh \
 && python -c "import torch, torchvision, torchaudio; print('torch', torch.__version__)"

EXPOSE 8188
VOLUME /workspace
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/viggle/start.sh"]
