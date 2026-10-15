#!/usr/bin/env bash
# Container entrypoint: prepare /workspace, start the model downloads, start ComfyUI.
set -euo pipefail
V=/opt/viggle
W=${WORKSPACE:-/workspace}
export WORKSPACE=$W

cat <<'EOF'
==============================================================================================================
 ComfyUI + Viggle-Animate                     source, licences and help: see README.md / NOTICE.md in the repo

 The models are downloaded from Hugging Face into /workspace, and by downloading and using them you accept their
 licences. In particular:
   * Viggle-Animate / MiniMax-H3: MiniMax H3 Community License. It does NOT cover use in the European Union,
     the United Kingdom, the United States or South Korea, and it requires AI-generated content to be disclosed
     as such when published.
   * Qwen-Image-2.1 and the Viggle turbo LoRA: Qwen Research License (research and evaluation only).
   * LTX-2.5 (optional): LTX-2 Community License (gated on Hugging Face).
 Only use footage of people who agreed to it.
==============================================================================================================
EOF

mkdir -p "$W"/{models,input,output,temp,custom_nodes,user/default/workflows,.cache}
python "$V/setup_workspace.py" prepare

if [ "${DOWNLOAD_FIRST:-0}" = 1 ]; then
  python "$V/setup_workspace.py" models
else
  # ComfyUI comes up straight away; the model lists fill in as the downloads finish (refresh the page).
  echo "[start] downloading models in the background; follow with: docker logs -f <container>"
  python "$V/setup_workspace.py" models &
fi

cd /opt/ComfyUI
# shellcheck disable=SC2086
exec python main.py --listen "${LISTEN:-0.0.0.0}" --port "${PORT:-8188}" --disable-auto-launch \
  --input-directory "$W/input" --output-directory "$W/output" --temp-directory "$W/temp" \
  --user-directory "$W/user" --extra-model-paths-config "$V/extra_model_paths.yaml" \
  ${COMFYUI_ARGS:-}
