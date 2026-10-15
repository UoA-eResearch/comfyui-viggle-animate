# ComfyUI + Viggle-Animate

Put a character from a picture into a video, and change the scene around it. This container is
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) with everything set up for two video models and their
combination, ready-to-run workflows that take a driving video and a reference image, AI-generated example videos
and characters, and an API you can script against (jobs from several people queue up):

- **[Viggle-Animate](https://huggingface.co/Viggle/Viggle-Animate)** keeps the motion, expressions and camera of
  the driving video and replaces the person in it with your character, with lip-sync to the video's soundtrack.
- **[LTX Ripple](https://huggingface.co/WepeNerd/LTX-Ripple)** (an IC-LoRA for
  [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)) carries any edit of the first frame through the video:
  characters, scenery, light, style.
- **Both together**: Viggle-Animate renders the character with lip-sync, Ripple the changed scene, and SAM 3.1
  lays one over the other.

(This page shows no example results on purpose: Viggle-Animate's licence does not allow its outputs to be
displayed in some countries, and a public page is shown everywhere. Run the bundled examples to see them.)

## Quick start

1. LTX-2.5 is gated: accept its licence at <https://huggingface.co/Lightricks/LTX-2.5> with your Hugging Face
   account, and create a [read token](https://huggingface.co/settings/tokens).
2. Start the container:

   ```bash
   docker run -d --name viggle --gpus all -p 8188:8188 -v viggle:/workspace -e HF_TOKEN=hf_... ghcr.io/uoa-eresearch/comfyui-viggle-animate:latest
   ```

3. Open <http://localhost:8188>, then **Workflows** (left sidebar), pick one of the three workflows and press
   **Run**. The example inputs are already selected.

On first start the container downloads the models from Hugging Face into the `viggle` volume (about 175 GB with
the full Viggle-Animate model, see below) while ComfyUI is already up. Follow it with `docker logs -f viggle`;
when it prints `all models ready`, refresh the page. Later starts reuse the volume and take seconds.

Without `HF_TOKEN` the container still works, with the Viggle-Animate workflow only (about 105 GB to download).

## Requirements

- Linux with an NVIDIA GPU, driver 570 or newer (CUDA 12.8), and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- GPU memory decides which Viggle-Animate build is downloaded (`VIGGLE_VARIANT=auto`):

  | GPU memory | Viggle-Animate build | Download: Viggle workflow only / all three |
  |---|---|---|
  | 80 GB or more (A100/H100 80 GB, RTX PRO 6000 96 GB) | full bf16 (66 GB) | ~105 GB / ~175 GB |
  | 48 GB or more (A6000, L40S, RTX 6000 Ada) | int8 (47 GB) | ~86 GB / ~156 GB |
  | less | pruned int8 (21 GB) | ~60 GB / ~130 GB |

  The thresholds are approximate: ComfyUI offloads what does not fit into GPU memory, which works but is slower.
  On the example clips the three builds give very similar results. LTX-2.5 (22B, bf16) needs about 48 GB of GPU
  memory, or offloading on smaller cards. Leave room on the volume for your outputs too.
- 64 GB of system RAM or more is recommended (the models are large).

## What is inside

| | |
|---|---|
| Workflows | All three take a driving video and a reference image. Step 1 paints your character into the video's first frame (Qwen-Image-2.1 + the Viggle turbo LoRA, after Qwen3-VL describes your picture). **Viggle-Animate (video + reference image)**: Viggle-Animate renders the whole video from that frame in 124-frame windows, with the soundtrack pinned for lip-sync. **LTX Ripple (video + reference image)**: LTX-2.5 + Ripple carries the edited frame through the video. **Viggle-Animate + LTX Ripple (character + scene)**: step 1 also changes the scene as you describe it; Viggle-Animate renders the character, Ripple the scene, and SAM 3.1 finds the character to composite it over Ripple's background. Each workflow has a note explaining its settings. |
| Examples | 5 driving videos (3 with speech) and 6 characters, AI-generated (`examples/`), copied into ComfyUI's `input/` |
| Nodes | [ComfyUI-Viggle-Animate-H3](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3), the lip-sync nodes of this repository (`custom_nodes/ViggleAudioPin`), VideoHelperSuite, KJNodes, Crystools and KikoStats (GPU use in the menu bar) |
| Base | Ubuntu 24.04, Python 3.12, PyTorch 2.11 (CUDA 12.8), ComfyUI 0.36 |

Everything you create stays in the volume: `/workspace/input`, `/workspace/output`, `/workspace/user` (your
workflows and settings), `/workspace/models`. The bundled workflows are refreshed on each start; save your own
versions under another name.

## Settings

| Environment variable | Default | |
|---|---|---|
| `HF_TOKEN` | - | Hugging Face token. Needed for LTX-2.5 (accept its licence first); also raises download rate limits. |
| `LTX` | `auto` | `auto`: LTX-2.5, Ripple and SAM 3.1 (and the two workflows that use them) when `HF_TOKEN` is set; `1` / `0` to force. |
| `VIGGLE_VARIANT` | `auto` | `bf16`, `int8` or `pruned-int8`; `auto` picks by GPU memory. |
| `DOWNLOAD_FIRST` | `0` | `1`: finish all downloads before ComfyUI starts. |
| `PORT` | `8188` | ComfyUI's port inside the container. |
| `COMFYUI_ARGS` | - | Extra ComfyUI options, e.g. `--use-sage-attention` if you add SageAttention, `--lowvram`. |

`docker-compose.yml` in this repository has the same setup for `docker compose up -d`.

## Speed

Measured on an RTX PRO 6000 Blackwell (96 GB), full bf16 models, for the 5 s example clips (121 frames), from
pressing Run to the saved video. Each workflow loads its models one after the other (they do not fit together),
from the disk cache after the first run:

| Workflow | Time |
|---|---|
| Viggle-Animate | 90-125 s (int8 build: 80 s, pruned int8: 60 s) |
| LTX Ripple | about 100 s |
| Viggle-Animate + LTX Ripple | about 170 s |

## Tips

- Driving videos: one clearly visible main person works best; the whole person should be in the first frame
  (the character is painted into that frame). Any length works; it is rendered in overlapping 5 s windows.
- Reference images: a full-body picture on a plain background works best. Animals, toys and drawings work too.
- If the first frame is not right, add a sentence to the extra-instructions text box ("keep the hat", "the
  character holds the phone in the left hand") or change the seed; only the changed steps re-run.
- Results are in `output/viggle/`, `output/ripple/` and `output/combined/`, each with a comparison video, with
  the original soundtrack.
- After a run ComfyUI keeps the last model in GPU memory (up to ~90 GB with the full builds), which makes the next
  run faster. On a GPU shared with other work, free it with
  `curl -X POST http://localhost:8188/free -H 'Content-Type: application/json' -d '{"unload_models": true, "free_memory": true}'`.

## Using the API (scripts, queues)

ComfyUI queues every job, so several people can submit at once. `tools/run_api.py` (standard-library Python)
uploads a video and an image, queues the bundled workflow and downloads the results:

```bash
python3 tools/run_api.py --host http://localhost:8188 --video my_clip.mp4 --image my_character.png --out results/
python3 tools/run_api.py --workflow ltx_ripple --video my_clip.mp4 --image my_character.png --extra "Keep the red scarf."
python3 tools/run_api.py --workflow viggle_animate_ltx_ripple --video my_clip.mp4 --image my_character.png \
    --extra "Turn the room into a spaceship cabin with round windows."
```

The API graphs it sends are in `workflows/api/`; see ComfyUI's
[API documentation](https://docs.comfy.org/development/comfyui-server/comms_routes) for more.

## Licences and responsible use

**Read [NOTICE.md](NOTICE.md) before using this.** The code here is GPL-3.0; the example media are CC0. The image
contains no model weights: the container downloads them for you from Hugging Face, and you use them under their
own licences:

- **Viggle-Animate and MiniMax-H3: [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
  It does not cover the European Union, the United Kingdom, the United States or South Korea, for the models or
  their outputs**, unless MiniMax grants a separate licence. It requires AI-generated content you publish to be
  disclosed as such, and anyone who lets others generate with it (a shared server) to have safeguards against
  misuse and a way to report it.
- **Qwen-Image-2.1 and the Viggle turbo LoRA: [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)**,
  research and evaluation only.
- **LTX-2.5 and LTX Ripple: [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)**,
  with a paid licence needed by entities over US$10 M revenue for commercial use.
- **SAM 3.1: [SAM License](https://huggingface.co/Comfy-Org/sam3.1/blob/main/LICENSE)** (Meta).

This tool changes who appears in a video. Only use footage of people who agreed to it, do not use it to
impersonate or deceive, and label what you make as AI-generated.

## Building and publishing

```bash
docker build -t comfyui-viggle-animate .
```

`.github/workflows/publish.yml` builds the image on GitHub Actions, checks that ComfyUI starts and every node pack
loads (CPU only), and publishes it to the GitHub Container Registry: pushes to `main` publish `:latest`, tags
`v1.2.3` publish `:1.2.3`. After the first publish, set the package's visibility to public (the package's
settings page on GitHub) so `docker pull` works without logging in.

To regenerate the bundled workflows after changing `tools/build_workflows.py`, start the container with LTX
enabled and run `python3 tools/build_workflows.py http://localhost:8188`.

## Credits

[Viggle](https://huggingface.co/Viggle) (Viggle-Animate, the turbo LoRA), [MiniMax](https://huggingface.co/MiniMaxAI)
(MiniMax-H3), [drbaph](https://huggingface.co/drbaph/Viggle-Animate-ComfyUI) (the ComfyUI conversion),
[Saganaki22](https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3) (the Viggle nodes),
[Comfy-Org](https://huggingface.co/Comfy-Org) (ComfyUI and the repackaged weights), [Qwen](https://huggingface.co/Qwen),
[Lightricks](https://huggingface.co/Lightricks) (LTX-2.5), [WepeNerd](https://huggingface.co/WepeNerd/LTX-Ripple)
(LTX Ripple), [Meta](https://ai.meta.com/sam3/) (SAM 3.1), and the authors of VideoHelperSuite, KJNodes, Crystools and KikoStats. The idea of a ready-to-run
ComfyUI image that fetches its models into a volume comes from community templates such as
[HearmemanAI's](https://github.com/Hearmeman24); this repository uses none of their code.
