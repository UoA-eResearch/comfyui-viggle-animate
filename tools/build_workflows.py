"""Build the bundled workflows (workflows/*.json for the ComfyUI Workflows tab, workflows/api/*.json for the API).

The graphs are written here in Python; the UI layout needs the node definitions of a running instance of this
image (its /object_info), with the LTX node pack loaded (start it with HF_TOKEN set):

  python3 tools/build_workflows.py http://localhost:8188

Standard library only.
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open

VIGGLE_MODEL = "minimax_h3_ref2va_viggle_bf16.safetensors"     # swapped for the installed variant on start
VIGGLE_LORA = "viggle_animate_dmd_lora.safetensors"
TEXT_COND = "fixed_embed_fwd_anyframe.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
VIGGLE_SIGMAS = "1.0, 0.8571428571428571, 0.6, 0.0"            # the model card's 4 steps = 3 Euler updates
QWEN_MODEL = "qwen_image_2.1_bf16.safetensors"
QWEN_TE = "qwen3vl_8b_bf16.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
TURBO_LORA = "Qwen-Image-2.1-viggle-turbo-v0.3-6step-lora-r128.safetensors"
TURBO_SIGMAS = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"
LTX_MODEL = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
LTX_TE = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
LTX_VAE = "ltx-2.5-video-vae-conv-bf16.safetensors"
LTX_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
RIPPLE_LORA = "LTX25_Ripple_v11.safetensors"
SAM_MODEL = "sam3.1_multiplex_fp16.safetensors"
RIPPLE_PROMPT = ("Use the reference video for motion, timing, camera movement, composition, and unchanged scene "
                 "content, while consistently propagating the visual edit established in the first frame throughout "
                 "the video.")
RIPPLE_NEGATIVE = ("blurry, low resolution, ugly, garbled mess, compression artifacts, glitch, stop motion, still "
                   "image, pixelated, robotic voice, oversaturated")
DESCRIBE = ("Describe the main character in this image in one sentence, for a painter who must paint exactly this "
            "character into another photo: what it is (a person, an animal, a creature, a robot...), apparent age and "
            "gender if a person, face, hair or fur, build, clothing or costume, with colours and materials. Do not "
            "describe the background, the pose or the camera. Answer with the description only.")
EDIT_PREFIX = "Completely replace the main person in image 1 with this character: "
EDIT_SUFFIX = (". The character takes the exact pose, limbs, head angle, gaze and expression of the person it "
               "replaces, at the same size and position in the frame; none of the original person remains. Keep the "
               "background, the framing and the light of image 1.")


class G(dict):
    def add(self, cls, **inputs):
        nid = str(len(self) + 1)
        self[nid] = {"class_type": cls, "inputs": {k: (list(v) if isinstance(v, tuple) else v)
                                                    for k, v in inputs.items()}}
        return nid


def out(nid, slot=0):
    return (nid, slot)


def load_video(g, video, cap=0):
    return g.add("VHS_LoadVideo", video=video, force_rate=24, custom_width=0, custom_height=0,
                 frame_load_cap=cap, skip_first_frames=0, select_every_nth=1, format="None")


def save(g, images, video, prefix):
    return g.add("VHS_VideoCombine", images=out(images), audio=out(video, 2), frame_rate=24, loop_count=0,
                 filename_prefix=prefix, format="video/h264-mp4", pix_fmt="yuv420p", crf=16, save_metadata=False,
                 trim_to_audio=False, pingpong=False, save_output=True)


def unload_all(g, image):
    """Unload every model before the next one loads (KJNodes VRAM_Debug), so the editor and the video model are
    never in GPU memory together."""
    return out(g.add("VRAM_Debug", image_pass=out(image), empty_cache=True, gc_collect=True, unload_all_models=True), 1)


def repaint_first_frame(g, video, reference, seed=42, suffix=None, extra=""):
    """Step 1: the clip's first frame with the reference character painted in (Qwen-Image-2.1 + the Viggle turbo
    LoRA, 6 steps). The editor's own text encoder (Qwen3-VL) describes the reference image first, and the edit gets
    the frame plus that description: given the reference picture as well, the turbo model tended to return the
    reference photo's own scene."""
    m = g.add("UNETLoader", unet_name=QWEN_MODEL, weight_dtype="default")
    m = g.add("ViggleTurboLora", model=out(m), lora_name=TURBO_LORA, strength=1.0)
    m = g.add("QwenImage21Cache", model=out(m), device="auto", dtype="default")
    clip = g.add("CLIPLoader", clip_name=QWEN_TE, type="qwen_image", device="default")
    vae = g.add("VAELoader", vae_name=QWEN_VAE)
    first = unload_all(g, g.add("ImageFromBatch", image=out(video), batch_index=0, length=1))
    desc = g.add("TextGenerate", clip=out(clip), prompt=DESCRIBE, image=reference, max_length=160,
                 sampling_mode="off", thinking=False, use_default_template=True, mtp="auto")
    g.add("PreviewAny", source=out(desc))
    text = g.add("StringConcatenate", string_a=EDIT_PREFIX, string_b=out(desc), delimiter="")
    text = g.add("StringConcatenate", string_a=out(text), string_b=suffix or EDIT_SUFFIX, delimiter="")
    more = g.add("PrimitiveStringMultiline", value=extra)        # optional extra instructions for the editor
    text = g.add("StringConcatenate", string_a=out(text), string_b=out(more), delimiter=" ")
    enc = g.add("TextEncodeQwenImage21", clip=out(clip), prompt=out(text), negative_prompt="", resolution=1024,
                vae=out(vae), **{"images.image_1": first})
    s = g.add("SamplerCustomAdvanced", noise=out(g.add("RandomNoise", noise_seed=seed)),
              guider=out(g.add("BasicGuider", model=out(m), conditioning=out(enc))),
              sampler=out(g.add("KSamplerSelect", sampler_name="euler")),
              sigmas=out(g.add("ViggleTurboSigmas", latent=out(enc, 2), nodes=TURBO_SIGMAS)),
              latent_image=out(enc, 2))
    frame = g.add("VAEDecode", samples=out(s), vae=out(vae))
    g.add("SaveImage", images=out(frame), filename_prefix="viggle/frame")
    return unload_all(g, frame)


def viggle(driver="talking_kitchen.mp4", reference="knight.jpg", seed=42):
    """Driving video + reference image -> the clip with that character, lip-synced to the clip's soundtrack."""
    g = G()
    video = load_video(g, driver)
    frame = repaint_first_frame(g, video, out(g.add("LoadImage", image=reference)))
    small = to_lowres(g, video)
    result = viggle_branch(g, out(small), frame, video, seed)
    save(g, result, video, "viggle/result")
    side = g.add("ImageConcanate", image1=out(small), image2=out(result), direction="right", match_image_size=True)
    save(g, side, video, "viggle/side_by_side")
    return g


def ripple_branch(g, frames, edited, n, seed=123463, strength=1.35):
    """LTX Ripple (an IC-LoRA for LTX-2.5): `edited` (the edited first frame) is put in front of `frames` (the
    driving clip, sized for LTX: multiples of 32) as the IC-LoRA guide and carried through the clip. Settings as the
    LoRA's example workflow: distilled LTX-2.5, 8 steps, CFG 1, LoRA 1.35. Returns the decoded frames."""
    m = g.add("UNETLoader", unet_name=LTX_MODEL, weight_dtype="default")
    m = g.add("LoraLoaderModelOnly", model=out(m), lora_name=RIPPLE_LORA, strength_model=strength)
    clip = g.add("CLIPLoader", clip_name=LTX_TE, type="ltxv", device="default")
    vae = g.add("VAELoader", vae_name=LTX_VAE)
    avae = g.add("VAELoader", vae_name=LTX_AUDIO_VAE)
    cond = g.add("LTXVConditioning", positive=out(g.add("CLIPTextEncode", text=RIPPLE_PROMPT, clip=out(clip))),
                 negative=out(g.add("CLIPTextEncode", text=RIPPLE_NEGATIVE, clip=out(clip))), frame_rate=24.0)
    size = g.add("GetImageSize", image=frames)
    edited = g.add("ImageResizeKJv2", image=edited, width=out(size, 0), height=out(size, 1), upscale_method="lanczos",
                   keep_proportion="crop", pad_color="0, 0, 0", crop_position="center", divisible_by=32, device="cpu")
    guide = g.add("ImageFromBatch", batch_index=0, length=n,
                  image=out(g.add("BatchImagesNode", **{"images.image0": out(edited), "images.image1": frames})))
    ic = g.add("LTXAddVideoICLoRAGuide", positive=out(cond), negative=out(cond, 1), vae=out(vae),
               latent=out(g.add("EmptyLTXVLatentVideo", width=out(size, 0), height=out(size, 1), length=n,
                                batch_size=1)),
               image=out(guide), frame_idx=0, strength=1.0, latent_downscale_factor=1.0, crop="disabled",
               use_tiled_encode=False, tile_size=256, tile_overlap=64)
    av = g.add("LTXVConcatAVLatent", video_latent=out(ic, 2), audio_latent=out(
        g.add("LTXVEmptyLatentAudio", frames_number=n, frame_rate=24, batch_size=1, audio_vae=out(avae))))
    s = g.add("SamplerCustomAdvanced", noise=out(g.add("RandomNoise", noise_seed=seed)),
              guider=out(g.add("CFGGuider", model=out(m), positive=out(ic), negative=out(ic, 1), cfg=1.0)),
              sampler=out(g.add("KSamplerSelect", sampler_name="euler")),
              sigmas=out(g.add("BasicScheduler", model=out(m), scheduler="simple", steps=8, denoise=1.0)),
              latent_image=out(av))
    sep = g.add("LTXVSeparateAVLatent", av_latent=out(s))
    crop = g.add("LTXVCropGuides", positive=out(ic), negative=out(ic, 1), latent=out(sep))
    return g.add("VAEDecode", samples=out(crop, 2), vae=out(vae))


def viggle_branch(g, frames, frame, video, seed=42):
    """Viggle-Animate on `frames` (the driving clip at ~0.4 MP) with the repainted first frame `frame`, lip-synced
    to the clip's soundtrack. Returns the rendered frames."""
    m = g.add("UNETLoader", unet_name=VIGGLE_MODEL, weight_dtype="default")
    m = g.add("LoraLoaderModelOnly", model=out(m), lora_name=VIGGLE_LORA, strength_model=1.0)
    m = g.add("MiniMaxH3SigmaShift", model=out(m), shift_video=3.0, shift_audio=3.0)
    tc = g.add("ViggleTextCondLoader", text_cond=TEXT_COND)
    vvae = g.add("VAELoader", vae_name=VIDEO_VAE)
    avae = g.add("VAELoader", vae_name=AUDIO_VAE)
    cond = g.add("ViggleAnimateConditioningWindowed", cond_video=frames, ref_image=frame, text_cond=out(tc),
                 vae=out(vvae), width=0, height=0, chunk_frames=124, overlap_frames=22)
    rendered = g.add("ViggleChunkedSamplerAudioPin",
                     guider=out(g.add("BasicGuider", model=out(m), conditioning=out(cond, 1))),
                     sampler=out(g.add("KSamplerSelect", sampler_name="euler")),
                     sigmas=out(g.add("ManualSigmas", sigmas=VIGGLE_SIGMAS)), cond_set=out(cond), vae=out(vvae),
                     seed=seed, rerender_chunk=0, rerender_seed=0, audio_vae=out(avae), audio=out(video, 2),
                     start_seconds=0.0)
    # the sampler renders whole windows: back to the clip's own frame count
    return g.add("ImageFromBatch", image=out(rendered), batch_index=0, length=out(video, 1))


def to_lowres(g, video):
    """The clip at about 0.4 MP in multiples of 32: the size Viggle-Animate was trained at, and fine for LTX."""
    return g.add("ImageScaleToTotalPixels", image=out(video), upscale_method="lanczos", megapixels=0.39,
                 resolution_steps=32)


def ripple(driver="dance_studio.mp4", reference="robot.jpg", frames=121):
    """LTX Ripple with the same two inputs as the Viggle workflow: the first frame repainted with the reference,
    then carried through the clip. LTX frame counts are 8k+1."""
    g = G()
    n = out(g.add("PrimitiveInt", value=frames))                 # one place to set the length
    video = load_video(g, driver, cap=n)
    edited = repaint_first_frame(g, video, out(g.add("LoadImage", image=reference)))
    small = to_lowres(g, video)
    result = ripple_branch(g, out(small), edited, n)
    save(g, result, video, "ripple/result")
    side = g.add("ImageConcanate", image1=out(small), image2=out(result), direction="right", match_image_size=True)
    save(g, side, video, "ripple/side_by_side")
    return g


COMBINED_SUFFIX = (". The character takes the exact pose, limbs, head angle, gaze and expression of the person it "
                   "replaces, at the same size and position in the frame; none of the original person remains. Keep "
                   "the framing and the camera angle of image 1.")


def combined(driver="talking_kitchen.mp4", reference="knight.jpg", frames=121,
             scene="Turn the kitchen into a medieval castle hall with grey stone walls, banners and burning torches."):
    """Both models: one repainted first frame (the character and, optionally, a changed scene). Viggle-Animate
    renders the character with lip-sync; LTX Ripple carries the whole edited frame, background included, through
    the clip; SAM 3.1 finds the character in Viggle's render (and the person in the clip) and the character is
    composited over Ripple's background."""
    g = G()
    n = out(g.add("PrimitiveInt", value=frames))
    video = load_video(g, driver, cap=n)
    frame = repaint_first_frame(g, video, out(g.add("LoadImage", image=reference)), suffix=COMBINED_SUFFIX,
                                extra=scene)
    small = to_lowres(g, video)
    person = viggle_branch(g, out(small), frame, video)
    after_v = g.add("VRAM_Debug", any_input=out(person), image_pass=out(person), empty_cache=True, gc_collect=True,
                    unload_all_models=True)
    sam = g.add("CheckpointLoaderSimple", ckpt_name=SAM_MODEL)
    what = g.add("CLIPTextEncode", text="person", clip=out(sam, 1))     # what Viggle put in: person, robot, animal ...
    m1 = g.add("SAM3_Detect", model=out(sam), image=out(after_v, 1), conditioning=out(what), threshold=0.35,
               refine_iterations=1, individual_masks=False)
    m2 = g.add("SAM3_Detect", model=out(sam), image=out(small), conditioning=out(
        g.add("CLIPTextEncode", text="person", clip=out(sam, 1))), threshold=0.35, refine_iterations=1,
        individual_masks=False)
    mask = g.add("MaskComposite", destination=out(m1), source=out(m2), x=0, y=0, operation="or")
    mask = g.add("GrowMaskWithBlur", mask=out(mask), expand=10, incremental_expandrate=0.0, tapered_corners=True,
                 flip_input=False, blur_radius=6.0, lerp_alpha=1.0, decay_factor=1.0, fill_holes=True)
    edited = out(g.add("VRAM_Debug", any_input=out(mask), image_pass=frame, empty_cache=True, gc_collect=True,
                       unload_all_models=True), 1)
    scene_frames = ripple_branch(g, out(small), edited, n)
    result = g.add("ImageCompositeMasked", destination=out(scene_frames), source=out(after_v, 1), x=0, y=0,
                   resize_source=False, mask=out(mask))
    save(g, result, video, "combined/result")
    top = g.add("ImageConcanate", image1=out(small), image2=out(result), direction="right", match_image_size=True)
    bottom = g.add("ImageConcanate", image1=out(after_v, 1), image2=out(scene_frames), direction="right",
                   match_image_size=True)
    grid = g.add("ImageConcanate", image1=out(top), image2=out(bottom), direction="down", match_image_size=True)
    save(g, grid, video, "combined/grid")
    return g


# ---------------------------------------------------------------- UI workflows

def _widget(spec):
    t = spec[0]
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    return not opts.get("forceInput") and (isinstance(t, list) or t in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
                                           or str(t).startswith("COMFY_DYNAMICCOMBO"))


def to_ui(g, oi, notes=()):
    """API prompt -> UI workflow (nodes, links, layout) for the ComfyUI frontend. Widget values follow
    /object_info's input order."""
    ids = {k: i + 1 for i, k in enumerate(g)}
    depth = {}

    def d(k):
        if k not in depth:
            ups = [v[0] for v in g[k]["inputs"].values() if isinstance(v, list) and len(v) == 2 and v[0] in g]
            depth[k] = 1 + max((d(u) for u in ups), default=-1)
        return depth[k]
    for k in g:
        d(k)
    nodes, links, col_y, out_links = [], [], {}, {}
    for k, node in g.items():
        cls = node["class_type"]
        spec = {**oi[cls]["input"].get("required", {}), **oi[cls]["input"].get("optional", {})}
        inputs, widgets = [], []
        for name, sp in spec.items():
            v = node["inputs"].get(name)
            if isinstance(v, list) and len(v) == 2 and v[0] in g:
                lid = len(links) + 1
                typ = oi[g[v[0]]["class_type"]]["output"][v[1]]
                links.append([lid, ids[v[0]], v[1], ids[k], len(inputs), typ])
                out_links.setdefault((v[0], v[1]), []).append(lid)
                entry = {"name": name, "type": typ if not isinstance(sp[0], list) else "COMBO", "link": lid}
                if _widget(sp):
                    entry["widget"] = {"name": name}
                    widgets.append((sp[1] if len(sp) > 1 and isinstance(sp[1], dict) else {}).get("default"))
                inputs.append(entry)
            elif _widget(sp):
                widgets.append(v if v is not None else (sp[1] or {}).get("default") if len(sp) > 1 else None)
                if len(sp) > 1 and isinstance(sp[1], dict) and sp[1].get("control_after_generate"):
                    widgets.append("fixed")
        for name, v in node["inputs"].items():          # dynamic inputs (autogrow "images.image_1")
            if name not in spec and isinstance(v, list) and v[0] in g:
                lid = len(links) + 1
                typ = oi[g[v[0]]["class_type"]]["output"][v[1]]
                links.append([lid, ids[v[0]], v[1], ids[k], len(inputs), typ])
                out_links.setdefault((v[0], v[1]), []).append(lid)
                inputs.append({"name": name, "type": typ, "link": lid})
        if cls.startswith("VHS_"):
            widgets = {n: (v if not isinstance(v, list) else
                           (spec[n][1] if len(spec[n]) > 1 and isinstance(spec[n][1], dict) else {}).get("default"))
                       for n, v in node["inputs"].items() if n in spec or not isinstance(v, list)}
        elif cls == "LoadImage":
            widgets = [node["inputs"]["image"], "image"]
        n_w = len(widgets)
        h = 46 + 22 * max(len(inputs), len(oi[cls]["output"])) + 26 * n_w + (330 if cls.startswith("VHS_") else 0)
        h += 300 if cls == "LoadImage" else 0
        x = 40 + 390 * depth[k]
        y = col_y.get(depth[k], 40)
        col_y[depth[k]] = y + h + 40
        nodes.append({"id": ids[k], "type": cls, "pos": [x, y], "size": [350, h], "flags": {}, "order": 0,
                      "mode": 0, "inputs": inputs,
                      "outputs": [{"name": oi[cls]["output_name"][i], "type": t, "slot_index": i, "links": []}
                                  for i, t in enumerate(oi[cls]["output"])],
                      "properties": {"Node name for S&R": cls}, "widgets_values": widgets})
    for nd in nodes:
        k = list(g)[nd["id"] - 1]
        for o in nd["outputs"]:
            o["links"] = out_links.get((k, o["slot_index"]), []) or None
    for i, nd in enumerate(sorted(nodes, key=lambda n: depth[list(g)[n["id"] - 1]])):
        nd["order"] = i
    nid = len(nodes)
    for i, (text, size) in enumerate(notes):
        nid += 1
        nodes.append({"id": nid, "type": "MarkdownNote", "pos": [40 + i * 620, -60 - size[1]], "size": size,
                      "flags": {}, "order": nid, "mode": 0, "inputs": [], "outputs": [], "properties": {},
                      "widgets_values": [text], "color": "#432", "bgcolor": "#653"})
    return {"last_node_id": nid, "last_link_id": len(links), "nodes": nodes, "links": links, "groups": [],
            "config": {}, "extra": {"ds": {"scale": 0.65, "offset": [100, 700]}}, "version": 0.4}


STEPS = """1. **Load Video**: the driving clip, read at 24 fps. Examples: `talking_kitchen.mp4`, `talking_bench.mp4`,
   `presenter.mp4` (with speech), `dance_studio.mp4`, `wave_street.mp4`, or upload your own.
2. **Load Image**: who or what to put in: a photo, a drawing, an animal. Examples: `knight.jpg`, `robot.jpg`,
   `wizard.jpg`, `pirate.jpg`, `fox.jpg`, `gingerbread.jpg`.
3. **Queue.** Step 1, the clip's first frame with that character painted in, appears in *Save Image* (the
   editor's description of your image is in *Preview Any*). If it is not right, add a sentence in the extra
   instructions box (the empty text box) or change the seed and queue again: only the changed parts re-run."""
SHARED = ("If this ComfyUI is shared, everyone using it can see what is uploaded and rendered, and jobs run one at a "
          "time in one queue. Only use footage of people who agreed to it, and label the results as AI-generated "
          "when you publish them. Results are in `output/viggle/` and `output/ripple/`.")
NOTE_VIGGLE = f"""## Viggle-Animate: put a character from a picture into a video

{STEPS}
4. Step 2, Viggle-Animate, renders the whole clip in 124-frame windows. The clip's soundtrack is held in the model
   as clean audio, so mouths follow the speech (lip-sync, as the model card's `sample.py --audio`).

The clip is scaled to about 0.4 MP, the size Viggle-Animate was trained at. With the models loaded a 5 s clip
takes about a minute on an RTX PRO 6000; loading takes a minute or two the first time. Identity or light can
drift between windows on long clips. {SHARED}"""
NOTE_RIPPLE = f"""## LTX Ripple: the same two inputs, with LTX-2.5 and the LTX Ripple IC-LoRA

[LTX Ripple](https://huggingface.co/WepeNerd/LTX-Ripple) carries an edit of a video's *first* frame through the
video. It works for any kind of edit (light, style, objects, not just people).

{STEPS}
4. Step 2, LTX-2.5 distilled + Ripple at strength 1.35, 8 steps, propagates that frame through the clip.

**Frames** (the Int node) sets the length: 8k+1 frames (97, 121, 161 ...). Set the size on the first *Resize
Image* (multiples of 32). Raise the LoRA strength if the edit fades, lower it if things that should stay change.
The mouth follows the driving clip only. {SHARED}"""


NOTE_COMBINED = f"""## Viggle-Animate + LTX Ripple: a new character in a changed scene

Both models together. Step 1 paints the character into the first frame and can also change the scene: describe
the change in the text box (the example turns the kitchen into a castle hall; leave it empty to keep the scene).
Viggle-Animate renders the character with lip-sync, LTX Ripple carries the whole edited frame, background
included, through the clip, and SAM 3.1 finds the character so it can be laid over Ripple's background.

{STEPS}
4. Results in `output/combined/`: the composite, and a grid of the clip, the composite, Viggle's render and
   Ripple's render.

If parts of the character are cut off or a halo of the old background shows, change what SAM looks for (the
text box under the SAM loader, e.g. "robot" or "fox") or the mask growth (*Grow Mask With Blur*). Length: the
*Frames* Int (8k+1). Models are loaded one after the other, so a 5 s clip takes a few minutes. {SHARED}"""


def main(host):
    oi = json.load(_open(host.rstrip("/") + "/object_info"))
    wf = os.path.join(ROOT, "workflows")
    api = os.path.join(wf, "api")
    os.makedirs(api, exist_ok=True)
    for name, g, note in (("Viggle-Animate (video + reference image)", viggle(), NOTE_VIGGLE),
                          ("LTX Ripple (video + reference image)", ripple(), NOTE_RIPPLE),
                          ("Viggle-Animate + LTX Ripple (character + scene)", combined(), NOTE_COMBINED)):
        with open(os.path.join(wf, name + ".json"), "w") as f:
            json.dump(to_ui(g, oi, [(note, [980, 640])]), f, indent=1)
        slug = name.split(" (")[0].lower().replace(" + ", "_").replace(" ", "_").replace("-", "_")
        with open(os.path.join(api, slug + ".json"), "w") as f:
            json.dump(g, f, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8188")
