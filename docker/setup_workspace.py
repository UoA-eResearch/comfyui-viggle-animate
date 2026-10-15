"""Prepare the /workspace volume: example inputs, workflows, extra nodes, and the model files of models.json.

  python setup_workspace.py prepare   # quick, before ComfyUI starts: examples, workflows, node files, the plan
  python setup_workspace.py models    # download the model files that are missing (resumes after interruption)

Environment:
  VIGGLE_VARIANT  auto (default: by GPU memory) | bf16 | int8 | pruned-int8
  LTX             auto (default: on when HF_TOKEN is set) | 1 | 0
  HF_TOKEN        Hugging Face token; needed for LTX-2.5, whose licence must first be accepted on its model page
"""
import json
import os
import shutil
import subprocess
import sys
import time

W = os.environ.get("WORKSPACE", "/workspace")
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = json.load(open(os.path.join(HERE, "models.json")))["groups"]
STATE = os.path.join(W, ".setup-state.json")
TOKEN = os.environ.get("HF_TOKEN") or None
BF16_WORKFLOW_NAME = MANIFEST["viggle"]["variants"]["bf16"]["file"]


def log(*a):
    print("[setup]", *a, flush=True)


def vram_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 2**30
    except Exception as e:
        log("could not query the GPU:", e)
    return 0.0


def viggle_variant():
    want = os.environ.get("VIGGLE_VARIANT", "auto").strip().lower()
    variants = MANIFEST["viggle"]["variants"]
    if want != "auto":
        if want not in variants:
            raise SystemExit(f"VIGGLE_VARIANT must be auto or one of {', '.join(variants)}, not {want!r}")
        return want, None
    gb = vram_gb()
    for name, v in variants.items():            # largest first
        if gb >= v["min_vram_gb"]:
            return name, gb
    return "pruned-int8", gb


def ltx_wanted():
    v = os.environ.get("LTX", "auto").strip().lower()
    return bool(TOKEN) if v == "auto" else v in ("1", "true", "yes", "on")


def ltx_access():
    """True if HF_TOKEN may download LTX-2.5 (its licence accepted on the model page)."""
    from huggingface_hub import auth_check
    repo = MANIFEST["ltx"]["gated_repo"]
    try:
        auth_check(repo, token=TOKEN)
        return True
    except Exception as e:
        log(f"LTX-2.5 is gated and this token cannot download it ({type(e).__name__}).")
        log(f"  Accept the licence at https://huggingface.co/{repo} with the Hugging Face account of HF_TOKEN,")
        log("  then restart the container. Viggle-Animate works without it.")
        return False


def plan():
    variant, gb = viggle_variant()
    groups = ["viggle", "qwen"]
    if ltx_wanted():
        if ltx_access():
            groups += ["ltx", "sam"]              # SAM 3.1 is used by the combined workflow only
    else:
        log("LTX Ripple is off (set HF_TOKEN, after accepting the LTX-2.5 licence, to add it).")
    state = {"viggle_variant": variant, "groups": groups}
    json.dump(state, open(STATE, "w"))
    how = f"{gb:.0f} GB GPU" if gb is not None else "VIGGLE_VARIANT"
    log(f"Viggle-Animate variant: {variant} (from {how}); model groups: {', '.join(groups)}")
    return state


def files(state):
    for g in state["groups"]:
        for f in MANIFEST[g]["files"]:
            f = dict(f)
            if "{variant_file}" in f["path"]:
                f["path"] = f["path"].format(variant_file=MANIFEST[g]["variants"][state["viggle_variant"]]["file"])
            yield g, f


def remote_size(f):
    from huggingface_hub import HfApi
    try:
        info = HfApi().get_paths_info(f["repo"], [f["path"]], revision=f["revision"], token=TOKEN)
        return info[0].size if info else None
    except Exception:
        return None


def fetch(f):
    """Download one file to /workspace/<dest>/<name>, skipping it when already complete."""
    from huggingface_hub import hf_hub_download
    dest_dir = os.path.join(W, f["dest"])
    dest = os.path.join(dest_dir, os.path.basename(f["path"]))
    size = remote_size(f)
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        return dest, False
    os.makedirs(dest_dir, exist_ok=True)
    tmp = os.path.join(W, ".downloads", f["repo"].replace("/", "_"))
    t0 = time.time()
    got = hf_hub_download(f["repo"], f["path"], revision=f["revision"], local_dir=tmp, token=TOKEN)
    os.replace(got, dest)
    gb = os.path.getsize(dest) / 1e9
    log(f"got {os.path.basename(dest)} ({gb:.1f} GB in {time.time() - t0:.0f} s)")
    return dest, True


def node_files(state):
    """Small node files (the turbo LoRA's two Qwen-Image nodes) and the LTX node pack, before ComfyUI starts."""
    for g, f in files(state):
        if f["dest"].startswith("custom_nodes/"):
            dest, _ = fetch(f)
            init = os.path.join(os.path.dirname(dest), "__init__.py")
            if not os.path.exists(init):        # ComfyUI imports a node folder as a package
                mod = os.path.splitext(os.path.basename(dest))[0]
                with open(init, "w") as fh:
                    fh.write(f"from . import {mod} as _m\nNODE_CLASS_MAPPINGS = _m.NODE_CLASS_MAPPINGS\n"
                             "NODE_DISPLAY_NAME_MAPPINGS = getattr(_m, 'NODE_DISPLAY_NAME_MAPPINGS', {})\n")
    if "ltx" in state["groups"]:
        n = MANIFEST["ltx"]["nodes"]
        d = os.path.join(W, "custom_nodes", n["dir"])
        if not os.path.isdir(os.path.join(d, ".git")):
            subprocess.run(["git", "clone", "-q", n["repo"], d], check=True)
        subprocess.run(["git", "-C", d, "fetch", "-q", "origin", n["ref"]], check=False)
        subprocess.run(["git", "-C", d, "checkout", "-q", n["ref"]], check=True)
        marker = f"/opt/viggle/.ltx-deps-{n['ref'][:12]}"
        if not os.path.exists(marker):             # once per container (the venv is in the image layer)
            log("installing the LTX node pack's Python requirements")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--cache-dir", os.path.join(W, ".cache", "pip"),
                            "-r", os.path.join(d, "requirements.txt")], check=True)
            open(marker, "w").close()


def examples():
    src = "/opt/viggle/examples"
    dst = os.path.join(W, "input")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for sub in ("drivers", "characters"):
        for name in sorted(os.listdir(os.path.join(src, sub))):
            if not os.path.exists(os.path.join(dst, name)):
                shutil.copy(os.path.join(src, sub, name), os.path.join(dst, name))
                n += 1
    if n:
        log(f"{n} example files copied to input/")


def workflows(state):
    """The bundled workflows, with the chosen Viggle file name; refreshed on every start (save your own changes
    under another name)."""
    src = "/opt/viggle/workflows"
    dst = os.path.join(W, "user", "default", "workflows")
    os.makedirs(dst, exist_ok=True)
    viggle_file = MANIFEST["viggle"]["variants"][state["viggle_variant"]]["file"]
    for name in sorted(os.listdir(src)):
        if not name.endswith(".json") or ("LTX" in name and "ltx" not in state["groups"]):
            continue
        text = open(os.path.join(src, name)).read().replace(BF16_WORKFLOW_NAME, viggle_file)
        with open(os.path.join(dst, name), "w") as fh:
            fh.write(text)


def check_space(state):
    need = 0
    for g, f in files(state):
        dest = os.path.join(W, f["dest"], os.path.basename(f["path"]))
        size = remote_size(f) or 0
        if not os.path.exists(dest) or os.path.getsize(dest) != size:
            need += size
    free = shutil.disk_usage(W).free
    log(f"to download: {need / 1e9:.0f} GB; free on /workspace: {free / 1e9:.0f} GB")
    if need > free:
        log("NOT ENOUGH DISK SPACE on the /workspace volume: downloads will fail. Give it more room.")


def models(state):
    t0 = time.time()
    check_space(state)
    failed = []
    for g, f in files(state):
        try:
            fetch(f)
        except Exception as e:
            failed.append(os.path.basename(f["path"]))
            log(f"FAILED {f['repo']}/{f['path']}: {str(e).splitlines()[0][:300]}")
    shutil.rmtree(os.path.join(W, ".downloads"), ignore_errors=True)
    if failed:
        log(f"{len(failed)} file(s) failed: {', '.join(failed)}. Restart the container to retry.")
    else:
        log(f"all models ready ({time.time() - t0:.0f} s). Open the Workflows tab and run an example.")
        open(os.path.join(W, "models", ".ready"), "w").close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if cmd == "prepare":
        st = plan()
        examples()
        workflows(st)
        node_files(st)
    elif cmd == "models":
        models(json.load(open(STATE)))
    else:
        raise SystemExit(__doc__)
