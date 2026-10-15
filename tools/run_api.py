"""Run a bundled workflow through the ComfyUI API: upload a driving video and a reference image, queue the job,
wait for it, download the results. Jobs from several people simply queue up on the server.

  python3 tools/run_api.py --video clip.mp4 --image character.png
  python3 tools/run_api.py --host http://gpu-box:8188 --video clip.mp4 --image character.png \
      --workflow ltx_ripple --extra "Keep the red scarf." --seed 7 --out results/

Standard library only (Python 3.8+).
"""
import argparse
import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
_open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open


def upload(host, path):
    """POST /upload/image (ComfyUI stores any file type in input/); returns the stored name."""
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        data = f.read()
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(host + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    r = json.load(_open(req))
    return (r.get("subfolder") + "/" if r.get("subfolder") else "") + r["name"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://localhost:8188")
    ap.add_argument("--video", required=True, help="driving video (its soundtrack drives the lip-sync)")
    ap.add_argument("--image", required=True, help="the character to put in")
    ap.add_argument("--workflow", default="viggle_animate", choices=["viggle_animate", "ltx_ripple", "viggle_animate_ltx_ripple"],
                    help="viggle_animate (default), ltx_ripple, or both combined (character + scene)")
    ap.add_argument("--extra", default=None, help="extra instructions for the first-frame edit (combined workflow: "
                                                  "the scene change; default: the workflow's own)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--viggle-model", default=None,
                    help="which Viggle-Animate build to use when the server has several: bf16, int8 or pruned")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    host = a.host.rstrip("/")

    graph = json.load(open(os.path.join(HERE, "..", "workflows", "api", a.workflow + ".json")))
    video, image = upload(host, a.video), upload(host, a.image)
    for node in graph.values():
        cls, inp = node["class_type"], node["inputs"]
        if cls == "VHS_LoadVideo":
            inp["video"] = video
        elif cls == "LoadImage":
            inp["image"] = image
        elif cls == "PrimitiveStringMultiline" and a.extra is not None:
            inp["value"] = a.extra
        elif a.seed is not None and cls in ("RandomNoise", "ViggleChunkedSamplerAudioPin"):
            inp["noise_seed" if cls == "RandomNoise" else "seed"] = a.seed
    # the installed Viggle variant may differ from the file name in the graph: use what the server has
    models = json.load(_open(host + "/models/diffusion_models"))
    viggle = [m for m in models if "ref2va_viggle" in m]
    if a.viggle_model:
        want = {"bf16": "_bf16", "int8": "_viggle_int8", "pruned": "pruned_int8"}.get(a.viggle_model, a.viggle_model)
        viggle = [m for m in viggle if want in m]
        if not viggle:
            raise SystemExit(f"no Viggle-Animate build matching {a.viggle_model!r} on the server")
    for node in graph.values():
        if node["class_type"] == "UNETLoader" and "ref2va_viggle" in node["inputs"]["unet_name"] and viggle:
            node["inputs"]["unet_name"] = viggle[0]

    req = urllib.request.Request(host + "/prompt", json.dumps({"prompt": graph, "client_id": uuid.uuid4().hex}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        pid = json.load(_open(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(e.read().decode()[:3000])
    print("queued", pid)
    t0, last = time.time(), 0
    while True:
        h = json.load(_open(f"{host}/history/{pid}"))
        if pid in h:
            break
        if time.time() - last > 30:
            q = json.load(_open(host + "/queue"))
            ahead = sum(1 for it in q["queue_pending"] if it[1] != pid)
            print(f"  {time.time() - t0:.0f} s; running {len(q['queue_running'])}, waiting {ahead}")
            last = time.time()
        time.sleep(3)
    st = h[pid]["status"]
    if st.get("status_str") == "error":
        raise SystemExit("failed: " + json.dumps(st.get("messages"))[-3000:])
    os.makedirs(a.out, exist_ok=True)
    for o in h[pid]["outputs"].values():
        for f in o.get("gifs", []) + o.get("images", []):
            q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
            dest = os.path.join(a.out, f["filename"])
            with open(dest, "wb") as fh:
                fh.write(_open(f"{host}/view?{q}").read())
            print("saved", dest)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
