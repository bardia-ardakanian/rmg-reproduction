"""Tiny web server for the RMG demo: POST a prompt, get back generated joint positions; the browser
renders the motion in 3D (three.js). The model loads once at startup.

    python web/app.py            # serves on 0.0.0.0:8000  (tunnel: ssh -L 8000:localhost:8000 <host>)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in [os.path.join(HERE, "..", "src"), os.path.join(HERE, ".."),
          os.environ.get("MOTION_REAL", os.path.expanduser("~/rmg/motion_real"))]:
    if os.path.isdir(p):
        sys.path.insert(0, p)

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory

import s3
import qwen_text
from eval_hml import positions
from rmg_model import RMGTransformer
from rmg_flow import RMGFlow

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CKPT = os.environ.get("RMG_CKPT", os.path.join(HERE, "..", "runs", "rmg_base", "model.pth"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"loading {CKPT} on {DEV} ...")
c = torch.load(CKPT, map_location=DEV)
model = RMGTransformer(**c["config"]).to(DEV)
model.load_state_dict(c.get("ema_state_dict", c["state_dict"])); model.eval()
flow = RMGFlow(c.get("sigma_trans", 1.0), c.get("sigma_rot", 1.0))
print("ready.")


def smooth(J, win):
    if not win or win <= 1:
        return J
    sig = win / 3.0
    t = np.arange(win) - win // 2
    k = np.exp(-0.5 * (t / sig) ** 2); k /= k.sum()
    sh = J.shape; jf = J.reshape(sh[0], -1)
    jp = np.pad(jf, ((win // 2, win // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(jp[:, i], k, mode="valid") for i in range(jf.shape[1])], 1).reshape(sh)


app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/generate", methods=["POST"])
@torch.no_grad()
def generate():
    d = request.get_json(force=True)
    prompt = (d.get("prompt") or "a person walks forward").strip()
    guidance = float(d.get("guidance", 6.5))
    L = max(40, min(196, int(d.get("length", 120))))
    win = int(d.get("smooth", 9))
    seed = int(d.get("seed", 0))
    torch.manual_seed(seed)
    cond = qwen_text.encode([prompt], device=DEV).to(DEV)
    tr, q = flow.sample(model, 1, L, text=cond, guidance=guidance, n_steps=100, device=DEV)
    P = positions(tr[0].cpu(), s3.quat_to_matrix(q[0].cpu())).numpy()        # (L,22,3) HumanML3D FK
    P = smooth(P, win).astype(np.float32)
    P[:, :, 1] -= P[:, :, 1].min()                                          # feet on ground
    P[:, :, [0, 2]] -= P[0:1, 0:1, [0, 2]]                                  # start at origin
    return jsonify(joints=P.tolist(), parents=PARENTS, fps=20, prompt=prompt)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
