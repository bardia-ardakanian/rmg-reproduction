"""
Body renderer (paper Figure 1 look), driven by generated joints, with onion-skin "ghosts".

The paper renders a SMPL body (Loper et al. 2015). We do the same: fit SMPL-X to the generated joints and
render it as a smooth gray clay body, 3/4 from the top-left, with past poses left behind as fading ghosts
so a walk spreads across the floor instead of walking out of frame. `--body capsule` falls back to a plain
capsule figure (no body model needed). Two modes:

  --mode montage : one still PNG, K evenly spaced poses fading oldest->newest, with the prompt caption.
  --mode gif     : animation where each frame trails a few fading ghosts of the recent poses.

Runs in an env with smplx + pyrender (EGL headless) + PIL. Input = report/mesh_joints.npz (P,L,22,3).

    python render_mannequin.py --joints report/mesh_joints.npz --mode montage --model_path $SMPLX_PATH --out report
    python render_mannequin.py --joints report/mesh_joints.npz --mode gif     --model_path $SMPLX_PATH --out report
"""
import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import trimesh
import pyrender
import imageio
from PIL import Image, ImageDraw, ImageFont

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SPINE = [(0, 3), (3, 6), (6, 9), (9, 12)]
HEAD_J = 15
CLAY = [0.70, 0.74, 0.82]
FLOOR = [0.88, 0.88, 0.91]
FOV = np.pi / 3.8


def look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = map(lambda v: np.asarray(v, np.float32), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = s; m[:3, 1] = u; m[:3, 2] = -f; m[:3, 3] = eye
    return m


def smooth_time(x, win=9):
    if not win or win <= 1:
        return x
    win = win + 1 if win % 2 == 0 else win
    sig = win / 3.0
    t = np.arange(win) - win // 2
    k = np.exp(-0.5 * (t / sig) ** 2); k /= k.sum()
    sh = x.shape; xf = x.reshape(sh[0], -1)
    xp = np.pad(xf, ((win // 2, win // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, c], k, mode="valid") for c in range(xf.shape[1])], 1).reshape(sh)


# --------------------------------------------------------------------------- capsule fallback mannequin
def _capsule(a, b, r):
    if np.linalg.norm(b - a) < 1e-5:
        return trimesh.creation.icosphere(radius=r, subdivisions=2).apply_translation(a)
    c = trimesh.creation.cylinder(radius=r, segment=[a, b], sections=16)
    caps = [trimesh.creation.icosphere(radius=r, subdivisions=2).apply_translation(p) for p in (a, b)]
    return trimesh.util.concatenate([c] + caps)


def capsule_mesh(P):
    rl, rb, rt, rh = 0.062, 0.072, 0.115, 0.135
    parts = []
    for j, p in enumerate(PARENTS):
        if p >= 0 and (j, p) not in SPINE and (p, j) not in SPINE:
            parts.append(_capsule(P[p], P[j], rl))
    for a, b in SPINE:
        parts.append(_capsule(P[a], P[b], rt))
    parts.append(_capsule(P[1], P[2], rt * 0.8)); parts.append(_capsule(P[16], P[17], rt * 0.7))
    for j in [0, 1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]:
        parts.append(trimesh.creation.icosphere(radius=rb, subdivisions=2).apply_translation(P[j]))
    head = trimesh.creation.icosphere(radius=rh, subdivisions=3)
    head.apply_scale([0.9, 1.2, 0.95]); head.apply_translation(P[HEAD_J] + np.array([0, 0.04, 0]))
    parts.append(head)
    return trimesh.util.concatenate(parts)


# --------------------------------------------------------------------------- rendering
def _mat(rgb, rough=0.72):
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=rgb + [1.0], metallicFactor=0.0, roughnessFactor=rough)


class Renderer:
    def __init__(self, res, azim, elev, dist, target_y=0.85, ss=2):
        self.r = pyrender.OffscreenRenderer(res * ss, res * ss)
        self.res, self.ss = res, ss
        az, el = np.radians(azim), np.radians(elev)
        self.target = np.array([0, target_y, 0])
        self.eye = self.target + np.array([dist * np.cos(el) * np.sin(az), dist * np.sin(el),
                                           dist * np.cos(el) * np.cos(az)])
        self.key = look_at([2.5, 4.5, 1.5], [0, 0, 0])
        self.fill = look_at([-3.0, 2.0, 1.5], [0, 0.8, 0])

    def _down(self, a):
        return np.array(Image.fromarray(a).resize((self.res, self.res), Image.LANCZOS))

    def _cam(self, sc):
        sc.add(pyrender.PerspectiveCamera(yfov=FOV), pose=look_at(self.eye, self.target))

    def ground(self):
        sc = pyrender.Scene(bg_color=[0.95, 0.95, 0.97, 1.0], ambient_light=[0.6, 0.6, 0.6])
        g = trimesh.creation.box(extents=[16, 0.02, 16]); g.apply_translation([0, -0.01, 0])
        sc.add(pyrender.Mesh.from_trimesh(g, material=_mat(FLOOR, rough=1.0)))
        self._cam(sc); sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=2.0), pose=self.key)
        return self._down(self.r.render(sc)[0])

    def body(self, mesh):
        sc = pyrender.Scene(bg_color=[0.95, 0.95, 0.97, 1.0], ambient_light=[0.40, 0.40, 0.44])
        sc.add(pyrender.Mesh.from_trimesh(mesh, material=_mat(CLAY), smooth=True))
        self._cam(sc)
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.4), pose=self.key)
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.4), pose=self.fill)
        col, dep = self.r.render(sc)
        return self._down(col), self._down((dep > 0).astype(np.float32) * 255)[..., None] / 255.0


def composite(bg, layers):
    acc = bg.astype(np.float32).copy()
    for rgb, mask, alpha in layers:
        a = mask * alpha
        acc = acc * (1 - a) + rgb.astype(np.float32) * a
    return acc.clip(0, 255).astype(np.uint8)


def _font(size):
    try:
        import matplotlib.font_manager as fm
        return ImageFont.truetype(fm.findfont(fm.FontProperties(weight="bold")), size)
    except Exception:
        return ImageFont.load_default()


def caption(img, text, pad=0.13):
    h, w = img.shape[:2]
    bar = int(h * pad)
    canvas = Image.new("RGB", (w, h + bar), (247, 247, 250))
    canvas.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(canvas)
    f = _font(int(bar * 0.44))
    t = '"' + text + '"'
    tw = d.textbbox((0, 0), t, font=f)[2]
    d.text(((w - tw) / 2, h + bar * 0.26), t, fill=(40, 42, 48), font=f)
    return np.array(canvas)


def fit_view(pts, margin=1.20):
    """pts (.., 3) already floor-dropped + xz-centered. Returns (dist, target_y)."""
    H = float(pts[..., 1].max())
    W = float(max(pts[..., 0].ptp(), pts[..., 2].ptp())) + 0.5
    vert = H + 0.45
    th = np.tan(FOV / 2)
    return max(2.5, (vert / 2) / th, (W / 2) / th) * margin, H * 0.46 + 0.1


# --------------------------------------------------------------------------- per-clip mesh providers
def smplx_fit_clip(J_clip, model_path, dev, iters):
    """Fit SMPL-X to the clip's joints once; return (verts (L,V,3), faces)."""
    import torch
    import smplx
    from fit_render_mesh import fit as smplx_fit
    L = J_clip.shape[0]
    bm = smplx.create(model_path, model_type="smplx", gender="neutral", ext="npz",
                      use_pca=False, flat_hand_mean=True, batch_size=L).to(dev)
    verts, faces, mpjpe = smplx_fit(bm, torch.tensor(J_clip, dtype=torch.float32, device=dev), dev, iters)
    print(f"   smpl-x fit MPJPE {mpjpe*1000:.1f}mm")
    return verts, faces


def render_one(pts, frame_mesh, prompt, mode, out, idx, res, azim, elev, dist_override,
               n_ghost, every, fps):
    L = pts.shape[0]
    floor = pts[..., 1].min()
    cx, cz = pts[..., 0].mean(), pts[..., 2].mean()
    shift = np.array([-cx, -floor, -cz])
    ptsc = pts + shift
    dist, ty = fit_view(ptsc)
    R = Renderer(res, azim, elev, dist_override or dist, target_y=ty)
    bg = R.ground()
    slug = "".join(c if c.isalnum() else "_" for c in prompt)[:28]

    def layer(f, alpha):
        m = frame_mesh(f); m.apply_translation(shift)
        rgb, mask = R.body(m)
        return (rgb, mask, alpha)

    if mode == "montage":
        sel = np.linspace(0, L - 1, n_ghost).astype(int)
        al = np.linspace(0, 1, len(sel)) ** 1.3
        al = 0.14 + (1 - 0.14) * al
        img = caption(composite(bg, [layer(f, a) for f, a in zip(sel, al)]), prompt)
        imageio.imwrite(os.path.join(out, f"mannequin_{idx}_{slug}.png"), img)
        print(f"wrote mannequin_{idx}_{slug}.png | {prompt}")
    else:
        frames = []
        for t in range(L):
            ks = [k for k in range(every * n_ghost, 0, -every) if t - k >= 0]
            ly = [layer(t - k, 0.10 + 0.30 * (1 - k / (every * n_ghost))) for k in ks]
            ly.append(layer(t, 1.0))
            frames.append(caption(composite(bg, ly), prompt))
        imageio.mimsave(os.path.join(out, f"mannequin_{idx}_{slug}.gif"), frames, fps=fps)
        print(f"wrote mannequin_{idx}_{slug}.gif | {prompt}")
    R.r.delete()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joints", default="report/mesh_joints.npz")
    ap.add_argument("--mode", choices=["montage", "gif"], default="montage")
    ap.add_argument("--body", choices=["smplx", "capsule"], default="smplx")
    ap.add_argument("--model_path", default=os.environ.get("SMPLX_PATH", "models/smplx"))
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--res", type=int, default=540)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--azim", type=float, default=-32.0)
    ap.add_argument("--elev", type=float, default=13.0)
    ap.add_argument("--dist", type=float, default=0.0)
    ap.add_argument("--ghosts", type=int, default=6, help="montage: # poses; gif: # trailing ghosts")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if a.body == "smplx" and __import__("torch").cuda.is_available() else "cpu"

    d = np.load(a.joints, allow_pickle=True)
    J, prompts = d["joints"], list(d["prompts"])
    J = np.stack([smooth_time(J[i], a.smooth) for i in range(J.shape[0])])
    rot = trimesh.transformations.rotation_matrix(np.radians(180.0), [0, 1, 0])[:3, :3]
    for i in range(J.shape[0]):
        if a.only >= 0 and i != a.only:
            continue
        if a.body == "smplx":
            verts, faces = smplx_fit_clip(J[i], a.model_path, dev, a.iters)
            pts = verts @ rot.T
            fm = lambda f, P=pts, F=faces: trimesh.Trimesh(P[f], F, process=False)
        else:
            pts = J[i] @ rot.T
            fm = lambda f, P=pts: capsule_mesh(P[f])
        render_one(pts, fm, prompts[i], a.mode, a.out, i, a.res, a.azim, a.elev, a.dist,
                   a.ghosts, a.every, a.fps)


if __name__ == "__main__":
    main()
