"""
Prepare a MotionMillion split for RMG training: resolve the on-disk clips, grab a few captions each,
and precompute their Qwen3-Embedding vectors once (training then just indexes into the memmap).

Writes two files under --out:
  mm_<split>_index.npz   paths (abs .npy motion paths), capoff + caprows (ragged caption rows per clip)
  mm_<split>_emb.npy     (M, 1024) float16 caption embeddings (memmapped at train time)

    python mm_prep.py --split train --caps 4 --out cache/mm
    python mm_prep.py --split val   --caps 1 --out cache/mm
"""
import argparse
import os
import random
import sys
import tarfile

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mm_data
import qwen_text


def gather_captions(names, caps_per, rng):
    """Stream texts.tar.gz once and pull up to caps_per captions for each wanted clip name.
    Returns {name: [captions]}. Falls back to extracted texts/ dir if the tar is absent."""
    wanted = {f"texts/{n}.txt": n for n in names}
    out = {}
    tar_path = os.path.join(mm_data.MM_ROOT, "texts.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as t:
            for m in tqdm(t, desc="scan texts.tar"):
                n = wanted.get(m.name)
                if n is None or not m.isfile():
                    continue
                lines = [l.strip() for l in t.extractfile(m).read().decode("utf-8", "replace").splitlines() if l.strip()]
                if lines:
                    out[n] = lines if len(lines) <= caps_per else rng.sample(lines, caps_per)
    else:
        for n in tqdm(names, desc="read texts/"):
            p = mm_data.caption_path(n)
            if not os.path.exists(p):
                continue
            lines = [l.strip() for l in open(p) if l.strip()]
            if lines:
                out[n] = lines if len(lines) <= caps_per else rng.sample(lines, caps_per)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--caps", type=int, default=4, help="captions kept per clip (random subset)")
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--enc_batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cache/mm")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    print(f"resolving {a.split} clips on disk ...")
    clips = mm_data.resolvable_clips(a.split, max_clips=a.max_clips, shuffle_seed=a.seed)
    print(f"  {len(clips)} resolvable clips")

    name2path = {n: p for n, p in clips}
    caps_by_name = gather_captions(set(name2path), a.caps, rng)
    paths, per_clip = [], []
    for name, _ in clips:                       # keep clip order stable
        if name in caps_by_name:
            paths.append(name2path[name]); per_clip.append(caps_by_name[name])
    print(f"  {len(paths)} clips with captions")

    flat, capoff = [], [0]
    for caps in per_clip:
        flat.extend(caps); capoff.append(len(flat))
    M = len(flat)
    print(f"  {M} captions to encode")

    emb_path = a.out + f"_{a.split}_emb.npy"
    emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(M, qwen_text.DIM))
    for s in tqdm(range(0, M, a.enc_batch), desc="encode"):
        e = qwen_text.encode(flat[s:s + a.enc_batch], device=dev)
        emb[s:s + e.shape[0]] = e.numpy().astype(np.float16)
    emb.flush()

    index_path = a.out + f"_{a.split}_index.npz"
    np.savez(index_path,
             paths=np.array(paths, dtype=object),
             capoff=np.array(capoff, dtype=np.int64),
             caprows=np.arange(M, dtype=np.int64))
    print(f"wrote {index_path}  and  {emb_path}  (clips={len(paths)} caps={M})")


if __name__ == "__main__":
    main()
