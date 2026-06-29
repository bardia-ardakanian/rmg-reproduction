#!/bin/bash
# MotionMillion setup: point MM_ROOT at the dataset, extract the split lists, sanity-check.
# The dataset itself (rpr272 motions + texts.tar.gz + split.tar.gz + mean_std) is gated on HuggingFace:
#   https://huggingface.co/datasets/VankouF/MotionMillion   (request access, then download)
# On our box it already lives at $MM_ROOT.
set -e
export MM_ROOT=${MM_ROOT:-$MM_ROOT}
export MM_META=${MM_META:-$MM_META}

mkdir -p "$MM_META"
tar -xzf "$MM_ROOT/split.tar.gz" -C "$MM_META"        # split/version1/t2m_60_300/{train,val,test}.txt
echo "split lists -> $MM_META/split/version1/t2m_60_300/"
ls "$MM_META/split/version1/t2m_60_300/"

# captions stay packed in $MM_ROOT/texts.tar.gz; mm_prep.py streams it (no need to extract 1.5M files).
python - <<'PY'
import os, glob
root = os.environ["MM_ROOT"]
n = sum(len(os.listdir(d)) for d in glob.glob(os.path.join(root, "folder*")))
print(f"on-disk rpr272 clips: {n}")
print("mean/std:", os.path.exists(os.path.join(root, "mean_std", "Mean.npy")))
print("texts.tar.gz:", os.path.exists(os.path.join(root, "texts.tar.gz")))
PY
