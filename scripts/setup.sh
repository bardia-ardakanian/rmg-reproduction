#!/bin/bash
# Downloads everything needed to build the project (except the AMASS-gated HumanML3D dataset).
# Run from the repo root:  bash scripts/setup.sh
set -e
pip install -r requirements.txt
mkdir -p data/t2m_eval external

# 1) Official Guo et al. T2M evaluators (FID / R-precision / MM-Dist / Diversity) + GloVe
pushd data/t2m_eval >/dev/null
gdown 1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb -O t2m.zip      # t2m evaluator bundle (finest.tar, Comp_v6, meta, ...)
gdown 1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n -O glove.zip    # glove embeddings
unzip -q t2m.zip && unzip -q glove.zip && rm -f t2m.zip glove.zip
popd >/dev/null

# 2) text-to-motion repo (evaluator networks + metric code used by src/eval_official.py)
[ -d external/text-to-motion ] || git clone https://github.com/EricGuo5513/text-to-motion external/text-to-motion

# 3) HumanML3D repo (skeleton/paramUtil + dataset-processing pipeline)
[ -d external/HumanML3D ] || git clone https://github.com/EricGuo5513/HumanML3D external/HumanML3D

# 4) Qwen3-Embedding-0.6B text encoder (auto-cached by transformers)
python scripts/fetch_qwen.py

cat <<NOTE

Setup done. Export these:
  export T2M_EVAL=$(pwd)/data/t2m_eval
  export T2M_REPO=$(pwd)/external/text-to-motion
  export HML3D_REPO=$(pwd)/external/HumanML3D
  export HML_DIR=/path/to/HumanML3D     # processed dataset (see data/README.md)
NOTE
