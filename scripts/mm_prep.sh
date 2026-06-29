#!/bin/bash
# Precompute the MotionMillion clip index + Qwen caption embeddings (run once before training).
# Streams texts.tar.gz, encodes captions with Qwen3-Embedding-0.6B, writes cache/mm_<split>_{index.npz,emb.npy}.
# Run from the repo root. Set HF_HOME to a persistent dir so the encoder is cached.
set -e
export HF_HOME=${HF_HOME:-$HF_HOME}
mkdir -p cache
python src/mm_prep.py --split train --caps 4 --out cache/mm     # ~458k clips, a few captions each
python src/mm_prep.py --split val   --caps 1 --out cache/mm
