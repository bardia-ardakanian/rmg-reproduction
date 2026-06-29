# data/

Put the **HumanML3D** dataset here (or point `HML_DIR` at it elsewhere). Expected contents:
```
HumanML3D/
  new_joint_vecs/   # <id>.npy  (T, 263) motion features
  new_joints/       # <id>.npy  (T, 22, 3) joint positions
  texts/            # <id>.txt  captions
  train.txt val.txt test.txt
```
The dataset needs **AMASS** (account) + **SMPL** and the official processing pipeline — follow
`external/HumanML3D/README.md` (cloned by `scripts/setup.sh`).

`scripts/setup.sh` also downloads the Guo evaluator models + GloVe into `data/t2m_eval/`.

Environment variables read by the code (defaults shown):
```
HML_DIR     = /path/to/HumanML3D
T2M_EVAL    = ./data/t2m_eval
T2M_REPO    = ./external/text-to-motion
HML3D_REPO  = ./external/HumanML3D
```
