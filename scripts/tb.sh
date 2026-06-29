#!/bin/bash
# TensorBoard for training + eval curves. View locally via:  ssh -L 6006:localhost:6006 <host>
tensorboard --logdir runs --port 6006 --host 0.0.0.0
