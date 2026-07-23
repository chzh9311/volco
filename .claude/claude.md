## Overview

This project implements a contact representation based on volumes

## environment

The code need to be run in conda environment `hoi_common`, with CUDA 12.4.

## Training procedure

The training of the module takes 3 steps:
1. Train GridAE. This is triggered by `scripts/train_gridae.py`.
2. Train HandVAE. This is triggered by `scripts/train_handvae.py`.
3. Train VolcoDiff. triggered by `scripts/train_diffusion.py`.