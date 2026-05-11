## Environment Setup

The Python virtual environment for this project is a conda environment named (hoi_common), which can be activated by 

```shell
conda activate hoi_common
```

Run python scripts using this environments to avoid unexpected dependency issues.

## Training scripts

Currently, the script used for training is `scripts/train_diffusion.py`. The rest scripts are for small test or training that are already done, so no need to look into.

## Code ignores

* Code under `prev_sota` folder are the code copied from previous methods, and do not need review.
* Files under `tmp` are temporarily stored here. If not specified, do not review.