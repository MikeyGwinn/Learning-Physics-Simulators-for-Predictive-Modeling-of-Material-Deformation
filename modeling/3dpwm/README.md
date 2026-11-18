# [3D Point World Models: Point Completion Enables More Accurate Dynamics Learning](https://pvskand.github.io/3dpwm/)

[![Code](https://img.shields.io/badge/Code-pvskand%2F3dpwm-blue)](https://github.com/pvskand/3dpwm)
[![Project](https://img.shields.io/badge/Project-Website-brightgreen)](https://pvskand.github.io/3dpwm/)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/pdf/2607.00148)
[![arXiv](https://img.shields.io/badge/arXiv-2607.00148-b31b1b)](https://arxiv.org/abs/2607.00148)
[![Branch](https://img.shields.io/badge/Branch-main-blue)](https://github.com/pvskand/3dpwm/tree/main)


<a href="https://pvskand.github.io/">Skand Peri</a>,
<a href="https://www.linkedin.com/in/hungnguyenalexx">Hung Nguyen</a>,
<a href="https://chkim403.github.io/">Chanho Kim</a>,
<a href="https://web.engr.oregonstate.edu/~lif/">Li Fuxin</a>,
<a href="https://web.engr.oregonstate.edu/~leestef/">Stefan Lee</a>

<div align="center">
  <img src="https://pvskand.github.io/3dpwm/img/teaser.png" alt="3DPWM overview" width="100%">
</div>

## Overview

3DPWM is a task-agnostic world model that operates entirely in 3D space by first completing partial point clouds and then learning action-conditioned dynamics in the completed scene. Operating on completed geometry enables reliable long-horizon rollouts and more accurate cost evaluation for model-based planning, while supporting adaptation to new tasks.

See the [project page](https://pvskand.github.io/3dpwm/) for videos, 3D rollouts, and the [appendix](https://pvskand.github.io/3dpwm/appendix.html).

## Installation

This code release uses Linux x86-64, Python 3.11, and an NVIDIA GPU.
The dependency versions below were with PyTorch 2.8 and CUDA 12.8.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**NOTE:** The requirements include prebuilt FlashAttention and torch-scatter wheels for
this Python/PyTorch combination. Other platforms require matching wheels.

## Data

The public dataset repository is
[pvskand/3dpwm-datasets](https://huggingface.co/datasets/pvskand/3dpwm-datasets).


After installing the requirements, run the following from the `3dpwm` repository
root.

```bash
source .venv/bin/activate

# 1. Download both splits and their checksums.
hf download pvskand/3dpwm-datasets --repo-type dataset \
  --revision 8137892aa0a234bdf9abc6583afa10ff4abc2290 \
  --include "stackcube/*" --local-dir data/hf_download

# 2. Verify the downloaded archives; both should report OK.
(cd data/hf_download/stackcube && sha256sum -c SHA256SUMS)

# 3. Extract into the directory used by configs/stackcube.yaml.
mkdir -p data/StackCube-v1-demos
tar -xzf data/hf_download/stackcube/train.tar.gz -C data/StackCube-v1-demos
tar -xzf data/hf_download/stackcube/test.tar.gz -C data/StackCube-v1-demos
```


## Training

```bash
WANDB_MODE=disabled OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python -u -m tools.train --config_path configs/stackcube.yaml --epochs 20
```

Resume a release checkpoint:

```bash
python -u -m tools.train --config_path configs/stackcube.yaml \
  --resume runs/stackcube/epoch_01/model.pt --epochs 2
```


## Test inference

```bash
python -m tools.test --config_path configs/stackcube.yaml \
  --checkpoint runs/stackcube/epoch_15/model.pt
```


## Licence

This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## BibTeX

```
@article{3dpwm,
  title={3D Point World Models: Point Completion Enables More Accurate Dynamics Learning},
  author={Peri, Skand and Nguyen, Hung and Kim, Chanho and Fuxin, Li and Lee, Stefan},
  journal={arXiv preprint arXiv:2607.00148},
  year={2026}
}
```
