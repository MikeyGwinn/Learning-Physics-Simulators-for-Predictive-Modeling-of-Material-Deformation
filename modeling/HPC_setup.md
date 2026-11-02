# HPC Setup Guide: gs-particle-dynamics

Step-by-step setup for running the `gs-particle-dynamics` repo on a GPU node in the OSU HPC cluster. It uses the README from the repo with a modified setup for training on the HPC.

**Paper and repo:** *Learning a Particle Dynamics Model with Real-world Videos* (Kim, Sumukh, Fuxin, CVPR 2026 Findings), [github.com/chkim403/gs-particle-dynamics](https://github.com/chkim403/gs-particle-dynamics).

**Versions used:** Python 3.9, CUDA 12.4, PyTorch 2.5.0, PyTorch3D 0.7.8, gsplat 1.4.0. The repo lists the authors' full environment for reference but build the env with the steps below so the CUDA packages are compiled for the cluster's GPUs.

## 1. Connect to the HPC cluster

**If you have not already done so**, you will need to login to TEACH with your ONID and setup a HPC account so you have access to the server. One you have setup your account you can return to these instructions.

Replace `<ONID>` with your ONID username. You'll be asked for your ONID password. If the connection times out from off campus, connect to the OSU VPN first. Either way, you land on one of the login nodes (`submit-a`, `submit-b`, or `submit-c`).

**Option A: VS Code (Remote - SSH)**

1. Install the **Remote - SSH** extension (Microsoft) in VS Code.
2. Open the Command Palette (`Ctrl+Shift+P`, or `Cmd+Shift+P` on Mac) and run **Remote-SSH: Connect to Host...**
3. Enter `ssh <ONID>@submit.hpc.engr.oregonstate.edu`, and pick **Linux** if asked for the platform.
4. Enter your password. The bottom-left corner shows `SSH: submit.hpc.engr.oregonstate.edu` once connected.
5. Open a terminal with `` Ctrl+`  `` (all commands below run there). After cloning the repo in step 2, use **File > Open Folder** to open `~/path/to/project/gs-particle-dynamics`.

The VS Code server runs on the login node and uses a lot of its thread limit, so keep only one VS Code window connected (see step 6).

**Option B: A regular terminal**

From Terminal (Mac/Linux) or PowerShell (Windows):

```bash
ssh <ONID>@submit.hpc.engr.oregonstate.edu
```

## 2. Clone the repo

On the login node:

```bash
mkdir -p ~/path/to/project/ && cd ~/path/to/project/
git clone https://github.com/chkim403/gs-particle-dynamics.git gs-particle-dynamics
```

If `git` fails with `pthread_create error Resource temporarily unavailable`, you've hit the login node's thread limit. See step 6.

## 3. Initialize Conda

Conda is installed system-wide, but `conda activate` fails until your shell is initialized:

```bash
conda init bash
source ~/.bashrc  # conda init only takes effect in a new or reloaded shell
```

If you get `conda: command not found` instead of the "run conda init" error, conda isn't on your PATH yet. Find and load its module first, then run the two commands above:

```bash
module avail conda
module load <name from the list> # use (D) default module unless you have a reason otherwise
```

## 4. Load Slurm automatically in \~/.bashrc

Slurm commands (`srun`, `sinfo`, `sbatch`, `squeue`) come from a module and aren't on PATH by default. Add this to `~/.bashrc`, **above** the `# >>> conda initialize >>>` block:

```bash
# Slurm (scheduler commands)
if type module >/dev/null 2>&1; then
    module load slurm 2>/dev/null
fi
```

Then `source ~/.bashrc` and check with `which srun sinfo`. Make sure no line in `~/.bashrc` sets `PATH=...` without `:$PATH`, since that wipes Slurm and VS Code paths.

## 5. Move large files and caches off home

The home quota fills fast (conda env \~15 GB, caches \~3 GB). Use your share directory, `/nfs/hpc/share/``$``U``SE``R`, for everything large:

```bash
mkdir -p /nfs/hpc/share/$USER/.cache/pip /nfs/hpc/share/$USER/conda_pkgs
conda config --add pkgs_dirs /nfs/hpc/share/$USER/conda_pkgs

cat >> ~/.bashrc <<'EOF'
export PIP_CACHE_DIR=/nfs/hpc/share/$USER/.cache/pip
export HF_HOME=/nfs/hpc/share/$USER/.cache/huggingface
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
export TORCH_EXTENSIONS_DIR=/nfs/hpc/share/$USER/.cache/torch_extensions
EOF
source ~/.bashrc

rm -rf ~/.cache/pip   # reclaim space from the old pip cache
```

Check usage with `du -sh ~/.conda ~/.cache ~/path/to/project`. The share is probably not backed up, so keep code in git and irreplaceable files (checkpoints, dataset links) somewhere else too.

## 6. Do heavy work on a compute node, not the login node

The login node caps each user at 400 threads/processes, and VS Code Remote uses a lot of that. Hitting the cap causes `pthread_create error Resource temporarily unavailable` from `git`, `sinfo`, and builds. For the dataset download, installs, and builds, request a CPU node:

```bash
srun -p share -c 4 --mem=16G -t 01:30:00 --pty bash
```

If you hit the thread cap, close extra VS Code windows, or kill stale processes with targeted `pkill` commands (for example `pkill -u $USER git`). Don't use an unfiltered `pkill -u $USER`.

## 7. Download the dataset to the share

**Download the processed dataset.** It's hosted on Hugging Face at [chkim403/gs\_physics](https://huggingface.co/datasets/chkim403/gs_physics). Review the dataset license and terms of use (`docs/dataset_license.md` and `docs/dataset_terms_of_use.md` in the repo) first. Training and testing only need `processed_dataset.zip`; `dataset.zip` holds the raw recordings and isn't required.

From a compute node (step 6), download and extract it straight into the share:

```bash
mkdir -p /nfs/hpc/share/$USER/gs-particle-dynamics
cd /nfs/hpc/share/$USER/gs-particle-dynamics
wget https://huggingface.co/datasets/chkim403/gs_physics/resolve/main/processed_dataset.zip
unzip processed_dataset.zip -d processed_data/
```

If `wget` is blocked, finish step 8 first, then from the same folder inside the `cvpr26` env run `pip install huggingface_hub` and `huggingface-cli download chkim403/gs_physics processed_dataset.zip --repo-type dataset --local-dir .` instead. Compare the extracted layout with `docs/processed_dataset.md` in the repo: `data_dir` in step 11 points at the folder that directly holds the dataset contents. The code reads the data straight from the share, so nothing needs to be copied or linked into the repo.

## 8. Create the conda environment

On the compute node:

```bash
cd ~/path/to/project/gs-particle-dynamics
conda create -n cvpr26 python=3.9
conda activate cvpr26
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install iopath scikit-image matplotlib imageio plotly opencv-python fvcore ninja
```

This matches the README. PyTorch itself is prebuilt and never needs rebuilding.

## 9. Add the `cvpr` helper function

CUDA and gcc are loaded per session, not in every shell. Add this function to `~/.bashrc`:

```bash
cvpr() {
    module load gcc/12.5
    module load cuda/12.4
    conda activate cvpr26
    cd ~/path/to/project/gs-particle-dynamics
    export PYTHONPATH=$PWD:$PYTHONPATH
    echo "cvpr26 ready: $(python --version), gcc $(gcc -dumpversion), CUDA_HOME=$CUDA_HOME"
}
```

Run `cvpr` at the start of every session, every compute node shell, and every new tmux window.

## 10. Build PyTorch3D and gsplat for the right GPU

These two packages compile their own CUDA code, so they must target the GPU you'll run on. Two changes from the README:

- **Compiler:** the default gcc (8.5) is too old for PyTorch 2.5 extensions (GCC 9+ needed). `gcc/12.5` works with CUDA 12.4. The `cvpr` function loads it.
- **Architecture:** builds originally targeted only 8.0 (A100) and failed with "no kernel image available" on other GPUs. Build for both A100 (8.0) and A40 (8.6).

```bash
cvpr
nvcc --version                                                  # should be 12.4
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 2.5.0 12.4

export FORCE_CUDA=1 MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
pip install --no-build-isolation --force-reinstall --no-deps "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
pip install --no-build-isolation --force-reinstall --no-deps gsplat==1.4.0
```

`FORCE_CUDA=1` lets the build compile CUDA kernels on a node without a GPU. If you ever need a different GPU type, add its compute capability to `TORCH_CUDA_ARCH_LIST` and rebuild both.

The README installs gsplat from GitHub (`git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0`). `gsplat==1.4.0` from PyPI is the same release; either works with the flags above.

### 10b. Install the remaining project dependencies

Still on the compute node, from the repo root:

```bash
cvpr
pip install -r requirements.txt
python -c "import torch, pytorch3d, gsplat; print(torch.__version__, pytorch3d.__version__, gsplat.__version__)"
```

Expected: `2.5.0 0.7.8 1.4.0`. If `pip` tries to replace torch, pytorch3d, or gsplat, stop and rerun step 10 afterward, since a reinstall from PyPI would drop the custom CUDA builds.

## 11. Edit `configs/config.yaml`

- `data_dir`: the processed dataset folder on the share from step 7, for example `/nfs/hpc/share/<ONID>/gs-particle-dynamics/processed_data`.
- `exp_dir`: where checkpoints are saved (`/nfs/hpc/share/<ONID>/gs-particle-dynamics/exp`).&#32;
- `exp_name`: a name for the run, used as the checkpoint subfolder.&#32;
- `output_dir`: where `tools/test.py` writes rollouts (`/nfs/hpc/share/<ONID>/gs-particle-dynamics/rollouts`). Type your actual ONID in these paths, since the config file doesn't expand `$USER`.
- `train.num_workers`: 6 to **2**. With 6 workers on a 4-CPU, 32 GB job, a data loader worker was killed for running out of RAM.

  ```bash
  sed -i 's/^  num_workers: 6/  num_workers: 2/' configs/config.yaml
  ```
- `train.epoch`: set to 2 for a quick smoke test, then back to 50 for real runs.

## 12. Get a GPU node and sanity-check it

Use the `ampere` partition (A40, 46 GB, compute 8.6). `preempt` can hand out old GPUs like a GTX 1080 Ti (compute 6.1), which these builds don't support, and jobs there can be kicked off.

```bash
srun -p ampere --gres=gpu:1 -c 4 --mem=32G -t 02:00:00 --pty bash
cvpr
nvidia-smi

python -c "
import torch, gsplat
from pytorch3d.ops import knn_points
a = torch.rand(1,100,3,device='cuda'); b = torch.rand(1,100,3,device='cuda')
print(torch.cuda.get_device_name(0), knn_points(a,b,K=1).dists.shape)
"
```

Expected output: `NVIDIA A40 torch.Size([1, 100, 1])`. To give data loading more room, raise `-c` and `--mem` together (for example `-c 8 --mem=64G`) and increase `num_workers` to match.

## 13. Run training inside tmux

Start tmux **on the compute node**, so a VS Code disconnect doesn't kill the run:

```bash
tmux new -s run
cvpr      # tmux opens a fresh shell, so re-run this

python tools/train.py --config_path configs/config.yaml --scenario bowling
```

- `Ctrl+b` then `d` detaches; `tmux attach -t run` reattaches.
- `Ctrl+b` then `c` opens a second window for monitoring (`nvidia-smi`, `top -u $USER`, `free -h`).
- tmux and the run end when the `srun` time limit (`-t`) runs out. For long runs, use `sbatch` with `source ~/.bashrc` and `cvpr` in the job script.
- "scene\_XXXXX doesn't have enough frames to look ahead, skipping" messages are harmless.

Scenarios are `bowling` and `cube_stacks`. To fix a seed, add `--seed 12345`. The final checkpoint is saved to `<exp_dir>/<exp_name>[_seed_<seed>]_<scenario>/epoch_<epoch>/model.pt`.

## 14. Test and generate rollouts

Run `tools/test.py` with the same config, scenario, and seed as training. `--epoch` picks the checkpoint, so it must match an epoch that was saved (2 for the smoke test, 50 for a full run):

```bash
cvpr
python tools/test.py --config_path configs/config.yaml --scenario bowling --epoch 2
# seeded run: add --seed 12345 (same seed as training)
```

Rollouts are written to `<output_dir>/<scenario>/<exp_name>[_seed_<seed>]_epoch<epoch>_<scenario>/`.

## 15. Full runs with sbatch and run\_experiment.sh

The repo's `scripts/run_experiment.sh` trains and then tests for each configured seed. It activates conda itself, so edit the settings at the top before using it:

- Set `env_name="cvpr26"`.
- Replace its conda setup with `source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh`, and its CUDA module lines with `module load gcc/12.5 cuda/12.4`.
- Set `train.epoch` back to 50 in `configs/config.yaml`, and the test epoch in the script to match.

Then submit it as a batch job so it survives disconnects and isn't limited by an interactive session. Save this as `train_job.sh` in the repo root:

```bash
#!/bin/bash
#SBATCH -J gs-train
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH -o /nfs/hpc/share/%u/gs-particle-dynamics/logs/%x-%j.out

module load slurm gcc/12.5 cuda/12.4
source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate cvpr26
cd ~/path/to/project/gs-particle-dynamics
export PYTHONPATH=$PWD
nvidia-smi
bash scripts/run_experiment.sh
```

```bash
mkdir -p /nfs/hpc/share/$USER/gs-particle-dynamics/logs
sbatch train_job.sh
squeue -u $USER            # job status
tail -f /nfs/hpc/share/$USER/gs-particle-dynamics/logs/gs-train-<jobid>.out
```

Check the partition's time limit with `sinfo -p ampere` and adjust `-t`. To run a single training command instead of the script, replace the last line with the `python tools/train.py ...` command from step 13.

## Running again after setup

Once steps 1–11 are done, everything persists between logins: the env, the `cvpr` function, the dataset on the share, and your config edits. For each new training run and rollout, only these steps are needed.

**1. Connect and get a GPU node**

```bash
ssh <ONID>@submit.hpc.engr.oregonstate.edu      # or connect with VS Code (step 1)
srun -p ampere --gres=gpu:1 -c 4 --mem=32G -t 02:00:00 --pty bash
```

Raise `-t` (and `-c`/`--mem` with `num_workers`) for longer or faster runs.

**2. Start tmux and load the environment**

```bash
tmux new -s run
cvpr
nvidia-smi            # confirm you got an A40
```

**3. Pick the run settings in `configs/config.yaml`**

- `exp_name`: use a new name for each run, or the new checkpoints overwrite the last run's.
- `train.epoch`: 2 for a quick check, 50 for a full run.
- Leave `data_dir`, `exp_dir`, and `output_dir` as set in step 11.

**4. Train**

```bash
python tools/train.py --config_path configs/config.yaml --scenario bowling
# other scenario: --scenario cube_stacks; fixed seed: add --seed 12345
```

The checkpoint lands in `<exp_dir>/<exp_name>[_seed_<seed>]_<scenario>/epoch_<epoch>/model.pt`.

**5. Generate rollouts**

Use the same scenario and seed as training, and set `--epoch` to the `train.epoch` you trained for:

```bash
python tools/test.py --config_path configs/config.yaml --scenario bowling --epoch 50
```

Rollouts land in `<output_dir>/<scenario>/<exp_name>[_seed_<seed>]_epoch<epoch>_<scenario>/`.

**6. Leave it running or clean up**

- Detach with `Ctrl+b` then `d`, and come back later with `tmux attach -t run` (from the same compute node, so note its name, like `cn-s-2`, and `ssh` to it from the login node if needed).
- Free the GPU when you're done: `exit` tmux, then `exit` the `srun` shell.

For a full 50-epoch run, skip steps 1–2 and submit the batch job from step 15 instead (`sbatch train_job.sh` from the login node). It trains and then generates rollouts on its own.

**Pulling repo updates:** your `configs/config.yaml` edits are local changes, so use `git stash`, `git pull`, `git stash pop` to keep them.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `CondaError: Run 'conda init' before 'conda activate'` | Run `source ~/.bashrc` after `conda init` (step 3). |
| `srun: command not found` | Slurm isn't loaded. Run `module load slurm` and check step 4. |
| `pthread_create error Resource temporarily unavailable` | Login node thread cap. Move to a compute node, close extra VS Code windows (step 6). |
| "no kernel image available" | GPU architecture not in the build. Check `nvidia-smi` and rebuild with the right `TORCH_CUDA_ARCH_LIST` (step 10). |
| `DataLoader worker ... killed by signal: Killed` | Out of CPU RAM. Lower `num_workers` or `batch_sz`, or request more `--mem` (step 11). |
| Prompt shows `(base)` inside tmux | New shell. Run `cvpr` again. |
| Disk quota errors in home | Move data and caches to the share (step 5). |
| Training seems stalled | Check GPU use with `nvidia-smi` and worker CPU with `top`. Near-0% GPU with busy workers means data loading is the bottleneck: add workers or copy data to node-local `/tmp`. |
