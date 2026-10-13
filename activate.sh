#!/usr/bin/env bash
# Source this file to load the CUDA toolkit and activate the PhaseFieldX env.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source activate.sh" >&2
    exit 1
fi

source /usr/share/lmod/lmod/init/bash
module load conda/25.3 cuda/12.9
_phasefieldx_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${_phasefieldx_root}/.conda"
unset _phasefieldx_root

# Lmod can prepend its base Conda bin ahead of the active environment when
# another Conda environment was already active. Keep this environment first.
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r
if [[ "$(command -v python)" != "${CONDA_PREFIX}/bin/python" ]]; then
    echo "PhaseFieldX activation failed: python does not resolve to ${CONDA_PREFIX}/bin/python" >&2
    return 1
fi

# The conda MPICH build is not GPU-aware; PETSc stages MPI transfers on host.
export PETSC_OPTIONS="${PETSC_OPTIONS:-} -use_gpu_aware_mpi 0"
export PHASEFIELDX_USE_CUDA=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONDONTWRITEBYTECODE=1
