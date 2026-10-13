#!/usr/bin/env bash
# Use: source activate.sh
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source activate.sh" >&2
    exit 1
fi

_sem_activate() {
    local root prefix conda_base backend
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)" || return 1
    prefix="${SEM_ENV_PREFIX:-$root/.conda}"
    [[ "$prefix" == /* ]] || prefix="$PWD/$prefix"
    backend="${SEM_PETSC_BACKEND:-auto}"
    case "$backend" in auto|cpu|gpu) ;; *) echo "SEM_PETSC_BACKEND must be auto, cpu, or gpu" >&2; return 1 ;; esac

    if ! type module >/dev/null 2>&1 && [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        source /usr/share/lmod/lmod/init/bash || return 1
    fi
    if ! command -v conda >/dev/null 2>&1 && type module >/dev/null 2>&1; then
        module load conda/25.3 || return 1
    fi
    command -v conda >/dev/null 2>&1 || { echo "Conda is required; install Miniforge or load your Conda module." >&2; return 1; }
    conda_base="$(conda info --base)" || return 1
    source "$conda_base/etc/profile.d/conda.sh" || return 1

    if [[ "$backend" == auto ]]; then
        backend=cpu
        if type module >/dev/null 2>&1 && module load cuda/12.9; then
            backend=gpu
        fi
    elif [[ "$backend" == gpu ]] && type module >/dev/null 2>&1; then
        module load cuda/12.9 || return 1
    fi
    # An existing CUDA environment needs its libraries even for CPU solves.
    if [[ "$backend" == cpu ]] && compgen -G "$prefix/conda-meta/petsc-*-cuda*.json" >/dev/null && type module >/dev/null 2>&1; then
        module load cuda/12.9 || return 1
    fi

    if [[ ! -f "$prefix/conda-meta/history" ]]; then
        conda env create --yes --prefix "$prefix" --file "$root/environment.yml" || return 1
    fi
    conda activate "$prefix" || return 1
    export PATH="$CONDA_PREFIX/bin:$PATH"
    hash -r
    [[ "$(command -v python)" == "$CONDA_PREFIX/bin/python" ]] || return 1
    if [[ " ${PETSC_OPTIONS:-} " != *" -use_gpu_aware_mpi "* ]]; then
        export PETSC_OPTIONS="${PETSC_OPTIONS:+$PETSC_OPTIONS }-use_gpu_aware_mpi 0"
    fi
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export PYTHONDONTWRITEBYTECODE=1

    # DOLFINx metadata pins CPU PETSc. Replace only that same-version library
    # for CUDA; the cluster cuda/12.9 module supplies its runtime libraries.
    if [[ "$backend" == gpu ]] && ! python -c 'from petsc4py import PETSc; raise SystemExit(not PETSc.Sys.hasExternalPackage("cuda"))'; then
        conda install --yes --prefix "$prefix" --no-deps \
            'https://conda.anaconda.org/conda-forge/linux-64/petsc-3.25.6-cuda12_real_h0d6b94f_0.conda' || return 1
    fi
    python - "$backend" <<'PY'
import sys, subprocess
import dolfinx, numpy, scipy, PIL, skimage, shapely, gmsh, matplotlib, cffi
from mpi4py import MPI
from petsc4py import PETSc
assert dolfinx.__version__ == "0.11.0" and PETSc.Sys.getVersion() == (3, 25, 6)
if sys.argv[1] == "gpu":
    assert PETSc.Sys.hasExternalPackage("cuda"), "CUDA PETSc unavailable"
subprocess.run([sys.prefix+"/bin/ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL)
print("SEM environment ready:", sys.executable)
PY
}

if _sem_activate; then
    unset -f _sem_activate
else
    unset -f _sem_activate
    echo "SEM environment setup failed." >&2
    return 1
fi
