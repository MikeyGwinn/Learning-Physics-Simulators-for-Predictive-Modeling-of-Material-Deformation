#!/bin/bash
#SBATCH -p share
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -J moose
#SBATCH -o moose_%j.log
# Usage: sbatch scripts/moose_container.sbatch path/to/input.i
# Runs MOOSE (combined-opt) inside INL's container. See logs/ for setup notes.
SIF=/nfs/hpc/share/brockert/capstone/moose/moose.sif
INPUT=${1:?give a MOOSE input file}
apptainer exec --cleanenv -B /nfs/hpc/share "$SIF" bash -c "
  export PATH=/opt/openmpi/bin:/opt/python/bin:/opt/moose/bin:\$PATH
  cd \"$(dirname "$INPUT")\" && mpiexec -n ${SLURM_CPUS_PER_TASK:-1} combined-opt -i \"$(basename "$INPUT")\""
