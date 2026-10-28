#!/bin/bash
#SBATCH --job-name=uav-hf-p0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=60
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

: "${LSDYNA_EXECUTABLE:?Bind the SCNet-provided LS-DYNA MPP double-precision executable before running}"
DECK="p0_velocity_mesh_calibration.k"
EXPECTED_DECK_SHA256="__BIND_FROM_MANIFEST_BEFORE_SUBMISSION__"

actual_sha256="$(sha256sum "${DECK}" | awk '{print $1}')"
if [[ "sha256:${actual_sha256}" != "${EXPECTED_DECK_SHA256}" ]]; then
  echo "deck hash mismatch" >&2
  exit 65
fi

mkdir -p result
cp "${DECK}" result/
cd result
srun --mpi=pmi2 -n "${SLURM_NTASKS}" "${LSDYNA_EXECUTABLE}" i="${DECK}" jobid=p0 memory=2000m
