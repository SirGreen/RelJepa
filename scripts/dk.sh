#!/usr/bin/env bash
# Run a command inside the pinned environment container.
#
# Usage (from Git Bash on the host):
#   scripts/dk.sh <command...>                  # foreground, removed on exit
#   DK_NAME=myrun scripts/dk.sh <command...>    # detached, named container; follow with
#                                               #   docker logs -f myrun
# The working directory inside the container is the vendored TVE root, so TVE commands such as
# `python src/train_baseline.py -c cfg/...` work unchanged. Large data and model weights live in
# Docker volumes (fast ext4 I/O); code, logs and results are on the bind-mounted project folder.
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT_WIN="$(cd "$(dirname "$0")/.." && pwd -W)"
TVE=/workspace/third_party/task-vector-estimation
IMAGE="${DK_IMAGE:-reljepa-env:1}"
# Commit of this repo for the run record; "-dirty" marks uncommitted changes.
GIT_COMMIT="$(git -C "${ROOT_WIN}" rev-parse --short=12 HEAD)"
if [[ -n "$(git -C "${ROOT_WIN}" status --porcelain --untracked-files=no)" ]]; then
  GIT_COMMIT="${GIT_COMMIT}-dirty"
fi

args=(--gpus all --shm-size=8g
      -v "${ROOT_WIN}:/workspace"
      -v "reljepa-data:${TVE}/data"
      -v "reljepa-weights:${TVE}/weights"
      -e WANDB_MODE=offline
      -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
      -e TRANSFORMERS_OFFLINE="${HF_HUB_OFFLINE:-1}"
      -e PYTHONPATH=/workspace/src
      -e GIT_COMMIT="${GIT_COMMIT}"
      -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
      -w "${TVE}")

if [[ -n "${DK_NAME:-}" ]]; then
  docker run -d --name "${DK_NAME}" "${args[@]}" "${IMAGE}" "$@"
else
  docker run --rm "${args[@]}" "${IMAGE}" "$@"
fi
