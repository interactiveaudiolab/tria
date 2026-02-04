#!/bin/bash
set -euo pipefail

################################################################################
# Environment
################################################################################

DL_SCRIPTS_DIR=$(eval dirname "$(readlink -f "$0")")
SCRIPTS_DIR="$(dirname "$DL_SCRIPTS_DIR")"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"
LOCAL_DATA_DIR="${PROJECT_DIR}/data"

if [ ! -e "$LOCAL_DATA_DIR" ]; then
  echo "Error: expected $LOCAL_DATA_DIR to exist. Run `download_data.sh` to create symlink."
  exit 1
fi

if [ -L "$LOCAL_DATA_DIR" ]; then
  DATA_DIR=$(readlink "$LOCAL_DATA_DIR")
else
  DATA_DIR="$LOCAL_DATA_DIR"
fi

echo "Using data directory: $DATA_DIR"

################################################################################
# Helpers
################################################################################

dir_nonempty () { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null || true)" ]; }

cleanup_partial () {
  rm -f "$DATA_DIR/musan.tar.gz" || true
  rm -f "$DATA_DIR/Audio.zip" || true
  rm -f "$DATA_DIR/high-res-wham.zip" || true
}
trap cleanup_partial EXIT

mkdir -p "$DATA_DIR"

################################################################################
# FreeSound Loop Dataset (11G)
################################################################################

FSL="$DATA_DIR/FSL10K"

if dir_nonempty "$FSL"; then
  echo "Found FSL — skipping."
else
  echo "Downloading FSL..."
  mkdir -p "$FSL"

  wget -O "$DATA_DIR/FSL10K.zip" \
    https://zenodo.org/records/3967852/files/FSL10K.zip
  echo "Unzipping FSL..."
  unzip -q "$DATA_DIR/FSL10K.zip" -d "$FSL"
  rm -f "$DATA_DIR/FSL10K.zip"
fi

echo "Done."
