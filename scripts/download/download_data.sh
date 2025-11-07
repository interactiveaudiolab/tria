#!/bin/bash
set -e

################################################################################
# Environment
################################################################################

DL_SCRIPTS_DIR=$(eval dirname "$(readlink -f "$0")")
SCRIPTS_DIR="$(dirname "$DL_SCRIPTS_DIR")"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"
LOCAL_DATA_DIR="${PROJECT_DIR}/data"

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <DATA-DIR>"
  exit 1
fi

# Normalize path, create if missing
DATA_DIR=$(echo "$1" | sed 's:/*$::')
mkdir -p "$DATA_DIR"

echo "Target data dir: $DATA_DIR"
echo "Project data symlink: $LOCAL_DATA_DIR"

# Handle existing $LOCAL_DATA_DIR
if [ -L "$LOCAL_DATA_DIR" ]; then
  CURRENT_TARGET=$(readlink "$LOCAL_DATA_DIR")
  if [ "$CURRENT_TARGET" = "$DATA_DIR" ]; then
    echo "Symlink already points to $DATA_DIR — leaving as-is."
  else
    echo "Updating symlink: $LOCAL_DATA_DIR -> $DATA_DIR"
    rm "$LOCAL_DATA_DIR"
    ln -s "$DATA_DIR" "$LOCAL_DATA_DIR"
  fi
elif [ -e "$LOCAL_DATA_DIR" ]; then
  # Exists but not a symlink — don't touch
  echo "Error: $LOCAL_DATA_DIR exists and is not a symlink. Please remove or rename it."
  exit 1
else
  echo "Creating symlink: $LOCAL_DATA_DIR -> $DATA_DIR"
  ln -s "$DATA_DIR" "$LOCAL_DATA_DIR"
fi

################################################################################
# Helpers
################################################################################

dir_nonempty () { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null || true)" ]; }

################################################################################
# Download data (~26G) — SKIP if already present
################################################################################

# MUSDB18-HQ (23G)
MUS_DIR="$DATA_DIR/musdb_hq"
if dir_nonempty "$MUS_DIR"; then
  echo "Found MUSDB18-HQ at $MUS_DIR — skipping."
else
  echo "Downloading MUSDB18-HQ..."
  mkdir -p "$MUS_DIR"
  wget -O "$DATA_DIR/musdb18hq.zip" \
    https://zenodo.org/records/3338373/files/musdb18hq.zip
  echo "Unzipping MUSDB18-HQ..."
  unzip -q "$DATA_DIR/musdb18hq.zip" -d "$MUS_DIR"
  rm -f "$DATA_DIR/musdb18hq.zip"
fi

echo "Done."
