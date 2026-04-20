#!/usr/bin/env bash
set -euo pipefail

# Downloads the MovieLens 25M dataset zip into data/raw, extracts it, then removes the zip file.
#
# Usage:
#   ./scripts/download_and_extract.sh
#
# Dataset source:
#   https://files.grouplens.org/datasets/movielens/ml-25m.zip

if [[ $# -ne 0 ]]; then
  echo "Usage: ./scripts/download_and_extract.sh"
  exit 1
fi

ZIP_URL="https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_NAME="ml-25m.zip"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
ZIP_PATH="${RAW_DIR}/${ZIP_NAME}"

mkdir -p "${RAW_DIR}"

download_file() {
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$1" "$2"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$1" "$2"
  else
    echo "Error: neither curl nor wget is installed."
    exit 1
  fi
}

extract_zip() {
  if command -v unzip >/dev/null 2>&1; then
    unzip -o "$1" -d "$2"
  elif command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$1" -C "$2"
  else
    echo "Error: neither unzip nor bsdtar is installed."
    exit 1
  fi
}

echo "Downloading ${ZIP_URL} -> ${ZIP_PATH}"
download_file "${ZIP_PATH}" "${ZIP_URL}"

echo "Extracting ${ZIP_PATH} into ${RAW_DIR}"
extract_zip "${ZIP_PATH}" "${RAW_DIR}"

echo "Removing ${ZIP_PATH}"
rm -f "${ZIP_PATH}"

echo "Done. MovieLens 25M extracted in ${RAW_DIR}/ml-25m."
