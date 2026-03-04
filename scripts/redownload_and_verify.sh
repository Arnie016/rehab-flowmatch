#!/usr/bin/env bash
set -euo pipefail

if [[ "${#}" -lt 1 ]]; then
  echo "Usage: $0 <dataset_url> [archive_name]"
  echo "Example: $0 https://example.com/study30-latest.tar.gz study30-latest.tar.gz"
  exit 1
fi

DATASET_URL="$1"
ARCHIVE_NAME="${2:-study30-latest.tar.gz}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DIR="${HOME}/datasets/strokerehab/downloads"
RAW_DIR="${HOME}/datasets/strokerehab/raw/study30-latest"
REPO_DATA_DIR="${PROJECT_ROOT}/data/strokerehab"

mkdir -p "${DOWNLOAD_DIR}" "${RAW_DIR}" "${PROJECT_ROOT}/data/manifests"
cd "${DOWNLOAD_DIR}"

echo "[1/6] Downloading archive to ${DOWNLOAD_DIR}/${ARCHIVE_NAME}"
if command -v aria2c >/dev/null 2>&1; then
  aria2c -c -x 8 -s 8 -k 1M "${DATASET_URL}" -o "${ARCHIVE_NAME}"
else
  curl -L --fail --retry 10 --retry-delay 2 -o "${ARCHIVE_NAME}" "${DATASET_URL}"
fi

echo "[2/6] Verifying gzip integrity"
gzip -t "${ARCHIVE_NAME}"

echo "[3/6] Verifying tar listing"
tar -tzf "${ARCHIVE_NAME}" >/dev/null

echo "[4/6] Recording archive checksum"
shasum -a 256 "${ARCHIVE_NAME}" | tee "${ARCHIVE_NAME}.sha256"

echo "[5/6] Extracting archive to ${RAW_DIR}"
rm -rf "${RAW_DIR}"
mkdir -p "${RAW_DIR}"
tar -xzf "${ARCHIVE_NAME}" -C "${RAW_DIR}"

echo "[6/6] Pointing repo data/strokerehab to extracted data"
mkdir -p "${PROJECT_ROOT}/data"
rm -rf "${REPO_DATA_DIR}"
ln -s "${RAW_DIR}" "${REPO_DATA_DIR}"

echo
echo "Done. Next steps:"
echo "  cd \"${PROJECT_ROOT}\""
echo "  python3 tools/make_manifest.py --root data/strokerehab --out data/manifests/strokerehab.study30-latest.manifest.json"
echo "  python3 tools/dataset_healthcheck.py --root data/strokerehab --manifest data/manifests/strokerehab.study30-latest.manifest.json --verify-sha256"

