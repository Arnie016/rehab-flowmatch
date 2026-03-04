#!/usr/bin/env python3
"""Fail-loud dataset presence/integrity healthcheck."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset against manifest.")
    parser.add_argument(
        "--root",
        default="data/strokerehab",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/strokerehab.study30-latest.manifest.json",
        help="Manifest JSON path",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.98,
        help="Minimum fraction of expected files that must exist (0-1).",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Verify SHA-256 for all files in the manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not root.exists():
        print(f"FAIL: dataset root missing: {root}")
        return 2

    if not manifest_path.exists():
        print(f"FAIL: manifest missing: {manifest_path}")
        return 3

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_files = manifest.get("files", [])
    expected_count = int(manifest.get("file_count", len(expected_files)))

    if expected_count <= 0:
        print("FAIL: manifest file_count is 0")
        return 4

    present_count = 0
    missing = []
    size_mismatch = []
    hash_mismatch = []

    for rec in expected_files:
        rel = rec["path"]
        expected_bytes = int(rec["bytes"])
        expected_sha = rec.get("sha256")
        file_path = root / rel

        if not file_path.exists():
            missing.append(rel)
            continue

        present_count += 1
        actual_bytes = file_path.stat().st_size
        if actual_bytes != expected_bytes:
            size_mismatch.append((rel, expected_bytes, actual_bytes))

        if args.verify_sha256 and expected_sha:
            actual_sha = sha256_file(file_path)
            if actual_sha != expected_sha:
                hash_mismatch.append((rel, expected_sha, actual_sha))

    min_required = int(expected_count * args.min_fraction)
    if present_count < min_required:
        print(
            f"FAIL: too few files present. expected={expected_count}, "
            f"present={present_count}, required>={min_required}"
        )
        if missing:
            print("First missing files:")
            for rel in missing[:20]:
                print(f"  - {rel}")
        return 5

    if size_mismatch:
        print(f"FAIL: file size mismatches={len(size_mismatch)}")
        for rel, exp, got in size_mismatch[:20]:
            print(f"  - {rel}: expected={exp} got={got}")
        return 6

    if hash_mismatch:
        print(f"FAIL: sha256 mismatches={len(hash_mismatch)}")
        for rel, exp, got in hash_mismatch[:10]:
            print(f"  - {rel}: expected={exp} got={got}")
        return 7

    print(
        f"OK: dataset healthcheck passed. expected={expected_count} present={present_count} "
        f"verify_sha256={args.verify_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

