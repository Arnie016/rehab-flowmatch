#!/usr/bin/env python3
"""Build a deterministic dataset manifest with file sizes and SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
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


def build_manifest(root: Path) -> dict:
    files = []
    for entry in sorted(root.rglob("*")):
        if not entry.is_file():
            continue
        rel = entry.relative_to(root).as_posix()
        files.append(
            {
                "path": rel,
                "bytes": entry.stat().st_size,
                "sha256": sha256_file(entry),
            }
        )

    return {
        "schema": "strokerehab-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": root.as_posix(),
        "file_count": len(files),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a dataset manifest with checksums."
    )
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--out", required=True, help="Manifest output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not root.exists() or not root.is_dir():
        print(f"FAIL: root directory does not exist: {root}")
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {manifest['file_count']} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

