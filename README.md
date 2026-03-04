# rehab-flowmatch

Physics-constrained flow matching for rehabilitation motion trajectory denoising and generation, with optional SNN/neuromorphic extensions.

## Project focus

- Learn a time-conditioned velocity field that maps noisy IMU trajectories to clean trajectories.
- Enforce kinematic plausibility (smoothness, limits, constraint checks).
- Build reproducible dataset handling so training never runs on silently broken data.
- Automate report syncing from GitHub to Overleaf.

## Current structure

- `tools/make_manifest.py`: create dataset manifest with file sizes and SHA-256 hashes.
- `tools/dataset_healthcheck.py`: fail-loud integrity check against manifest.
- `scripts/redownload_and_verify.sh`: safe download + archive verification + extraction helper.
- `.github/workflows/overleaf-sync.yml`: sync `report/` folder to Overleaf using token-based auth.
- `data/manifests/`: place dataset manifests here.
- `data/strokerehab/`: expected dataset root.

## Dataset workflow (recommended)

1. Download archive into a stable, non-synced directory.
2. Verify with `gzip -t` and `tar -tzf`.
3. Extract to `data/strokerehab` (or symlink this path to your dataset root).
4. Generate manifest:

```bash
python3 tools/make_manifest.py \
  --root data/strokerehab \
  --out data/manifests/strokerehab.study30-latest.manifest.json
```

5. Validate before training:

```bash
python3 tools/dataset_healthcheck.py \
  --root data/strokerehab \
  --manifest data/manifests/strokerehab.study30-latest.manifest.json \
  --verify-sha256
```

## Overleaf automation

Set these GitHub repository secrets:

- `OVERLEAF_PROJECT_ID`
- `OVERLEAF_GIT_TOKEN`

Then push changes under `report/` to `main`, or trigger the workflow manually.
