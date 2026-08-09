# Dataset preparation and retraining

The primary release workflow is pretrained inference. Retraining is provided
for researchers who need to build a model from their own labeled data.

## Label contract

The binary training target is:

- `Classification = 0`: woody / non-leaf
- `Classification = 1`: leaf

Ground or other non-target classes should not leak into binary supervised
labels. Preserve them separately when needed.

## Prepare blocks

Inspect the preparation interface:

```bash
python -m points2sbl.prepare_data --help
```

The released Point Transformer configuration uses 8192 points per block and a
7-channel feature contract based on centered XYZ plus linearity, planarity,
scattering, and curvature.

## Validate the configuration

```bash
points2sbl validate-config --config configs/point_transformer.yaml
```

Expected Point Transformer feature width:

```text
runtime.in_dim = 7
```

## Train

```bash
python -m points2sbl.train \
  --config configs/point_transformer.yaml \
  --data_root /path/to/prepared_data \
  --out_dir /path/to/new_run
```

Windows PowerShell:

```powershell
python -m points2sbl.train `
  --config ".\configs\point_transformer.yaml" `
  --data_root "C:\path\to\prepared_data" `
  --out_dir "C:\path\to\new_run"
```

## Checkpoint selection

Training writes `best.pt` and `last.pt`. Use `best.pt` for normal deployment.
The checkpoint embeds its training configuration; inference prefers that
embedded model/feature configuration to reduce feature/model mismatch risk.

## Releasing a replacement pretrained model

Do not commit checkpoints to the Git repository. Upload the approved
`best.pt` as the GitHub release asset `point_transformer_best.pt`, update the
SHA256 in `src/points2sbl/model_manager.py`, run the release audit, and tag a new
package version.
