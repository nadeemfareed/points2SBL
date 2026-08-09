# Release procedure

## 1. Clean audit

```powershell
powershell -ExecutionPolicy Bypass -File ".\tools\audit_release.ps1"
```

## 2. Build distributions

```bash
python -m build
python -m twine check dist/*
```

## 3. TestPyPI (recommended)

Upload and install from TestPyPI before the production tag.

## 4. GitHub release

Create tag `v0.3.0` and upload the pretrained checkpoint as:

```text
point_transformer_best.pt
```

Expected checkpoint:

```text
size:   18383398 bytes
sha256: fd43c5f83463f00d189292b4d4034bec21f3147c453232c4fbf8336cfd2047f9
```

The release asset must be present before users run `points2sbl model download`.

## 5. PyPI

Configure PyPI Trusted Publishing for the GitHub repository/environment named
`pypi`. Pushing a `v*` tag runs `.github/workflows/release-pypi.yml`.

## 6. conda-forge

After PyPI publication, calculate the PyPI source-distribution SHA256 and replace
the placeholder in `packaging/conda-forge/recipe.yaml`. Submit that recipe to
`conda-forge/staged-recipes`.

## 7. Colab smoke test

Run `examples/colab_quickstart.ipynb` in a fresh GPU runtime.
