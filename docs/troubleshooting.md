# Troubleshooting

## Windows OpenMP runtime conflict

A Windows environment can fail during geometric-feature computation with a
message similar to:

```text
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```

This indicates two OpenMP runtimes were loaded by compiled dependencies. The
preferred fix is a clean environment with a consistent binary package stack.

Recommended clean bootstrap:

```powershell
conda create -n points2sbl python=3.11 pip -c conda-forge
conda activate points2sbl
python -m pip install points2sbl
points2sbl model download
```

If GPU PyTorch must be installed separately, install the appropriate PyTorch
build first, then install points2SBL.

`KMP_DUPLICATE_LIB_OK=TRUE` may permit a diagnostic run, but it is not enabled
by points2SBL and is not recommended as a permanent package configuration.

## CUDA requested but unavailable

The predictor reports the unavailable CUDA device and falls back to CPU. For
GPU processing, verify:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

## Model not found

```bash
points2sbl model status
points2sbl model download
```

or set `POINTS2SBL_CKPT` to a verified checkpoint.

## Massive LAS/LAZ file exceeds RAM

The current reader loads one source LAS/LAZ file into host memory. Pre-tile very
large acquisitions into manageable LAS/LAZ files, then use folder inference.
