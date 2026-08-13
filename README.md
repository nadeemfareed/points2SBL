<div align="center">

# 🌲 points2SBL

> **Toward Operational Wildfire Fuel Mapping: Sensor-Agnostic Deep Learning Semantic Segmentation of Terrestrial LiDAR Across Global Forest Ecosystems**

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)
![License](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Paper](https://img.shields.io/badge/Paper-Under%20Review-blue.svg)

</div>

<p align="center">
<img src="docs/images/points2sbl_pipeline_banner.png" width="96%" alt="points2SBL processing pipeline">
</p>

**points2SBL** is an open-source framework for binary semantic segmentation of forest LiDAR point clouds into **woody** and **foliar** components. The current release is designed to make pretrained-model inference straightforward for plot-scale and individual-tree point clouds while retaining the complete data-preparation and training workflow for advanced users.
# The research behind points2SBL is now available as a preprint, providing the scientific basis for our deep-learning approach to wood–foliar semantics of forest LiDAR point clouds https://www.preprints.org/manuscript/202608.0737

The recommended production model is the **Point Transformer**. PointNeXt and PointNet++ remain available for comparison, ablation, and alternative deployment requirements.

---

## Highlights

| Capability | Support |
|---|:---:|
| Wood–leaf segmentation | ✅ |
| Point Transformer | ✅ Recommended |
| TLS plots | ✅ |
| MLS / BLS / PLS | ✅ |
| ULS | ✅ |
| CPU and CUDA execution | ✅ |
| Single-file inference | ✅ |
| Recursive folder inference | ✅ |
| Automatic plot/tree detection | ✅ |
| Multi-vote probability aggregation | ✅ |
| Spatial confidence weighting | ✅ |
| Woody-structure refinement | ✅ |
| Prediction probability export | ✅ |
| JSON inference sidecars | ✅ |

---

# Installation

points2SBL can be installed from **PyPI**, **GitHub**, or used in **Google Colab**.

USE: https://www.anaconda.com/download

> **Recommended:** Python 3.11 and a CUDA-enabled PyTorch installation for local GPU inference.

---

## Option 1 — PyPI

Create a clean environment:

```powershell
conda create -n points2sbl python=3.11 -y
conda activate points2sbl
```

Install the validated CUDA build of PyTorch:

```powershell
python -m pip install `
  torch==2.5.1 `
  torchvision==0.20.1 `
  torchaudio==2.5.1 `
  --index-url https://download.pytorch.org/whl/cu121
```

Install points2SBL:

```powershell
pip install points2sbl
```

Download and verify the pretrained model:

```powershell
points2sbl model download
points2sbl model status
```

---

## Option 2 — GitHub

```powershell
git clone https://github.com/nadeemfareed/points2SBL.git
cd points2SBL

conda create -n points2sbl python=3.11 -y
conda activate points2sbl
```

Install PyTorch:

```powershell
python -m pip install `
  torch==2.5.1 `
  torchvision==0.20.1 `
  torchaudio==2.5.1 `
  --index-url https://download.pytorch.org/whl/cu121
```

Install points2SBL:

```powershell
python -m pip install -e .
```

Download and verify the pretrained model:

```powershell
points2sbl model download
points2sbl model status
```

---

## Option 3 — Google Colab

```python
!pip install points2sbl
!points2sbl model download
!points2sbl model status
```

Enable a GPU runtime in Colab before inference.

---

# Inference

points2SBL accepts `.las` and `.laz` point clouds.

Use:

- `plot` for forest plots or multi-tree scenes.
- `single_tree` for isolated trees.
- `full` as the recommended inference mode.

---

## Ground classification for forest plots

> **Important:** For forest plots containing terrain, ground classification should be performed before points2SBL inference. Ground points should use the standard LAS **Classification = 2**.

[FAST-GC](https://github.com/nadeemfareed/FAST-GC) is recommended for ground classification before points2SBL.

FAST-GC can be installed directly with:

```powershell
pip install fastgc
```

For plot-level processing, the recommended order is:

```text
LAS/LAZ point cloud
        ↓
FAST-GC
Ground = Classification 2
        ↓
points2SBL
        ↓
Wood = 0 | Leaf = 1 | Ground = 2
```

See the [FAST-GC repository](https://github.com/nadeemfareed/FAST-GC) for the current recommended TLS ground-classification command and usage.

> FAST-GC preprocessing is not required for isolated individual-tree point clouds that do not contain ground.

---

## Forest plot — recommended

Use this configuration for routine plot processing.

```powershell
points2sbl predict `
  --input_type plot `
  --mode full `
  --config ".\configs\point_transformer.yaml" `
  --ckpt ".\runs\point_transformer_curated_20260327_170108\best.pt" `
  --in_las "D:\input\forest_plot.las" `
  --out_las "D:\output\forest_plot_points2sbl.las" `
  --device cuda `
  --geom_cache all `
  --progress tiles
```

---

## Forest plot — maximum quality

Use this configuration when prediction quality is preferred over processing time.

```powershell
points2sbl predict `
  --input_type plot `
  --mode full `
  --config ".\configs\point_transformer.yaml" `
  --ckpt ".\runs\point_transformer_curated_20260327_170108\best.pt" `
  --in_las "D:\input\forest_plot.las" `
  --out_las "D:\output\forest_plot_HYBRID8_V10.las" `
  --device cuda `
  --votes 10 `
  --vote_mode hybrid8 `
  --vote_weight confidence `
  --geom_cache all `
  --progress tiles
```

This was the highest-performing configuration in our validation tests.

---

## Forest plot — faster multi-vote

For large datasets, four deterministic votes provide a useful quality/runtime compromise.

```powershell
points2sbl predict `
  --input_type plot `
  --mode full `
  --config ".\configs\point_transformer.yaml" `
  --ckpt ".\runs\point_transformer_curated_20260327_170108\best.pt" `
  --in_las "D:\input\forest_plot.las" `
  --out_las "D:\output\forest_plot_GRID4_V4.las" `
  --device cuda `
  --votes 4 `
  --vote_mode grid4 `
  --vote_weight confidence `
  --geom_cache all `
  --progress tiles
```

---

## Individual tree

Use `single_tree` for isolated trees. Tile size is selected automatically.

```powershell
points2sbl predict `
  --input_type single_tree `
  --mode full `
  --config ".\configs\point_transformer.yaml" `
  --ckpt ".\runs\point_transformer_curated_20260327_170108\best.pt" `
  --in_las "D:\input\tree_001.las" `
  --out_las "D:\output\tree_001_points2sbl.las" `
  --device cuda `
  --geom_cache all `
  --progress tiles
```

---

# Batch processing

## Forest plots

For plot datasets containing terrain, perform ground classification with FAST-GC before running the batch.

```powershell
points2sbl predict `
  --input_type plot `
  --mode full `
  --config ".\configs\point_transformer.yaml" `
  --ckpt ".\runs\point_transformer_curated_20260327_170108\best.pt" `
  --in_dir "D:\input\forest_plots" `
  --out_dir "D:\output\forest_plots_points2sbl" `
  --recursive `
  --skip_existing `
  --device cuda `
  --geom_cache all `
  --progress tiles
```

## Individual trees

```powershell
points2sbl predict `
  --input_type single_tree `
  --mode full `
  --config ".\configs\point_transformer.yaml" `
  --ckpt ".\runs\point_transformer_curated_20260327_170108\best.pt" `
  --in_dir "D:\input\single_trees" `
  --out_dir "D:\output\single_trees_points2sbl" `
  --recursive `
  --skip_existing `
  --device cuda `
  --geom_cache all `
  --progress tiles
```

---

# Output classes

| Classification | Class |
|---:|---|
| `0` | Wood |
| `1` | Leaf / needle |
| `2` | Ground, when present |

The output point cloud also contains prediction information including `pred_class` and `pred_leaf_prob`.

---

# Troubleshooting

## Check installation

```powershell
points2sbl --help
points2sbl model status
```

## CUDA is unavailable

Check the installed PyTorch build:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns `False`, install a CUDA-enabled PyTorch build.

## Download the model again

```powershell
points2sbl model download --force
```

## GPU out of memory

Reduce the inference batch size:

```powershell
--batch_blocks 8
```

If necessary:

```powershell
--batch_blocks 4
```

---

# Help

View all available command-line options with:

```powershell
points2sbl predict --help
```
---

# Output

The output LAS/LAZ preserves the original point geometry and supported LAS attributes while adding prediction information.

| Attribute | Meaning |
|---|---|
| `Classification` | Final class when classification overwrite is enabled |
| `pred_class` | Binary predicted class |
| `pred_leaf_prob` | Aggregated probability of the leaf class |
| JSON sidecar | Resolved settings and inference diagnostics |



# Advanced inference controls

The public interface intentionally keeps common workflows simple. Advanced parameters remain available for expert experiments.

## Voting

```text
--votes
--vote_mode {grid4,grid8,hybrid8,random}
--vote_weight {uniform,confidence}
```

## Geometry cache

```text
--geom_cache {none,all}
```

`all` computes geometric features once for the prediction points and reuses them during inference.

## Full-mode refinement

Advanced controls include:

```text
--t_low
--t_high
--geom_rescue_thr
--local_woody_thr
--local_leaf_thr
--smooth_k
--smooth_tau
--woody_refine_k
--woody_core_p_leaf_max
--woody_structure_k
```

## Adaptive mode

Advanced adaptive parameters include:

```text
--adaptive_hist_bins
--adaptive_hist_smooth_sigma
--adaptive_shoulder_fraction
--adaptive_min_transition_width
--adaptive_geom_ratio
--adaptive_local_support_min
```

Most users should use the mode defaults rather than changing these parameters.

---

# Performance notes

The current Point Transformer inference implementation reuses shared neighborhood information inside the model to reduce redundant computation during inference.

Runtime depends on:

- total number of points;
- tile size;
- number of overlapping layouts/votes;
- number of model blocks;
- geometric-feature computation;
- semantic refinement;
- GPU capability;
- disk speed.

On CUDA-capable systems, points2SBL automatically limits the default inference batch on lower-VRAM GPUs to reduce out-of-memory failures. Explicit `--batch_blocks` values override the automatic choice.

---

# Troubleshooting

## CUDA requested but unavailable

Check:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

A CUDA-capable GPU and driver are not sufficient by themselves; PyTorch must also be installed with CUDA support.

## Editable installation replaced CUDA PyTorch

Install the desired CUDA PyTorch build first, then reinstall points2SBL without dependencies:

```powershell
python -m pip install -e . --no-deps
```

## OpenMP conflict on Windows

An error such as:

```text
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized
```

means multiple OpenMP runtimes were loaded.

The preferred solution is to remove conflicting package builds and use a clean, consistent environment.

The following workaround may allow execution but is not recommended for production:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```.
## Automatic input detection is inappropriate

Override it explicitly:

```powershell
--input_type plot
```

or:

```powershell
--input_type single_tree
```

## Need the unrefined network result

Use:

```powershell
--mode raw
```

## Need probability-distribution-driven thresholding

Use:

```powershell
--mode adaptive
```

## GPU out of memory

Reduce:

```powershell
--batch_blocks 8
```

or:

```powershell
--batch_blocks 4
```

Keep the checkpoint-compatible number of points per block unless intentionally testing another model configuration.

## Resume an interrupted folder run

Use:

```powershell
--skip_existing
```

Previously completed outputs are skipped and the remaining files are processed.

---

# Results gallery

The image paths below intentionally retain the existing repository filenames.

## Training datasets

<p align="center">
<img src="docs/images/training_datasets.png" width="96%" alt="Training datasets">
</p>

Representative labeled forest point clouds used for model development.

## Benchmark datasets

<p align="center">
<img src="docs/images/benchmark_datasets.png" width="96%" alt="Benchmark datasets">
</p>

Independent datasets used to assess segmentation accuracy and transferability.

## Model performance

<p align="center">
<img src="docs/images/model_performance.png" width="88%" alt="Model performance">
</p>

Comparative performance of Point Transformer, PointNet++, and PointNeXt.

## Tropical forest generalization

<p align="center">
<img src="docs/images/tropical_tree.png" width="92%" alt="Tropical forest predictions">
</p>

Wood–leaf prediction on structurally complex tropical trees not used during training.

## Lin3D benchmark

<p align="center">
<img src="docs/images/lin3d.png" width="96%" alt="Lin3D prediction">
</p>

Reference labels and Point Transformer predictions for complex plot-level forest scenes.

## Large registered TLS plot

<p align="center">
<img src="docs/images/usa_tls_plot01.png" width="96%" alt="Registered TLS plot">
</p>

Large-scale registered TLS prediction processed through block-wise multi-vote inference.

## ULS prediction

<p align="center">
<img src="docs/images/ULS_OFF_BR06.png" width="96%" alt="ULS prediction">
</p>

Prediction on lower-density ULS data.

## BlueCat qualitative result

<p align="center">
<img src="docs/images/bluecat.png" width="96%" alt="BlueCat prediction">
</p>

Prediction on the structurally complex BlueCat TLS dataset.

## BlueCat reference, prediction, probability, and disagreement

<p align="center">
<img src="docs/images/BlueCat_reference_prediction_probability_disagreement_XZ_front_surface.png" width="96%" alt="BlueCat probability disagreement">
</p>

Reference labels, final prediction, leaf probability, and probability disagreement shown from complementary views.

> Keep the corresponding PNG files under `docs/images/`. If GitHub filenames differ, update the paths above to match the repository exactly.

---

# Advanced: data preparation

Most users using the released pretrained model can skip this section.

The transferable training workflow uses a common block representation across TLS, MLS, ULS, plot clouds, and individual trees.

## Mixed-sensor training corpus

Example layout:

```text
D:\points2SBL_training_raw\
├── TLS_plots\
├── TLS_single_trees\
├── MLS\
└── ULS\
```

Prepare the combined corpus:

```powershell
python -u -m points2sbl.prepare_data `
  --config "configs\point_transformer.yaml" `
  --data_root "D:\points2SBL_training_raw" `
  --recursive `
  --label_field Classification `
  --leaf_class 1 `
  --xy_size 2.0 2.0 `
  --stride 1.0 1.0 `
  --n_points 8192 `
  --min_points 64 `
  --val_ratio 0.20 `
  --rotate_train `
  --save_format npz
```

## TLS plot preparation

```powershell
python -u -m points2sbl.prepare_data `
  --config "configs\point_transformer.yaml" `
  --data_root "D:\training\TLS_plots" `
  --recursive `
  --label_field Classification `
  --leaf_class 1 `
  --xy_size 2.0 2.0 `
  --stride 1.0 1.0 `
  --n_points 8192 `
  --min_points 64 `
  --val_ratio 0.20 `
  --rotate_train `
  --save_format npz
```

## MLS / BLS / PLS preparation

```powershell
python -u -m points2sbl.prepare_data `
  --config "configs\point_transformer.yaml" `
  --data_root "D:\training\MLS" `
  --recursive `
  --label_field Classification `
  --leaf_class 1 `
  --xy_size 2.0 2.0 `
  --stride 1.0 1.0 `
  --n_points 8192 `
  --min_points 64 `
  --val_ratio 0.20 `
  --rotate_train `
  --save_format npz
```

## ULS preparation

A larger block can be used when building a dedicated lower-density ULS model:

```powershell
python -u -m points2sbl.prepare_data `
  --config "configs\point_transformer.yaml" `
  --data_root "D:\training\ULS" `
  --recursive `
  --label_field Classification `
  --leaf_class 1 `
  --xy_size 3.0 3.0 `
  --stride 1.5 1.5 `
  --n_points 8192 `
  --min_points 64 `
  --val_ratio 0.20 `
  --rotate_train `
  --save_format npz
```

## Individual-tree preparation

For a dedicated individual-tree training corpus:

```powershell
python -u -m points2sbl.prepare_data `
  --config "configs\point_transformer.yaml" `
  --data_root "D:\training\single_trees" `
  --recursive `
  --label_field Classification `
  --leaf_class 1 `
  --xy_size 5.0 5.0 `
  --stride 2.5 2.5 `
  --n_points 8192 `
  --min_points 64 `
  --val_ratio 0.20 `
  --rotate_train `
  --save_format npz
```

Prepared datasets typically contain:

```text
<data_root>/
├── train/
├── val/
├── test/                 # when requested
└── _prepare_report.json
```

Before training, inspect `_prepare_report.json` and confirm that files, classes, and train/validation blocks were created as expected.

---

# Advanced: training

Most users using the released checkpoint can skip this section.

Point Transformer is the recommended production architecture. PointNeXt and PointNet++ are retained for comparison and experimentation.

## Point Transformer

```powershell
python -u -m points2sbl.train `
  --config "configs\point_transformer.yaml" `
  --data_root "D:\points2SBL_training_raw" `
  --out_dir "runs\point_transformer_mixed_sensors" `
  --device cuda
```

## PointNeXt

```powershell
python -u -m points2sbl.train `
  --config "configs\pointnext.yaml" `
  --data_root "D:\points2SBL_training_raw" `
  --out_dir "runs\pointnext_mixed_sensors" `
  --device cuda
```

## PointNet++

```powershell
python -u -m points2sbl.train `
  --config "configs\pointnet2.yaml" `
  --data_root "D:\points2SBL_training_raw" `
  --out_dir "runs\pointnet2_mixed_sensors" `
  --device cuda
```

A training run typically produces:

```text
runs/<run_name>/
├── best.pt
├── last.pt
├── config_resolved.json
├── metrics.jsonl
└── train.log
```

Use `best.pt` for production inference unless a specific experiment requires another checkpoint.

---
```

Generated caches, build directories, local checkpoints, temporary patches, and prediction outputs should not be committed to Git.

---

# Citation

If you use points2SBL, please cite the accompanying manuscript:

```bibtex
@article{Nadeem2026points2SBL,
  title   = {Toward Operational Wildfire Fuel Mapping: Sensor-Agnostic Deep Learning Semantic Segmentation of Terrestrial LiDAR Across Global Forest Ecosystems},
  author  = {Nadeem, Fareed et al.},
  journal = {Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

---

# License

points2SBL is released under the GNU General Public License v3.0 (GPL-3.0). See `LICENSE` for details..

---

# Contact

**Fareed Nadeem**  
School of Forest, Fisheries, and Geomatics Sciences  
University of Florida  
nadeem@geomatics.ncku.edu.tw
fareed.nadeem@ufl.edu

GitHub: [nadeemfareed](https://github.com/nadeemfareed)

<div align="center">

**points2SBL — pretrained inference first, reproducible training when needed**

</div>
