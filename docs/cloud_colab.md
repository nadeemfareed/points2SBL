# Google Colab and cloud execution

## Colab

```python
!pip -q install points2sbl
!points2sbl model download
```

Mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Example:

```python
!points2sbl predict \
  --mode full \
  --in_las "/content/drive/MyDrive/input.las" \
  --out_las "/content/drive/MyDrive/output_points2sbl.las"
```

Use a GPU runtime when available.

## Large datasets

For many LAS/LAZ files, use folder mode. For a single acquisition that is
larger than Colab/workstation RAM, pre-tile the source dataset first. Internal
network inference is tiled, but the current source-file reader is not an
out-of-core LAS streamer.
