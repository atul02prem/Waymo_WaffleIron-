# AWARE — 3D LiDAR Semantic Segmentation on Waymo Open Dataset

Fine-tuning [WaffleIron](https://github.com/valeoai/WaffleIron) (ICCV 2023) for 3D point cloud semantic segmentation on the [Waymo Open Dataset](https://waymo.com/open/) v1.4.3 with all 22 native semantic classes.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Data Conversion Pipeline](#data-conversion-pipeline)
- [WaffleIron Modifications for Waymo](#waffleiron-modifications-for-waymo)
- [Label Mapping Pipeline](#label-mapping-pipeline)
- [Training](#training)
- [Evaluation and Inference](#evaluation-and-inference)
- [Results](#results)
- [Metadata and Weather Classification](#metadata-and-weather-classification)
- [File Reference](#file-reference)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository adapts the WaffleIron-48-256 backbone for 3D LiDAR semantic segmentation on the Waymo Open Dataset. WaffleIron uses dense 2D convolutions and MLPs instead of sparse 3D convolutions, projecting 3D point clouds onto 2D grids along different axes.

**Key contributions:**
- Conversion pipeline from Waymo v1.4.3 TFRecords to SemanticKITTI format
- WaffleIron dataset adapter for Waymo's 22 native semantic classes (no class collapsing)
- Fine-tuning from pretrained SemanticKITTI weights (19 → 22 classes)
- FP16 training with gradient clipping for stability
- Validation inference pipeline with per-class IoU, confusion matrix, and distance-based accuracy analysis
- CLIP-based weather classification for all 1,150 segments
- LiDAR-based weather detection from point cloud statistics

**Best result:** 55.3% mIoU on Waymo validation set (22 classes)

---

## Environment Setup

### Hardware
- **GPU:** 2× NVIDIA Titan RTX (24 GB each)
- **CUDA:** 12.4
- **OS:** Ubuntu 24, HPC cluster

### Software
```bash
conda create -n waffleiron python=3.10 -y
conda activate waffleiron

# PyTorch
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# WaffleIron dependencies
pip install pyaml tqdm scipy tensorboard protobuf==3.20.3

# Waymo SDK (for data conversion only)
pip install tensorflow==2.13.0 waymo-open-dataset-tf-2-12-0

# Install WaffleIron
git clone https://github.com/valeoai/WaffleIron
cd WaffleIron && pip install -e ./
```

---

## Dataset

### Waymo Open Dataset v1.4.3
- **Train:** 798 segments, 23,691 labeled frames
- **Validation:** 202 segments, 5,976 labeled frames
- **Test:** 150 segments, 0 labeled frames (hidden ground truth — requires leaderboard submission)
- **Labeling frequency:** 2 Hz (~30 labeled frames per 20-second segment at 10 Hz capture rate)
- **LiDAR sensors:** 1 mid-range TOP lidar (64 beams, 75m range) + 4 short-range lidars (20m range)
- **Points per frame:** ~100K–200K from all 5 lidars combined

### 22 Waymo Semantic Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Undefined (ignored) | 12 | Bicycle |
| 1 | Car | 13 | Motorcycle |
| 2 | Truck | 14 | Building |
| 3 | Bus | 15 | Vegetation |
| 4 | Other Vehicle | 16 | Tree Trunk |
| 5 | Motorcyclist | 17 | Curb |
| 6 | Bicyclist | 18 | Road |
| 7 | Pedestrian | 19 | Lane Marker |
| 8 | Sign | 20 | Other Ground |
| 9 | Traffic Light | 21 | Walkable |
| 10 | Pole | 22 | Sidewalk |
| 11 | Construction Cone | | |

---

## Data Conversion Pipeline

WaffleIron expects data in SemanticKITTI format. We convert Waymo TFRecords to `.bin` (point clouds) and `.label` (per-point semantic labels) files.

### Why convert?
WaffleIron's data loading code reads SemanticKITTI-format files: `.bin` files containing `[x, y, z, intensity]` as float32, and `.label` files containing uint32 class IDs. Waymo stores data in TFRecord protobufs with compressed range images. The conversion extracts 3D point clouds and segmentation labels from all 5 LiDAR sensors.

### Conversion script: `convert_waymo_to_kitti.py`

```bash
# Convert training set (798 segments, ~6 hours)
python convert_waymo_to_kitti.py \
    --input_dir ~/data/waymo/train/ \
    --output_dir ~/data/waymo_kitti/train/ \
    --skip_scan

# Convert validation set (202 segments, ~1.5 hours)
python convert_waymo_to_kitti.py \
    --input_dir ~/data/waymo/val/ \
    --output_dir ~/data/waymo_kitti/val/ \
    --skip_scan
```

### Key details of conversion:
- **All 5 lidars** are included (TOP + FRONT + SIDE_LEFT + SIDE_RIGHT + REAR)
- **Only labeled frames** are converted (segmentation labels exist at 2 Hz, not every frame)
- **Raw Waymo class IDs (0–22) are preserved** — no remapping to SemanticKITTI classes. The `remap_waymo_to_kitti()` function exists in the script but is intentionally disabled (`#labels = remap_waymo_to_kitti(labels)`)
- **Segmentation labels** are decompressed from ZLIB-compressed `MatrixInt32` protobufs stored in `laser.ri_return1.segmentation_label_compressed`
- **Instance IDs** from Waymo are discarded — only semantic class IDs are saved
- Output structure follows SemanticKITTI format:
  ```
  output_dir/sequences/XXXX/velodyne/NNNNNN.bin   # [N, 4] float32: x, y, z, intensity
  output_dir/sequences/XXXX/labels/NNNNNN.label    # [N] uint32: semantic class ID
  ```

### Combined directory structure
WaffleIron reads from a single root directory. We symlink train and val into a combined directory with offset numbering:
```bash
# Train sequences: 0–797  (original numbering)
# Val sequences:   800–1001 (offset by +800 to avoid collision)

mkdir -p ~/data/waymo_kitti/combined/dataset/sequences

# Symlink training
for d in ~/data/waymo_kitti/train/dataset/sequences/*; do
    ln -s $d ~/data/waymo_kitti/combined/dataset/sequences/$(basename $d)
done

# Symlink validation with +800 offset
for d in ~/data/waymo_kitti/val/dataset/sequences/*; do
    name=$(basename $d)
    num=$((10#$name + 800))
    ln -s $d ~/data/waymo_kitti/combined/dataset/sequences/$num
done
```

---

## WaffleIron Modifications for Waymo

We modified or created the following files from the original WaffleIron codebase. All changes are minimal and justified.

### 1. `datasets/waymo_dataset.py` (NEW)
Copy of `semantic_kitti.py` with three changes:
- **Class name:** `WaymoDataset` instead of `SemanticKITTI`
- **CLASS_NAME list:** 22 Waymo class names (used for training log display)
- **YAML reference:** Reads `waymo.yaml` instead of `semantic-kitti.yaml`

**InstanceCutMix is disabled** (`instance_cutmix: False`) because:
1. Our conversion only saves semantic class IDs, not instance IDs needed for CutMix
2. The hardcoded instance counts in `test_loaded()` are SemanticKITTI-specific

### 2. `datasets/waymo.yaml` (NEW)
Defines Waymo's 22+1 classes and data splits:
- **`labels`**: Maps IDs 0–22 to human-readable names
- **`learning_map`**: Identity mapping (0→0, 1→1, ..., 22→22). Unlike SemanticKITTI which has non-sequential raw IDs (10, 40, 50...) requiring remapping, Waymo's IDs are already sequential
- **`learning_map_inv`**: Reverse identity mapping for converting predictions back to raw IDs
- **`split`**: Train = sequences 0–797, Valid = sequences 800–1001

### 3. `datasets/__init__.py` (MODIFIED)
Added one import and one dictionary entry:
```python
from .waymo_dataset import WaymoDataset
LIST_DATASETS = {..., "waymo": WaymoDataset}
```

### 4. `configs/WaffleIron-48-256__waymo.yaml` (NEW)
Copy of the SemanticKITTI config with these changes:
- `nb_class: 22` (was 19)
- `fov_xyz: [[-50,-50,-4], [50,50,6]]` — field of view in meters
- `grids_size: [[250,250], [250,24], [250,24]]` — 2D grid resolution
- `instance_cutmix: False`
- `max_epoch: 60`

**FOV justification:**
- **XY ±50m:** Captures 96.5% of all Waymo LiDAR points (TOP lidar max range is 75m, but only 3.5% of points fall between 50–75m). The WaffleIron paper shows grid resolution ρ is robust between 20–80cm, so keeping [250,250] grids at 50m FOV gives ρ=0.40m.
- **Z [-4, +6]:** Captures 100% of points. The old Z=[-3,+2] lost 23.1% of points (48% of lost points were Building tops, 36% were Vegetation canopy).
- **Grid Z=24:** Maintains resolution: 10m ÷ 24 = 0.42m per cell (same as original 5m ÷ 12 = 0.42m).

### 5. `utils/trainer.py` (MODIFIED)
Added gradient clipping for FP16 training stability (lines 226–227):
```python
self.scaler.unscale_(self.optim)
torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
```
**Reason:** Training hit NaN at epoch 10 without gradient clipping due to FP16 overflow. Gradient clipping prevents this while maintaining FP16 speed benefits.

### 6. `finetune_waymo.py` (NEW)
Creates a checkpoint that loads pretrained SemanticKITTI weights into the 22-class Waymo model:
- Loads all 981 parameters where shapes match
- Skips 2 parameters (classification layer weight and bias) due to shape mismatch (19→22 classes)
- Saves as `ckpt_last.pth` that `launch_train.py --restart` can load

### 7. `eval_waymo.py` (NEW)
Runs inference on the validation set and saves predicted `.label` files:
- Uses `WaymoDataset` loader and `waymo.yaml` for label remapping
- Adds +1 shift to predictions (model outputs 0–21 → raw IDs 1–22)
- Supports test-time augmentation via `--num_votes`

---

## Label Mapping Pipeline

The complete label flow from disk to model and back:

```
On disk (.label):     Raw Waymo ID (0–22)
                          ↓
learning_map:         Identity (0→0, 1→1, ..., 22→22)
                          ↓
In waymo_dataset.py:  labels = labels - 1
                      labels[labels == -1] = 255
                          ↓
Model input:          0=Car, 1=Truck, ..., 21=Sidewalk
                      255=Undefined (ignored in loss)
                          ↓
Model output:         22 logits → argmax → 0–21
                          ↓
eval_waymo.py:        predictions + 1 → 1–22 (back to raw Waymo IDs)
                          ↓
Saved .label:         Raw Waymo ID (1–22)
```

**Why identity mapping?** Waymo's raw IDs (0–22) are already sequential. SemanticKITTI uses non-sequential IDs (10, 11, 13, 15, 18, 30, 31, 40...) requiring a lookup table. We keep the `learning_map` mechanism for compatibility with WaffleIron's codebase but set it to identity.

**Why subtract 1?** The model's output layer has neurons 0–21. Class 0 (Undefined) is mapped to 255 (PyTorch's standard ignore index for cross-entropy loss). So raw class 1 (Car) → model class 0, raw class 22 (Sidewalk) → model class 21.

---

## Training

### Step 1: Create fine-tune checkpoint
```bash
python finetune_waymo.py \
    --pretrained pretrained_models/WaffleIron-48-256__kitti/ckpt_last.pth \
    --config configs/WaffleIron-48-256__waymo.yaml \
    --output_dir waymo_finetune/
```
Output: `Loaded 981 parameters, skipped 2`

### Step 2: Train
```bash
python launch_train.py \
    --dataset waymo \
    --path_dataset ~/data/waymo_kitti/combined/ \
    --log_path ./waymo_finetune/ \
    --config ./configs/WaffleIron-48-256__waymo.yaml \
    --fp16 \
    --restart
```

### Training history
Multiple training runs were needed due to NaN issues:

| Run | Config | Issue | Best mIoU |
|-----|--------|-------|-----------|
| 1 | FP16, batch 4, Z=[-3,2] | NaN at epoch 10 | 53.0% |
| 2 | FP16, LR 0.0005 | NaN persisted | — |
| 3 | FP32, batch 2 | Stuck at 44% (scheduler mismatch) | 44% |
| **4 (final)** | **FP16 + gradient clipping, batch 4, Z=[-4,6]** | **Stable** | **55.3%** |

The final run added gradient clipping (`clip_grad_norm_ max_norm=1.0`) to `trainer.py` and fixed the Z range from [-3,2] to [-4,6], which recovered 23.1% of previously lost points.

---

## Evaluation and Inference

### Run inference on validation set
```bash
python eval_waymo.py \
    --config configs/WaffleIron-48-256__waymo.yaml \
    --ckpt waymo_finetune/ckpt_best.pth \
    --path_dataset ~/data/waymo_kitti/combined/ \
    --result_folder ~/val_predictions/ \
    --phase val --batch_size 1
```

### Analyze predictions
```bash
python analyze_predictions.py
```
Generates: per-class IoU, confusion matrix CSV, distance-based accuracy, top confusion pairs.

### Merge predictions into single files
```bash
python merge_predictions.py
```
Output: `.bin` files with format `[x, y, z, intensity, predicted_class]` as float32.

---

## Results

### Per-Class IoU (55.3% model, validation set)

| Class | IoU | GT Points | Pred Points |
|-------|-----|-----------|-------------|
| Car | 39.6% | 69.5M | 165.4M |
| Truck | 44.7% | 7.8M | 7.1M |
| Bus | 53.4% | 3.3M | 1.8M |
| Other Vehicle | 26.5% | 1.8M | 1.2M |
| Motorcyclist | 0.0% | 503 | 3.2K |
| Bicyclist | 57.2% | 248K | 189K |
| Pedestrian | 71.5% | 5.9M | 4.8M |
| Sign | 55.4% | 4.9M | 4.1M |
| Traffic Light | 5.4% | 503K | 62K |
| Pole | 60.0% | 8.6M | 7.7M |
| Construction Cone | 55.1% | 488K | 385K |
| Bicycle | 49.2% | 172K | 102K |
| Motorcycle | 67.1% | 197K | 167K |
| Building | 81.5% | 244M | 215M |
| Vegetation | 76.3% | 160M | 138M |
| Tree Trunk | 53.9% | 13.6M | 10.4M |
| Curb | 57.3% | 11.3M | 10.7M |
| Road | 79.2% | 191M | 175M |
| Lane Marker | 29.4% | 5.4M | 2.4M |
| Other Ground | 11.2% | 4.2M | 747K |
| Walkable | 65.6% | 74.5M | 71.4M |
| Sidewalk | 63.1% | 45.9M | 36.0M |

**Overall:** mIoU 50.1%, oAcc 81.3%

### Known issue: Car over-prediction
Car is predicted 2.4× more than ground truth (165M vs 69M points). Building, Road, Vegetation, Walkable, and Sidewalk each lose 11–14% of their points to Car misclassification. This is the primary factor limiting mIoU.

### Accuracy by distance

| Distance | Accuracy | Points |
|----------|----------|--------|
| 0–10m | 84.2% | 222M |
| 10–20m | 83.7% | 334M |
| 20–30m | 83.0% | 133M |
| 30–40m | 81.1% | 72M |
| 40–50m+ | 79.1% | 42M |

---

## Metadata and Weather Classification

### Waymo native metadata
Extracted from TFRecords via `get_metadata_v2.py`:
- **Weather:** Sunny (1144), Rain (6) — Waymo provides only binary weather labels
- **Time of day:** Day (912), Night (125), Dawn/Dusk (113)
- **Location:** San Francisco (610), Phoenix (402), Other (138)
- **Timestamps:** Converted to local time using pytz

### CLIP weather classification
Zero-shot CLIP ViT-L/14 with driving-specific prompts (`classify_weather_clip.py`):
- Clear/Sunny: 402 (35%), Dawn/Dusk: 283 (25%), Overcast: 217 (19%)
- Foggy: 121 (10.5%), Nighttime: 98 (8.5%), Rainy: 29 (2.5%)
- Validated against Waymo's native labels: 78% Night segments → "nighttime", 81% Dawn/Dusk → "dawn/dusk"

### LiDAR weather detection
`detect_weather_lidar.py` computes per-frame weather scores from point cloud statistics:
- Mean/median intensity, low-intensity point ratio
- Close-range point density (fog/rain noise clusters near sensor)
- Effective range reduction (fog truncates max range)
- Composite weather score (0–8) flagging suspicious frames

---

## File Reference

### Data pipeline
| File | Purpose |
|------|---------|
| `convert_waymo_to_kitti.py` | Convert Waymo TFRecords → SemanticKITTI format |
| `get_metadata.py` | Extract segment metadata from TFRecords |
| `get_metadata_v2.py` | V2 with timezone-corrected local timestamps |
| `classify_weather_clip.py` | CLIP zero-shot weather classification |
| `detect_weather_lidar.py` | LiDAR-based weather detection |
| `extract_val_videos.py` | Extract front camera MP4 videos from TFRecords |

### WaffleIron Waymo files
| File | Status | Purpose |
|------|--------|---------|
| `waffleiron/datasets/waymo_dataset.py` | NEW | Waymo dataset loader |
| `waffleiron/datasets/waymo.yaml` | NEW | Class definitions, label map, splits |
| `waffleiron/datasets/__init__.py` | MODIFIED | Register WaymoDataset |
| `waffleiron/configs/WaffleIron-48-256__waymo.yaml` | NEW | Waymo training config |
| `waffleiron/utils/trainer.py` | MODIFIED | Added gradient clipping for FP16 |
| `waffleiron/finetune_waymo.py` | NEW | Create fine-tune checkpoint (19→22 classes) |
| `waffleiron/eval_waymo.py` | NEW | Run inference, save .label predictions |

### Analysis and evaluation
| File | Purpose |
|------|---------|
| `analyze_predictions.py` | Per-class IoU, confusion matrix, distance accuracy |
| `merge_predictions.py` | Merge .bin + .label → single [x,y,z,int,class] files |

### Saved results (not in repo)
| Path | Contents |
|------|----------|
| `results_50m_z6/ckpt_best.pth` | Best model weights (55.3% mIoU, epoch 59) |
| `results_50m_z6/config_50m.yaml` | Matching config for this checkpoint |
| `results_50m_z6/confusion_matrix.csv` | 22×22 confusion matrix |
| `results_50m_z6/per_class_results.csv` | Per-class IoU with point counts |
| `results_50m_z6/val_predictions/` | Predicted .label files for all 5,976 val frames |
| `val_predictions_merged/` | Merged [x,y,z,intensity,class] .bin files |
| `logs/` | All training logs |

---

## Acknowledgements

- [WaffleIron](https://github.com/valeoai/WaffleIron) — Puy, Boulch, Marlet (ICCV 2023) — Apache 2.0 License
- [Waymo Open Dataset](https://waymo.com/open/) v1.4.3 — Non-commercial license
- [OpenAI CLIP](https://github.com/openai/CLIP) — Weather classification
