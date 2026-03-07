# AWARE - Autonomous Vehicle Waymo Segmentation with WaffleIron

Fine-tuning WaffleIron 3D point cloud semantic segmentation model on Waymo Open Dataset.

## Setup

### Prerequisites
- Python 3.10
- PyTorch 2.2.2 with CUDA 12.1
- TensorFlow 2.13.0 (for data conversion)
- waymo-open-dataset-tf-2-12-0

### Installation
```bash
conda create -n waffleiron python=3.10 -y
conda activate waffleiron
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install pyaml tqdm scipy tensorboard protobuf==3.20.3
pip install tensorflow==2.13.0 waymo-open-dataset-tf-2-12-0
git clone https://github.com/valeoai/WaffleIron
cd WaffleIron && pip install -e ./
```

## Data Conversion
Convert Waymo v1.4.3 TFRecords to SemanticKITTI format:
```bash
python convert_waymo_to_kitti.py \
    --input_dir /path/to/waymo/train/ \
    --output_dir /path/to/waymo_kitti/train/ \
    --skip_scan
```

## Training
```bash
python launch_train.py \
    --dataset waymo \
    --path_dataset /path/to/waymo_kitti/train/ \
    --log_path ./waymo_finetune/ \
    --config ./configs/WaffleIron-48-256__waymo.yaml \
    --fp16
```

## Waymo Classes (22)
Car, Truck, Bus, Other Vehicle, Motorcyclist, Bicyclist, Pedestrian, Sign, Traffic Light, Pole, Construction Cone, Bicycle, Motorcycle, Building, Vegetation, Tree Trunk, Curb, Road, Lane Marker, Other Ground, Walkable, Sidewalk

## Based On
- [WaffleIron](https://github.com/valeoai/WaffleIron) (ICCV 2023)
- [Waymo Open Dataset](https://waymo.com/open/) v1.4.3
