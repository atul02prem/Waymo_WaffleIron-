#!/usr/bin/env python3
"""
Classify weather in Waymo camera images using OpenAI CLIP zero-shot.

WHY CLIP IS BETTER FOR THIS TASK:
- We define our own text descriptions (prompts)
- We can distinguish "dark nighttime road" from "rainy road"
- We can be specific: "sunny clear sky driving" vs "overcast cloudy driving"
- No training needed — CLIP was trained on 400M image-text pairs from the internet

HOW CLIP ZERO-SHOT WORKS:
1. Encode the image into a 512-dim vector
2. Encode each text prompt into a 512-dim vector
3. Compute cosine similarity between image and each text
4. Highest similarity = best matching description = predicted weather
"""

import os
import io
import csv
import datetime
import pytz
import numpy as np
from glob import glob
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from waymo_open_dataset import dataset_pb2 as open_dataset

# ============================================================
# LOAD CLIP MODEL
# ============================================================
print("Loading CLIP model from Hugging Face...")
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cpu")
model = model.to(device)
print(f"CLIP loaded on {device}")

# ============================================================
# DEFINE WEATHER PROMPTS
# ============================================================
# These are carefully crafted to distinguish driving conditions.
# The key insight: we separate LIGHTING from WEATHER.
# This prevents the night=rain confusion from the previous model.

WEATHER_PROMPTS = [
    "a front-camera photo from a self driving car on a clear sunny day with blue sky and dry road",
    "a front-camera photo from a self driving car on an overcast cloudy day with gray sky",
    "a front-camera photo from a self driving car driving through fog or haze with poor visibility",
    "a front-camera photo from a self driving car driving in rain with wet road and water on windshield",
    "a front-camera photo from a self driving car driving at nighttime on a dark road with headlights and street lights",
    "a front-camera photo from a self driving car driving at dawn or dusk with orange sky and setting sun",
]

WEATHER_LABELS = [
    "clear/sunny",
    "overcast/cloudy",
    "foggy/hazy",
    "rainy/wet",
    "nighttime",
    "dawn/dusk",
]

TIMEZONE_MAP = {
    'location_sf': pytz.timezone('US/Pacific'),
    'location_phx': pytz.timezone('US/Arizona'),
    'location_other': pytz.timezone('US/Pacific'),
}


def classify_image(image_bytes):
    """Classify weather using CLIP zero-shot."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    inputs = processor(
        text=WEATHER_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # CLIP outputs logits_per_image: similarity between image and each text
    logits = outputs.logits_per_image.squeeze()
    probs = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()
    
    pred_idx = int(np.argmax(probs))
    pred_label = WEATHER_LABELS[pred_idx]
    pred_conf = float(probs[pred_idx])
    
    all_probs = {WEATHER_LABELS[i]: round(float(probs[i]), 4) for i in range(len(probs))}
    
    return pred_label, pred_conf, all_probs


def process_segment(tfrecord_path, split):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    segment_name = None
    weather_waymo = None
    location = None
    time_of_day = None
    total_frames = 0
    labeled_frames = 0
    weather_pred = None
    weather_conf = None
    weather_all = None
    
    for raw_record in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(raw_record.numpy()))
        total_frames += 1
        
        if segment_name is None:
            segment_name = frame.context.name
            weather_waymo = frame.context.stats.weather
            location = frame.context.stats.location
            time_of_day = frame.context.stats.time_of_day
            first_timestamp = frame.timestamp_micros
            
            for img in frame.images:
                if img.name == 1:  # FRONT camera
                    weather_pred, weather_conf, weather_all = classify_image(img.image)
                    break
        
        has_semseg = False
        for laser in frame.lasers:
            if laser.name == 1:
                if len(laser.ri_return1.segmentation_label_compressed) > 0:
                    has_semseg = True
                    break
        if has_semseg:
            labeled_frames += 1
    
    dt_utc = datetime.datetime.fromtimestamp(first_timestamp / 1e6, tz=pytz.utc)
    local_tz = TIMEZONE_MAP.get(location, pytz.timezone('US/Pacific'))
    dt_local = dt_utc.astimezone(local_tz)
    
    return {
        'split': split,
        'segment_name': segment_name,
        'waymo_weather': weather_waymo,
        'waymo_time_of_day': time_of_day,
        'clip_weather': weather_pred,
        'clip_confidence': round(weather_conf, 4) if weather_conf else None,
        'prob_clear_sunny': weather_all.get('clear/sunny', 0) if weather_all else None,
        'prob_overcast': weather_all.get('overcast/cloudy', 0) if weather_all else None,
        'prob_foggy': weather_all.get('foggy/hazy', 0) if weather_all else None,
        'prob_rainy': weather_all.get('rainy/wet', 0) if weather_all else None,
        'prob_nighttime': weather_all.get('nighttime', 0) if weather_all else None,
        'prob_dawn_dusk': weather_all.get('dawn/dusk', 0) if weather_all else None,
        'location': location,
        'total_frames': total_frames,
        'labeled_frames': labeled_frames,
        'date': dt_local.strftime('%Y-%m-%d'),
        'day_of_week': dt_local.strftime('%A'),
        'hour_of_day': dt_local.hour,
        'local_time': dt_local.strftime('%H:%M:%S'),
        'tfrecord': os.path.basename(tfrecord_path),
    }


def main():
    train_dir = os.path.expanduser('~/AWARE/data/waymo/train/')
    val_dir = os.path.expanduser('~/AWARE/data/waymo/val/')
    test_dir = os.path.expanduser('~/AWARE/data/waymo/test/')
    
    train_files = sorted(glob(os.path.join(train_dir, '*.tfrecord')))
    val_files = sorted(glob(os.path.join(val_dir, '*.tfrecord')))
    test_files = sorted(glob(os.path.join(test_dir, '*.tfrecord')))
    
    print(f"Found {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    output_csv = os.path.expanduser('~/AWARE/logs/waymo_clip_weather.csv')
    
    fieldnames = [
        'split', 'segment_name', 'waymo_weather', 'waymo_time_of_day',
        'clip_weather', 'clip_confidence',
        'prob_clear_sunny', 'prob_overcast', 'prob_foggy',
        'prob_rainy', 'prob_nighttime', 'prob_dawn_dusk',
        'location', 'total_frames', 'labeled_frames',
        'date', 'day_of_week', 'hour_of_day', 'local_time', 'tfrecord'
    ]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            if not files:
                continue
            print(f"\nProcessing {split} ({len(files)} segments)...")
            for f in tqdm(files):
                meta = process_segment(f, split)
                writer.writerow(meta)
                csvfile.flush()
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLIP WEATHER CLASSIFICATION SUMMARY")
    print("=" * 60)
    
    with open(output_csv) as f:
        rows = list(csv.DictReader(f))
    
    from collections import Counter
    
    clip_counts = Counter(r['clip_weather'] for r in rows)
    print(f"\nCLIP weather distribution:")
    for k, v in sorted(clip_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} segments ({100*v/len(rows):.1f}%)")
    
    # CLIP vs Waymo time_of_day agreement for lighting
    print(f"\nCLIP lighting vs Waymo time_of_day:")
    for tod in ['Day', 'Night', 'Dawn/Dusk']:
        subset = [r for r in rows if r['waymo_time_of_day'] == tod]
        clip = Counter(r['clip_weather'] for r in subset)
        print(f"  Waymo {tod}: {dict(sorted(clip.items(), key=lambda x: -x[1]))}")
    
    # CLIP weather by location
    print(f"\nCLIP weather by location:")
    for loc in ['location_sf', 'location_phx', 'location_other']:
        subset = [r for r in rows if r['location'] == loc]
        clip = Counter(r['clip_weather'] for r in subset)
        print(f"  {loc}: {dict(sorted(clip.items(), key=lambda x: -x[1]))}")
    
    # Waymo sunny reclassified
    sunny = [r for r in rows if r['waymo_weather'] == 'sunny']
    reclass = Counter(r['clip_weather'] for r in sunny)
    print(f"\nWaymo 'sunny' reclassified by CLIP:")
    for k, v in sorted(reclass.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} ({100*v/len(sunny):.1f}%)")
    
    print(f"\nResults saved to: {output_csv}")


if __name__ == '__main__':
    main()
