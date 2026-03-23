#!/usr/bin/env python3
"""
Extract metadata from Waymo TFRecords used for WaffleIron finetuning.
Outputs: segment name, weather, location, total frames, labeled frames count.
"""

import os
import sys
import csv
import zlib
import numpy as np
from glob import glob
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import segmentation_pb2

def get_segment_metadata(tfrecord_path):
    """Extract metadata from a single TFRecord file."""
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    segment_name = None
    weather = None
    location = None
    time_of_day = None
    total_frames = 0
    labeled_frames = 0
    
    for raw_record in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(raw_record.numpy()))
        total_frames += 1
        
        # Get metadata from first frame
        if segment_name is None:
            segment_name = frame.context.name
            weather = frame.context.stats.weather
            location = frame.context.stats.location
            time_of_day = frame.context.stats.time_of_day
        
        # Check if this frame has lidar segmentation labels
        has_semseg = False
        for laser in frame.lasers:
            if laser.name == 1:  # TOP lidar
                if len(laser.ri_return1.segmentation_label_compressed) > 0:
                    has_semseg = True
                    break
        
        if has_semseg:
            labeled_frames += 1
    
    return {
        'segment_name': segment_name,
        'weather': weather,
        'location': location,
        'time_of_day': time_of_day,
        'total_frames': total_frames,
        'labeled_frames': labeled_frames,
        'tfrecord': os.path.basename(tfrecord_path),
    }


def main():
    train_dir = os.path.expanduser('~/AWARE/data/waymo/train/')
    val_dir = os.path.expanduser('~/AWARE/data/waymo/val/')
    
    train_files = sorted(glob(os.path.join(train_dir, '*.tfrecord')))
    val_files = sorted(glob(os.path.join(val_dir, '*.tfrecord')))
    
    print(f"Found {len(train_files)} train TFRecords")
    print(f"Found {len(val_files)} val TFRecords")
    print(f"Total: {len(train_files) + len(val_files)} TFRecords\n")
    
    output_csv = os.path.expanduser('~/AWARE/logs/waymo_metadata.csv')
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'split', 'segment_name', 'weather', 'location', 'time_of_day',
            'total_frames', 'labeled_frames', 'tfrecord'
        ])
        writer.writeheader()
        
        # Weather/location/time counters
        weather_counts = {}
        location_counts = {}
        time_counts = {}
        total_labeled_train = 0
        total_labeled_val = 0
        
        # Process training files
        print("Processing training TFRecords...")
        for f in tqdm(train_files):
            meta = get_segment_metadata(f)
            meta['split'] = 'train'
            writer.writerow(meta)
            csvfile.flush()
            
            total_labeled_train += meta['labeled_frames']
            weather_counts[meta['weather']] = weather_counts.get(meta['weather'], 0) + 1
            location_counts[meta['location']] = location_counts.get(meta['location'], 0) + 1
            time_counts[meta['time_of_day']] = time_counts.get(meta['time_of_day'], 0) + 1
        
        # Process validation files
        print("\nProcessing validation TFRecords...")
        for f in tqdm(val_files):
            meta = get_segment_metadata(f)
            meta['split'] = 'val'
            writer.writerow(meta)
            csvfile.flush()
            
            total_labeled_val += meta['labeled_frames']
            weather_counts[meta['weather']] = weather_counts.get(meta['weather'], 0) + 1
            location_counts[meta['location']] = location_counts.get(meta['location'], 0) + 1
            time_counts[meta['time_of_day']] = time_counts.get(meta['time_of_day'], 0) + 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("WAYMO FINETUNING DATA SUMMARY")
    print("=" * 60)
    print(f"\nTFRecords:  {len(train_files)} train + {len(val_files)} val = {len(train_files) + len(val_files)} total")
    print(f"Labeled frames:  {total_labeled_train} train + {total_labeled_val} val = {total_labeled_train + total_labeled_val} total")
    print(f"\nWeather distribution:")
    for k, v in sorted(weather_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} segments")
    print(f"\nLocation distribution:")
    for k, v in sorted(location_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} segments")
    print(f"\nTime of day distribution:")
    for k, v in sorted(time_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} segments")
    print(f"\nFull metadata saved to: {output_csv}")


if __name__ == '__main__':
    main()
