#!/usr/bin/env python3
"""
Extract metadata from Waymo TFRecords - v2 with local timestamps.
"""

import os
import csv
import datetime
import pytz
from glob import glob
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

TIMEZONE_MAP = {
    'location_sf': pytz.timezone('US/Pacific'),
    'location_phx': pytz.timezone('US/Arizona'),
    'location_other': pytz.timezone('US/Pacific'),
}

def get_segment_metadata(tfrecord_path, split):
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    segment_name = None
    weather = None
    location = None
    time_of_day = None
    total_frames = 0
    labeled_frames = 0
    first_timestamp = None
    
    for raw_record in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(raw_record.numpy()))
        total_frames += 1
        
        if segment_name is None:
            segment_name = frame.context.name
            weather = frame.context.stats.weather
            location = frame.context.stats.location
            time_of_day = frame.context.stats.time_of_day
            first_timestamp = frame.timestamp_micros
        
        has_semseg = False
        for laser in frame.lasers:
            if laser.name == 1:
                if len(laser.ri_return1.segmentation_label_compressed) > 0:
                    has_semseg = True
                    break
        if has_semseg:
            labeled_frames += 1
    
    # Convert to local time
    dt_utc = datetime.datetime.fromtimestamp(first_timestamp / 1e6, tz=pytz.utc)
    local_tz = TIMEZONE_MAP.get(location, pytz.timezone('US/Pacific'))
    dt_local = dt_utc.astimezone(local_tz)
    
    return {
        'split': split,
        'segment_name': segment_name,
        'weather': weather,
        'location': location,
        'time_of_day': time_of_day,
        'total_frames': total_frames,
        'labeled_frames': labeled_frames,
        'date': dt_local.strftime('%Y-%m-%d'),
        'day_of_week': dt_local.strftime('%A'),
        'hour_of_day': dt_local.hour,
        'local_time': dt_local.strftime('%H:%M:%S'),
        'timezone': str(local_tz),
        'tfrecord': os.path.basename(tfrecord_path),
    }


def main():
    train_dir = os.path.expanduser('~/AWARE/data/waymo/train/')
    val_dir = os.path.expanduser('~/AWARE/data/waymo/val/')
    test_dir = os.path.expanduser('~/AWARE/data/waymo/test/')
    
    train_files = sorted(glob(os.path.join(train_dir, '*.tfrecord')))
    val_files = sorted(glob(os.path.join(val_dir, '*.tfrecord')))
    test_files = sorted(glob(os.path.join(test_dir, '*.tfrecord')))
    
    print(f"Found {len(train_files)} train TFRecords")
    print(f"Found {len(val_files)} val TFRecords")
    print(f"Found {len(test_files)} test TFRecords")
    print(f"Total: {len(train_files) + len(val_files) + len(test_files)} TFRecords\n")
    
    output_csv = os.path.expanduser('~/AWARE/logs/waymo_metadata_v2.csv')
    
    fieldnames = [
        'split', 'segment_name', 'weather', 'location', 'time_of_day',
        'total_frames', 'labeled_frames', 'date', 'day_of_week',
        'hour_of_day', 'local_time', 'timezone', 'tfrecord'
    ]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        weather_counts = {}
        location_counts = {}
        time_counts = {}
        day_counts = {}
        hour_counts = {}
        total_labeled = {'train': 0, 'val': 0, 'test': 0}
        
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            if not files:
                print(f"No {split} files found, skipping.")
                continue
            print(f"Processing {split} TFRecords...")
            for f in tqdm(files):
                meta = get_segment_metadata(f, split)
                writer.writerow(meta)
                csvfile.flush()
                
                total_labeled[split] += meta['labeled_frames']
                weather_counts[meta['weather']] = weather_counts.get(meta['weather'], 0) + 1
                location_counts[meta['location']] = location_counts.get(meta['location'], 0) + 1
                time_counts[meta['time_of_day']] = time_counts.get(meta['time_of_day'], 0) + 1
                day_counts[meta['day_of_week']] = day_counts.get(meta['day_of_week'], 0) + 1
                hour_counts[meta['hour_of_day']] = hour_counts.get(meta['hour_of_day'], 0) + 1
    
    print("\n" + "=" * 60)
    print("WAYMO DATASET METADATA SUMMARY")
    print("=" * 60)
    print(f"\nTFRecords: {len(train_files)} train + {len(val_files)} val + {len(test_files)} test = {len(train_files) + len(val_files) + len(test_files)} total")
    for split in ['train', 'val', 'test']:
        print(f"  {split} labeled frames: {total_labeled[split]}")
    print(f"  Total labeled frames: {sum(total_labeled.values())}")
    
    print(f"\nWeather distribution:")
    for k, v in sorted(weather_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} segments")
    
    print(f"\nLocation distribution:")
    for k, v in sorted(location_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} segments")
    
    print(f"\nTime of day distribution:")
    for k, v in sorted(time_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v} segments")
    
    print(f"\nDay of week distribution (local time):")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for d in days_order:
        if d in day_counts:
            print(f"  {d}: {day_counts[d]} segments")
    
    print(f"\nHour of day distribution (local time):")
    for h in sorted(hour_counts.keys()):
        print(f"  {h:02d}:00 - {h:02d}:59: {hour_counts[h]} segments")
    
    print(f"\nFull metadata saved to: {output_csv}")


if __name__ == '__main__':
    main()
