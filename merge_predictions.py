#!/usr/bin/env python3
"""
Merge .bin (x,y,z,intensity) + predicted .label (class) into single .bin files.
Output format: float32 array [x, y, z, intensity, predicted_class] per point.
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm


def main():
    pred_dir = os.path.expanduser('~/AWARE/val_predictions/')
    data_dir = os.path.expanduser('~/AWARE/data/waymo_kitti/combined/dataset/')
    output_dir = os.path.expanduser('~/AWARE/val_predictions_merged/')
    
    pred_files = sorted(glob(os.path.join(pred_dir, 'sequences/*/predictions/*.label')))
    print(f"Found {len(pred_files)} prediction files")
    
    merged_count = 0
    for pred_f in tqdm(pred_files, desc="Merging"):
        # Get corresponding .bin file
        rel = pred_f[len(pred_dir):]  # sequences/800/predictions/000000.label
        bin_f = os.path.join(data_dir, rel.replace('predictions', 'velodyne').replace('.label', '.bin'))
        
        if not os.path.exists(bin_f):
            continue
        
        # Load point cloud and predictions
        pc = np.fromfile(bin_f, dtype=np.float32).reshape(-1, 4)  # x, y, z, intensity
        pred = np.fromfile(pred_f, dtype=np.uint32)
        
        # Match lengths
        min_len = min(len(pc), len(pred))
        pc = pc[:min_len]
        pred = pred[:min_len]
        
        # Merge: [x, y, z, intensity, class] as float32
        merged = np.column_stack([pc, pred.astype(np.float32)])  # [N, 5]
        
        # Save
        out_path = os.path.join(output_dir, rel.replace('predictions', 'merged').replace('.label', '.bin'))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        merged.astype(np.float32).tofile(out_path)
        merged_count += 1
    
    print(f"\nDone! Merged {merged_count} files to {output_dir}")
    
    # Verify one file
    sample = sorted(glob(os.path.join(output_dir, 'sequences/*/merged/*.bin')))[0]
    data = np.fromfile(sample, dtype=np.float32).reshape(-1, 5)
    print(f"\nVerification: {sample}")
    print(f"  Shape: {data.shape}")
    print(f"  Columns: [x, y, z, intensity, predicted_class]")
    print(f"  First 3 points:")
    for i in range(3):
        print(f"    x={data[i,0]:.2f}, y={data[i,1]:.2f}, z={data[i,2]:.2f}, "
              f"int={data[i,3]:.2f}, class={int(data[i,4])}")
    print(f"  Unique classes: {sorted(np.unique(data[:, 4].astype(int)).tolist())}")


if __name__ == '__main__':
    main()
