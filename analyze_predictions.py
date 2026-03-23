#!/usr/bin/env python3
"""
Analyze WaffleIron predictions vs ground truth on Waymo val set.
Generates: per-class IoU, confusion matrix, per-distance accuracy, class statistics.
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import Counter
import csv

CLASS_NAMES = {
    0: 'Undefined', 1: 'Car', 2: 'Truck', 3: 'Bus', 4: 'Other Vehicle',
    5: 'Motorcyclist', 6: 'Bicyclist', 7: 'Pedestrian', 8: 'Sign',
    9: 'Traffic Light', 10: 'Pole', 11: 'Construction Cone', 12: 'Bicycle',
    13: 'Motorcycle', 14: 'Building', 15: 'Vegetation', 16: 'Tree Trunk',
    17: 'Curb', 18: 'Road', 19: 'Lane Marker', 20: 'Other Ground',
    21: 'Walkable', 22: 'Sidewalk'
}

NUM_CLASSES = 23  # 0-22

def compute_confusion_matrix(pred_files, gt_files, bin_files):
    """Compute confusion matrix and per-distance stats."""
    
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    # Per-distance stats
    dist_bins = [0, 10, 20, 30, 40, 50]
    dist_correct = {d: 0 for d in dist_bins}
    dist_total = {d: 0 for d in dist_bins}
    
    # Per-class point counts
    gt_counts = Counter()
    pred_counts = Counter()
    
    for pred_f, gt_f, bin_f in tqdm(zip(pred_files, gt_files, bin_files), 
                                      total=len(pred_files), desc="Analyzing"):
        pred = np.fromfile(pred_f, dtype=np.uint32)
        gt = np.fromfile(gt_f, dtype=np.uint32)
        pc = np.fromfile(bin_f, dtype=np.float32).reshape(-1, 4)
        
        # Skip if sizes don't match
        min_len = min(len(pred), len(gt), len(pc))
        pred = pred[:min_len]
        gt = gt[:min_len]
        pc = pc[:min_len]
        
        # Only evaluate non-Undefined points
        valid = gt > 0
        pred_valid = pred[valid]
        gt_valid = gt[valid]
        pc_valid = pc[valid]
        
        # Confusion matrix
        for p, g in zip(pred_valid, gt_valid):
            if p < NUM_CLASSES and g < NUM_CLASSES:
                confusion[g, p] += 1
        
        # Per-class counts
        for g in gt_valid:
            gt_counts[int(g)] += 1
        for p in pred_valid:
            pred_counts[int(p)] += 1
        
        # Per-distance accuracy
        dist = np.sqrt(pc_valid[:, 0]**2 + pc_valid[:, 1]**2)
        for i, d in enumerate(dist_bins):
            d_next = dist_bins[i+1] if i+1 < len(dist_bins) else 999
            mask = (dist >= d) & (dist < d_next)
            if mask.sum() > 0:
                dist_correct[d] += np.sum(pred_valid[mask] == gt_valid[mask])
                dist_total[d] += mask.sum()
    
    return confusion, gt_counts, pred_counts, dist_correct, dist_total


def compute_iou(confusion):
    """Compute per-class IoU from confusion matrix."""
    ious = {}
    for c in range(1, NUM_CLASSES):  # Skip Undefined (0)
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        if tp + fp + fn > 0:
            ious[c] = tp / (tp + fp + fn)
        else:
            ious[c] = float('nan')
    return ious


def main():
    pred_dir = os.path.expanduser('~/AWARE/val_predictions/')
    data_dir = os.path.expanduser('~/AWARE/data/waymo_kitti/combined/dataset/')
    
    pred_files = sorted(glob(os.path.join(pred_dir, 'sequences/*/predictions/*.label')))
    print(f"Found {len(pred_files)} prediction files")
    
    # Build corresponding GT and BIN paths
    gt_files = []
    bin_files = []
    for pf in pred_files:
        # Extract sequence/frame path
        rel = pf[len(pred_dir):]  # sequences/800/predictions/000000.label
        gt = os.path.join(data_dir, rel.replace('predictions', 'labels'))
        bn = os.path.join(data_dir, rel.replace('predictions', 'velodyne').replace('.label', '.bin'))
        gt_files.append(gt)
        bin_files.append(bn)
    
    # Verify files exist
    missing = sum(1 for f in gt_files if not os.path.exists(f))
    print(f"Missing GT files: {missing}")
    missing_bin = sum(1 for f in bin_files if not os.path.exists(f))
    print(f"Missing BIN files: {missing_bin}")
    
    # Filter to existing files only
    valid = [(p, g, b) for p, g, b in zip(pred_files, gt_files, bin_files) 
             if os.path.exists(g) and os.path.exists(b)]
    pred_files, gt_files, bin_files = zip(*valid)
    print(f"Evaluating on {len(pred_files)} frames")
    
    # Compute stats
    confusion, gt_counts, pred_counts, dist_correct, dist_total = \
        compute_confusion_matrix(pred_files, gt_files, bin_files)
    
    ious = compute_iou(confusion)
    
    # Print results
    print("\n" + "=" * 70)
    print("WAFFLEIRON WAYMO VALIDATION RESULTS")
    print("=" * 70)
    
    print(f"\n--- PER-CLASS IoU ---")
    valid_ious = []
    for c in range(1, NUM_CLASSES):
        iou = ious[c]
        gt_n = gt_counts[c]
        pred_n = pred_counts[c]
        name = CLASS_NAMES[c]
        if np.isnan(iou):
            print(f"  {name:20s}: N/A (no GT points)")
        else:
            print(f"  {name:20s}: {100*iou:.1f}%  (GT: {gt_n:>10,} pts, Pred: {pred_n:>10,} pts)")
            valid_ious.append(iou)
    
    miou = 100 * np.nanmean(valid_ious)
    print(f"\n  mIoU: {miou:.1f}%")
    
    # Overall accuracy
    total_correct = sum(confusion[c, c] for c in range(1, NUM_CLASSES))
    total_points = sum(gt_counts[c] for c in range(1, NUM_CLASSES))
    print(f"  Overall accuracy: {100*total_correct/total_points:.1f}%")
    
    # Per-distance accuracy
    print(f"\n--- ACCURACY BY DISTANCE ---")
    dist_bins = [0, 10, 20, 30, 40, 50]
    for i, d in enumerate(dist_bins):
        d_next = dist_bins[i+1] if i+1 < len(dist_bins) else 50
        label = f"{d}-{d_next}m" if d_next != 50 else f"{d}m+"
        if dist_total[d] > 0:
            acc = 100 * dist_correct[d] / dist_total[d]
            print(f"  {label:10s}: {acc:.1f}% ({dist_total[d]:>10,} points)")
    
    # Most confused pairs
    print(f"\n--- TOP 10 CONFUSION PAIRS ---")
    pairs = []
    for gt_c in range(1, NUM_CLASSES):
        for pred_c in range(1, NUM_CLASSES):
            if gt_c != pred_c and confusion[gt_c, pred_c] > 0:
                pairs.append((confusion[gt_c, pred_c], gt_c, pred_c))
    pairs.sort(reverse=True)
    for count, gt_c, pred_c in pairs[:10]:
        gt_name = CLASS_NAMES[gt_c]
        pred_name = CLASS_NAMES[pred_c]
        gt_total = gt_counts[gt_c]
        pct = 100 * count / gt_total if gt_total > 0 else 0
        print(f"  {gt_name:20s} -> {pred_name:20s}: {count:>10,} pts ({pct:.1f}% of {gt_name})")
    
    # Save confusion matrix as CSV
    csv_path = os.path.expanduser('~/AWARE/logs/confusion_matrix.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['GT \\ Pred'] + [CLASS_NAMES[c] for c in range(1, NUM_CLASSES)]
        writer.writerow(header)
        for gt_c in range(1, NUM_CLASSES):
            row = [CLASS_NAMES[gt_c]] + [int(confusion[gt_c, pred_c]) for pred_c in range(1, NUM_CLASSES)]
            writer.writerow(row)
    print(f"\nConfusion matrix saved to: {csv_path}")
    
    # Save per-class results as CSV
    results_path = os.path.expanduser('~/AWARE/logs/per_class_results.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'class_name', 'iou', 'gt_points', 'pred_points'])
        for c in range(1, NUM_CLASSES):
            writer.writerow([c, CLASS_NAMES[c], round(ious[c], 4) if not np.isnan(ious[c]) else 'N/A', 
                           gt_counts[c], pred_counts[c]])
    print(f"Per-class results saved to: {results_path}")


if __name__ == '__main__':
    main()
