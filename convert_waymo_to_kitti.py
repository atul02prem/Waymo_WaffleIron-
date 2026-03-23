#!/usr/bin/env python3
"""
Convert Waymo v1.4.3 TFRecords to SemanticKITTI format for WaffleIron.

Based on the working segmentation label extraction from visualize_segm.py.
Segmentation labels are in frame.lasers[i].ri_return1.segmentation_label_compressed
as ZLIB-compressed MatrixInt32 protos with channels [instance_id, semantic_class_id].

Only segments/frames with segmentation labels are converted.
Labels are at 2Hz (every 5th frame), not every frame.

Usage:
    # Test on 2 segments
    python convert_waymo_v143_to_kitti.py \
        --input_dir ~/AWARE/data/waymo/train/ \
        --output_dir ~/AWARE/data/waymo_kitti/train/ \
        --max_segments 2

    # Full conversion in tmux
    python convert_waymo_v143_to_kitti.py \
        --input_dir ~/AWARE/data/waymo/train/ \
        --output_dir ~/AWARE/data/waymo_kitti/train/
"""

import os
import sys
import zlib
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils


# Waymo laser name enum
TOP_LIDAR = 1  # open_dataset.LaserName.TOP

# Waymo semantic class -> SemanticKITTI raw label ID
WAYMO_TO_KITTI_RAW = {
    0:  0,   # Undefined -> unlabeled
    1:  10,  # Car -> car
    2:  18,  # Truck -> truck
    3:  13,  # Bus -> bus
    4:  20,  # Other Vehicle -> other-vehicle
    5:  32,  # Motorcyclist -> motorcyclist
    6:  31,  # Bicyclist -> bicyclist
    7:  30,  # Pedestrian -> person
    8:  81,  # Sign -> traffic-sign
    9:  81,  # Traffic Light -> traffic-sign
    10: 80,  # Pole -> pole
    11: 80,  # Construction Cone -> pole
    12: 11,  # Bicycle -> bicycle
    13: 15,  # Motorcycle -> motorcycle
    14: 50,  # Building -> building
    15: 70,  # Vegetation -> vegetation
    16: 71,  # Tree Trunk -> trunk
    17: 72,  # Curb -> terrain
    18: 40,  # Road -> road
    19: 60,  # Lane Marker -> lane-marking
    20: 49,  # Other Ground -> other-ground
    21: 48,  # Walkable -> sidewalk
    22: 48,  # Sidewalk -> sidewalk
}

def remap_waymo_to_kitti(labels):
    remapped = np.zeros_like(labels)
    for waymo_id, kitti_id in WAYMO_TO_KITTI_RAW.items():
        remapped[labels == waymo_id] = kitti_id
    return remapped

def decompress_semseg_label(compressed_bytes):
    """
    Decompress segmentation_label_compressed from a Waymo RangeImage proto.
    Returns [H, W, 2] where channel 0 = semantic_class, channel 1 = instance_id.
    """
    decompressed = zlib.decompress(compressed_bytes)
    matrix = open_dataset.MatrixInt32()
    matrix.ParseFromString(decompressed)
    shape = list(matrix.shape.dims)
    raw = np.array(matrix.data, dtype=np.int32).reshape(shape)
    # Raw is [instance, semantic], swap to [semantic, instance]
    data = np.stack([raw[..., 1], raw[..., 0]], axis=-1)
    return data


def frame_has_semseg(frame):
    """Check if a frame has segmentation labels on the TOP lidar."""
    for laser in frame.lasers:
        if laser.name == TOP_LIDAR:
            return len(laser.ri_return1.segmentation_label_compressed) > 0
    return False


def extract_points_and_labels(frame):
    """
    Extract 3D point cloud and per-point semantic labels from a frame.
    Returns:
        points: (N, 4) float32 [x, y, z, intensity]
        labels: (N,) uint32 semantic class IDs
        has_labels: bool
    """
    # Parse range images
    range_images, camera_projections, _, range_image_top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    # We process ALL 5 lidars for points, but only lidars with seg labels get labels
    all_points = []
    all_labels = []
    has_any_labels = False

    for laser in frame.lasers:
        laser_name = laser.name

        if laser_name not in range_images:
            continue

        # Get range image for this laser
        ri = range_images[laser_name][0]  # Return 1
        ri_tensor = tf.reshape(tf.convert_to_tensor(ri.data), ri.shape.dims)
        ri_np = ri_tensor.numpy()

        # Valid pixel mask
        valid_mask = ri_np[:, :, 0] > 0

        if not valid_mask.any():
            continue

        # Check for segmentation labels on this laser
        has_seg = len(laser.ri_return1.segmentation_label_compressed) > 0

        if has_seg:
            semseg = decompress_semseg_label(
                laser.ri_return1.segmentation_label_compressed
            )
            semantic_classes = semseg[:, :, 0]  # [H, W] semantic class IDs
            point_labels = semantic_classes[valid_mask].flatten().astype(np.uint32)
            has_any_labels = True
        else:
            num_valid = valid_mask.sum()
            point_labels = np.zeros(num_valid, dtype=np.uint32)

        all_labels.append(point_labels)

    # Now get the actual 3D points using frame_utils
    ri_dict = {k: [range_images[k][0]] for k in sorted(range_images.keys())}
    cp_dict = {k: [camera_projections[k][0]] for k in sorted(camera_projections.keys())}

    pts_list, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, ri_dict, cp_dict, range_image_top_pose, keep_polar_features=False
    )

    # pts_list is ordered by sorted laser keys
    # We need to match the same order we used above
    ordered_keys = sorted(range_images.keys())

    all_points = []
    for idx, key in enumerate(ordered_keys):
        pts = pts_list[idx]  # (N, 3) xyz
        if len(pts) > 0:
            # Add intensity as 4th column (from range image channel 1)
            ri = range_images[key][0]
            ri_tensor = tf.reshape(tf.convert_to_tensor(ri.data), ri.shape.dims)
            ri_np = ri_tensor.numpy()
            valid_mask = ri_np[:, :, 0] > 0
            intensity = ri_np[:, :, 1][valid_mask].flatten()

            if len(intensity) == len(pts):
                pts_with_intensity = np.column_stack([pts, intensity])
            else:
                pts_with_intensity = np.column_stack([
                    pts, np.zeros(len(pts), dtype=np.float32)
                ])
            all_points.append(pts_with_intensity.astype(np.float32))

    if len(all_points) == 0:
        return None, None, False

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.uint32)

    # Sanity check: points and labels must match
    if len(points) != len(labels):
        min_len = min(len(points), len(labels))
        points = points[:min_len]
        labels = labels[:min_len]

    return points, labels, has_any_labels


def convert_segment(tfrecord_path, output_dir, segment_idx):
    """Convert one TFRecord segment to SemanticKITTI format."""
    seq_name = f"{segment_idx:04d}"
    vel_dir = os.path.join(output_dir, "sequences", seq_name, "velodyne")
    lab_dir = os.path.join(output_dir, "sequences", seq_name, "labels")
    os.makedirs(vel_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

    frame_count = 0
    seg_frame_count = 0
    skip_count = 0

    for frame_idx, raw_record in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytes(raw_record.numpy()))

        # Only process frames that have segmentation labels
        if not frame_has_semseg(frame):
            skip_count += 1
            continue

        try:
            points, labels, has_labels = extract_points_and_labels(frame)
        except Exception as e:
            continue

        if points is None or len(points) == 0:
            continue

        #labels = remap_waymo_to_kitti(labels) #disabled remapping

        # Save as .bin (x, y, z, intensity) - SemanticKITTI format
        bin_path = os.path.join(vel_dir, f"{frame_count:06d}.bin")
        points.tofile(bin_path)

        # Save as .label (uint32) - SemanticKITTI format
        label_path = os.path.join(lab_dir, f"{frame_count:06d}.label")
        labels.tofile(label_path)

        if has_labels and labels.max() > 0:
            seg_frame_count += 1

        frame_count += 1

    return frame_count, seg_frame_count, skip_count


def scan_for_segments_with_labels(tfrecord_paths, max_scan=None):
    """Quick scan to find which segments have segmentation labels."""
    segments_with_labels = []

    paths_to_scan = tfrecord_paths[:max_scan] if max_scan else tfrecord_paths

    for tfr in tqdm(paths_to_scan, desc="Scanning for segments with labels"):
        dataset = tf.data.TFRecordDataset(tfr, compression_type='')
        for raw in dataset.take(5):  # Check first 5 frames
            frame = open_dataset.Frame()
            frame.ParseFromString(bytes(raw.numpy()))
            if frame_has_semseg(frame):
                segments_with_labels.append(tfr)
                break

    return segments_with_labels


def main():
    parser = argparse.ArgumentParser(
        description="Convert Waymo v1.4.3 TFRecords to SemanticKITTI format"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Directory with .tfrecord files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory in SemanticKITTI format")
    parser.add_argument("--max_segments", type=int, default=-1,
                        help="Max segments to convert (-1 for all with labels)")
    parser.add_argument("--skip_scan", action="store_true",
                        help="Skip scanning and try all segments")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start from this segment index (for resuming)")
    args = parser.parse_args()

    # Find all tfrecords
    tfrecords = sorted(glob(os.path.join(args.input_dir, "*.tfrecord")))
    print(f"Found {len(tfrecords)} total segments in {args.input_dir}")

    if len(tfrecords) == 0:
        # Check for nested directory
        tfrecords = sorted(glob(os.path.join(args.input_dir, "*", "*.tfrecord")))
        if len(tfrecords) == 0:
            print("No .tfrecord files found!")
            sys.exit(1)
        print(f"Found {len(tfrecords)} segments in subdirectories")

    # Scan for segments with segmentation labels (not all have them)
    if not args.skip_scan:
        print("\nScanning for segments with segmentation labels...")
        print("(Not all segments have per-point labels, only a subset)")
        segments_with_labels = scan_for_segments_with_labels(tfrecords)
        print(f"\nFound {len(segments_with_labels)} segments WITH segmentation labels "
              f"out of {len(tfrecords)} total")

        if len(segments_with_labels) == 0:
            print("ERROR: No segments with segmentation labels found!")
            print("Make sure you downloaded the correct Waymo dataset version.")
            sys.exit(1)

        tfrecords = segments_with_labels

    # Apply start index and max segments
    if args.start_idx > 0:
        tfrecords = tfrecords[args.start_idx:]
        print(f"Starting from segment index {args.start_idx}")

    if args.max_segments > 0:
        tfrecords = tfrecords[:args.max_segments]

    print(f"\nWill convert {len(tfrecords)} segments")

    os.makedirs(args.output_dir, exist_ok=True)

    total_frames = 0
    total_seg_frames = 0
    total_skipped = 0

    for seg_idx, tfr in enumerate(tqdm(tfrecords, desc="Converting")):
        actual_idx = seg_idx + args.start_idx
        frame_count, seg_frames, skipped = convert_segment(
            tfr, args.output_dir, actual_idx
        )
        total_frames += frame_count
        total_seg_frames += seg_frames
        total_skipped += skipped

        if (seg_idx + 1) % 10 == 0:
            print(f"\n  Progress: {seg_idx+1}/{len(tfrecords)} segments | "
                  f"{total_frames} frames saved | "
                  f"{total_seg_frames} with labels | "
                  f"{total_skipped} frames skipped (no labels)")

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Segments processed:    {len(tfrecords)}")
    print(f"Frames saved:          {total_frames}")
    print(f"Frames with labels:    {total_seg_frames}")
    print(f"Frames skipped:        {total_skipped} (no segmentation labels)")
    print(f"Output directory:      {args.output_dir}")
    print(f"")
    print(f"Each sequence folder contains:")
    print(f"  velodyne/*.bin  — point clouds (x, y, z, intensity) float32")
    print(f"  labels/*.label  — per-point semantic class IDs, uint32")
    print(f"{'='*60}")

    # Quick verification
    print(f"\nVerification:")
    sample_vel = sorted(glob(os.path.join(args.output_dir, "sequences", "0000", "velodyne", "*.bin")))
    sample_lab = sorted(glob(os.path.join(args.output_dir, "sequences", "0000", "labels", "*.label")))
    if sample_vel and sample_lab:
        pts = np.fromfile(sample_vel[0], dtype=np.float32).reshape(-1, 4)
        lab = np.fromfile(sample_lab[0], dtype=np.uint32)
        print(f"  Sample point cloud shape: {pts.shape}")
        print(f"  Sample labels shape: {lab.shape}")
        print(f"  Unique labels: {np.unique(lab).tolist()}")
        print(f"  Point range X: [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]")
        print(f"  Point range Y: [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")
        print(f"  Point range Z: [{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]")


if __name__ == "__main__":
    main()
