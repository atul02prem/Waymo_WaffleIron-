#!/usr/bin/env python3
"""
Fine-tune WaffleIron on Waymo dataset.
Loads pretrained SemanticKITTI weights, skips the last classification layer
(19 classes -> 22 classes), and saves as a new checkpoint.
"""

import os
import sys
import torch
import yaml
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waffleiron.segmenter import Segmenter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = Segmenter(
        input_channels=config["embedding"]["size_input"],
        feat_channels=config["waffleiron"]["nb_channels"],
        depth=config["waffleiron"]["depth"],
        grid_shape=config["waffleiron"]["grids_size"],
        nb_class=config["classif"]["nb_class"],
        drop_path_prob=config["waffleiron"].get("drop", 0.2),
    )

    ckpt = torch.load(args.pretrained, map_location="cpu")
    pretrained_state = ckpt["net"]

    new_state = {}
    for k, v in model.state_dict().items():
        new_state["module." + k] = v

    loaded = 0
    skipped = 0
    for k, v in pretrained_state.items():
        if k in new_state:
            if v.shape == new_state[k].shape:
                new_state[k] = v
                loaded += 1
            else:
                print(f"  SKIP (shape mismatch): {k}: pretrained {v.shape} vs new {new_state[k].shape}")
                skipped += 1
        else:
            print(f"  SKIP (not in new model): {k}")
            skipped += 1

    print(f"\nLoaded {loaded} parameters, skipped {skipped}")

    new_ckpt = {
        "net": new_state,
        "epoch": -1,
        "best_miou": 0,
    }

    output_path = os.path.join(args.output_dir, "ckpt_last.pth")
    torch.save(new_ckpt, output_path)
    print(f"Saved new checkpoint to: {output_path}")


if __name__ == "__main__":
    main()
