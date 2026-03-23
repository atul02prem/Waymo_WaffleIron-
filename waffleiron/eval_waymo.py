#!/usr/bin/env python3
"""
Run inference on Waymo validation set using trained WaffleIron model.
Saves predicted .label files for every frame.
"""

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from waffleiron import Segmenter
from datasets import WaymoDataset, Collate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Waymo Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--path_dataset", type=str, required=True)
    parser.add_argument("--result_folder", type=str, required=True)
    parser.add_argument("--num_votes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--phase", type=str, default="val")
    args = parser.parse_args()
    assert args.num_votes % args.batch_size == 0
    os.makedirs(args.result_folder, exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Waymo learning_map_inv (identity for Waymo: 0->0, 1->1, ..., 22->22)
    with open("./datasets/waymo.yaml") as f:
        waymoyaml = yaml.safe_load(f)
    remapdict = waymoyaml["learning_map_inv"]
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    # Dataloader
    tta = args.num_votes > 1
    dataset = WaymoDataset(
        rootdir=args.path_dataset,
        input_feat=config["embedding"]["input_feat"],
        voxel_size=config["embedding"]["voxel_size"],
        num_neighbors=config["embedding"]["neighbors"],
        dim_proj=config["waffleiron"]["dim_proj"],
        grids_shape=config["waffleiron"]["grids_size"],
        fov_xyz=config["waffleiron"]["fov_xyz"],
        phase=args.phase,
        tta=tta,
    )
    if args.num_votes > 1:
        new_list = []
        for f in dataset.im_idx:
            for v in range(args.num_votes):
                new_list.append(f)
        dataset.im_idx = new_list
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=Collate(),
    )
    args.num_votes = args.num_votes // args.batch_size

    print(f"Dataset: {len(dataset)} frames")
    print(f"Phase: {args.phase}")
    print(f"Checkpoint: {args.ckpt}")

    # Build network
    net = Segmenter(
        input_channels=config["embedding"]["size_input"],
        feat_channels=config["waffleiron"]["nb_channels"],
        depth=config["waffleiron"]["depth"],
        grid_shape=config["waffleiron"]["grids_size"],
        nb_class=config["classif"]["nb_class"],
        drop_path_prob=config["waffleiron"]["drop"],
    )
    net = net.cuda()

    # Load weights
    ckpt = torch.load(args.ckpt, map_location="cuda:0")
    try:
        net.load_state_dict(ckpt["net"])
    except:
        state_dict = {}
        for key in ckpt["net"].keys():
            state_dict[key[len("module."):]] = ckpt["net"][key]
        net.load_state_dict(state_dict)
    net.compress()
    net.eval()

    print(f"Model loaded. Running inference...")

    # Re-activate droppath if voting
    if tta:
        import waffleiron
        for m in net.modules():
            if isinstance(m, waffleiron.backbone.DropPath):
                m.train()

    # Inference
    id_vote = 0
    saved = 0
    for it, batch in enumerate(
        tqdm(loader, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")
    ):
        if id_vote == 0:
            vote = None

        feat = batch["feat"].cuda(non_blocking=True)
        batch["upsample"] = [up.cuda(non_blocking=True) for up in batch["upsample"]]
        cell_ind = batch["cell_ind"].cuda(non_blocking=True)
        occupied_cell = batch["occupied_cells"].cuda(non_blocking=True)
        neighbors_emb = batch["neighbors_emb"].cuda(non_blocking=True)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        with torch.autocast("cuda", enabled=True):
            with torch.inference_mode():
                out = net(*net_inputs)
                for b in range(out.shape[0]):
                    temp = out[b, :, batch["upsample"][b]].T
                    if vote is None:
                        vote = torch.softmax(temp, dim=1)
                    else:
                        vote += torch.softmax(temp, dim=1)
        id_vote += 1

        if id_vote == args.num_votes:
            # +1 because model outputs 0-21, but raw labels are 1-22
            pred_label = (vote.max(1)[1] + 1).cpu().numpy().reshape(-1).astype(np.uint32)
            # Remap through learning_map_inv (identity for Waymo)
            pred_label = remap_lut[pred_label].astype(np.uint32)

            # Save prediction .label file
            assert batch["filename"][0] == batch["filename"][-1]
            label_file = batch["filename"][0][
                len(os.path.join(dataset.rootdir, "dataset/")):
            ]
            label_file = label_file.replace("velodyne", "predictions")[:-3] + "label"
            label_file = os.path.join(args.result_folder, label_file)
            os.makedirs(os.path.split(label_file)[0], exist_ok=True)
            pred_label.tofile(label_file)
            saved += 1
            id_vote = 0

    print(f"\nDone! Saved {saved} prediction files to {args.result_folder}")
