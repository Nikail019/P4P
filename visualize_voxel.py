#!/usr/bin/env python3
"""
Quick visualizer for .npz voxel outputs from step_ap242_to_voxel_semantic_layer.py.

Usage:
    python visualize_voxel.py Square_Hole_voxel_semantic.npz
    python visualize_voxel.py Square_Hole_voxel_semantic.npz --mode regions
    python visualize_voxel.py Square_Hole_voxel_semantic.npz --mode pmi --json Square_Hole_voxel_semantic.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_npz(path: Path):
    data = np.load(path)
    occupancy = data["occupancy"].astype(bool)
    region_grid = data["region_id_grid"].astype(np.int32)
    pitch = float(data["pitch"])
    transform = data["transform"]
    return occupancy, region_grid, pitch, transform


def show_occupancy(occupancy, pitch):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(occupancy, edgecolor="k", linewidth=0.02, alpha=0.3)
    ax.set_title(f"Occupancy | shape={occupancy.shape} | pitch={pitch}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def show_regions(occupancy, region_grid, pitch):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    occ_idx = np.argwhere(occupancy)
    unique_regions = sorted(set(int(region_grid[tuple(idx)]) for idx in occ_idx) - {0})
    cmap = plt.get_cmap("tab20")

    colors = np.empty(occupancy.shape, dtype=object)
    for idx in occ_idx:
        rid = int(region_grid[tuple(idx)])
        if rid > 0:
            colors[tuple(idx)] = cmap((rid - 1) % 20)
        else:
            colors[tuple(idx)] = (0.85, 0.85, 0.85, 0.15)

    ax.voxels(occupancy, facecolors=colors, edgecolor="k", linewidth=0.01)
    ax.set_title(f"Face Regions | {len(unique_regions)} regions | pitch={pitch}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def show_pmi(occupancy, region_grid, pitch, json_path: Path):
    data = json.load(json_path.open())
    pmi_records = data["pmi_records"]

    # Collect region IDs that have PMI attached
    pmi_regions = set()
    for rec in pmi_records:
        for rid in rec.get("linked_region_ids", []):
            pmi_regions.add(rid)

    if not pmi_regions:
        print("No PMI records have linked region IDs. Nothing to highlight.")
        print("PMI records:")
        for rec in pmi_records:
            print(f"  {rec['pmi_id']}: valid={rec['valid_association']} regions={rec.get('linked_region_ids', [])}")
        return

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    occ_idx = np.argwhere(occupancy)
    colors = np.empty(occupancy.shape, dtype=object)

    cmap = plt.get_cmap("Set1")
    pmi_region_list = sorted(pmi_regions)
    region_color_map = {rid: cmap(i % 9) for i, rid in enumerate(pmi_region_list)}

    for idx in occ_idx:
        rid = int(region_grid[tuple(idx)])
        if rid in pmi_regions:
            colors[tuple(idx)] = region_color_map[rid]
        else:
            colors[tuple(idx)] = (0.9, 0.9, 0.9, 0.08)

    ax.voxels(occupancy, facecolors=colors, edgecolor="k", linewidth=0.01)

    legend_lines = []
    for rec in pmi_records:
        rids = rec.get("linked_region_ids", [])
        if rids:
            label = rec.get("label") or rec["pmi_id"]
            val = rec.get("value")
            unit = rec.get("unit", "")
            if val is not None:
                label += f" = {val} {unit}"
            legend_lines.append(f"{label} -> regions {rids}")

    title = "PMI-linked voxel regions\n" + "\n".join(legend_lines)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize voxel NPZ outputs")
    parser.add_argument("npz_file", type=Path)
    parser.add_argument("--mode", choices=["occupancy", "regions", "pmi"], default="regions")
    parser.add_argument("--json", type=Path, default=None, help="JSON sidecar (required for --mode pmi)")
    args = parser.parse_args()

    occupancy, region_grid, pitch, transform = load_npz(args.npz_file)
    print(f"Loaded: shape={occupancy.shape} pitch={pitch} occupied={np.count_nonzero(occupancy)}")

    if args.mode == "occupancy":
        show_occupancy(occupancy, pitch)
    elif args.mode == "regions":
        show_regions(occupancy, region_grid, pitch)
    elif args.mode == "pmi":
        json_path = args.json or args.npz_file.with_suffix(".json")
        if not json_path.exists():
            raise SystemExit(f"JSON sidecar not found: {json_path}")
        show_pmi(occupancy, region_grid, pitch, json_path)


if __name__ == "__main__":
    main()
