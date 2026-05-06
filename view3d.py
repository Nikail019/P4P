#!/usr/bin/env python3
"""
Interactive 3D voxel viewer using PyVista.

Usage:
    python view3d.py Square_Hole_voxel_semantic.npz
    python view3d.py Square_Hole_voxel_semantic.npz --mode regions
    python view3d.py Square_Hole_voxel_semantic.npz --mode pmi --json Square_Hole_voxel_semantic.json
    python view3d.py Square_Hole_voxel_semantic.npz --mode explode --json Square_Hole_voxel_semantic.json
    python view3d.py Square_Hole_voxel_semantic.npz --mode mbd --json Square_Hole_voxel_semantic.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyvista as pv


def load_npz(path: Path):
    data = np.load(path)
    occupancy = data["occupancy"].astype(bool)
    region_grid = data["region_id_grid"].astype(np.int32)
    pitch = float(data["pitch"])
    transform = data["transform"]
    bounds = data["bounds"] if "bounds" in data else None
    return occupancy, region_grid, pitch, transform, bounds


def voxels_to_polydata(occupancy: np.ndarray, pitch: float, transform: np.ndarray) -> pv.PolyData:
    """Convert occupied voxel centers to a point cloud, then glyphed as cubes."""
    indices = np.argwhere(occupancy)
    # Transform voxel indices to world coordinates
    origin = transform[:3, 3]
    centers = origin + indices * pitch + pitch * 0.5
    return pv.PolyData(centers)


def make_voxel_grid(occupancy: np.ndarray, pitch: float, transform: np.ndarray) -> pv.ImageData:
    """Build a structured grid from the occupancy array."""
    grid = pv.ImageData(
        dimensions=np.array(occupancy.shape) + 1,
        spacing=(pitch, pitch, pitch),
        origin=transform[:3, 3],
    )
    grid.cell_data["occupancy"] = occupancy.flatten(order="F").astype(np.uint8)
    return grid


def show_occupancy(occupancy: np.ndarray, pitch: float, transform: np.ndarray):
    grid = make_voxel_grid(occupancy, pitch, transform)
    threshed = grid.threshold(0.5, scalars="occupancy")

    pl = pv.Plotter()
    pl.add_mesh(threshed, color="steelblue", opacity=0.6, show_edges=True, edge_color="gray", line_width=0.3)
    pl.add_axes()
    pl.add_text(f"Occupancy | shape={occupancy.shape} | pitch={pitch}", font_size=10)
    pl.show()


def show_regions(occupancy: np.ndarray, region_grid: np.ndarray, pitch: float, transform: np.ndarray):
    grid = make_voxel_grid(occupancy, pitch, transform)
    grid.cell_data["region_id"] = region_grid.flatten(order="F").astype(np.float32)

    threshed = grid.threshold(0.5, scalars="occupancy")

    pl = pv.Plotter()
    unique_regions = sorted(set(int(v) for v in threshed.cell_data["region_id"] if v > 0))
    pl.add_mesh(
        threshed,
        scalars="region_id",
        cmap="tab20",
        show_edges=False,
        opacity=0.85,
        scalar_bar_args={"title": "Region ID", "n_labels": min(len(unique_regions) + 1, 10)},
    )
    pl.add_axes()
    pl.add_text(f"Face Regions | {len(unique_regions)} regions", font_size=10)
    pl.show()


def show_pmi(
    occupancy: np.ndarray,
    region_grid: np.ndarray,
    pitch: float,
    transform: np.ndarray,
    json_data: dict,
):
    pmi_records = json_data.get("pmi_records", [])
    r2v = json_data.get("region_to_voxel_count", {})

    # Collect PMI-linked region IDs
    pmi_regions: Dict[int, List[str]] = {}
    for rec in pmi_records:
        for rid in rec.get("linked_region_ids", []):
            label = rec.get("label") or rec["pmi_id"]
            val = rec.get("value")
            unit = rec.get("unit", "")
            desc = label
            if val is not None:
                desc += f" = {val} {unit}"
            pmi_regions.setdefault(rid, []).append(desc)

    if not pmi_regions:
        print("No PMI records have linked regions. Falling back to region view.")
        show_regions(occupancy, region_grid, pitch, transform)
        return

    # Build highlight array: 0 = non-PMI, 1..N = PMI group index
    highlight = np.zeros_like(region_grid, dtype=np.float32)
    pmi_rid_list = sorted(pmi_regions.keys())
    for i, rid in enumerate(pmi_rid_list, 1):
        highlight[region_grid == rid] = float(i)

    grid = make_voxel_grid(occupancy, pitch, transform)
    grid.cell_data["pmi_highlight"] = highlight.flatten(order="F")
    grid.cell_data["occupancy_f"] = occupancy.flatten(order="F").astype(np.float32)

    threshed = grid.threshold(0.5, scalars="occupancy_f")

    pl = pv.Plotter()

    # Non-PMI voxels: gray, transparent
    non_pmi = threshed.threshold(0.5, scalars="pmi_highlight", invert=True)
    if non_pmi.n_cells > 0:
        pl.add_mesh(non_pmi, color="lightgray", opacity=0.08, show_edges=False)

    # PMI voxels: colored
    pmi_mesh = threshed.threshold(0.5, scalars="pmi_highlight")
    if pmi_mesh.n_cells > 0:
        pl.add_mesh(
            pmi_mesh,
            scalars="pmi_highlight",
            cmap="Set1",
            show_edges=True,
            edge_color="gray",
            line_width=0.3,
            opacity=0.9,
            scalar_bar_args={"title": "PMI Group"},
        )

    # Build legend text
    lines = ["PMI-linked regions:"]
    for i, rid in enumerate(pmi_rid_list, 1):
        descs = pmi_regions[rid]
        for d in descs:
            lines.append(f"  [{i}] Region {rid}: {d}")
    legend_text = "\n".join(lines)

    pl.add_axes()
    pl.add_text(legend_text, font_size=8, position="lower_left")
    pl.show()


def show_exploded(
    occupancy: np.ndarray,
    region_grid: np.ndarray,
    pitch: float,
    transform: np.ndarray,
    json_data: dict,
):
    """Exploded view: each face region pulled apart from the center."""
    occ_regions = json_data.get("occ_regions", [])
    origin = transform[:3, 3]

    # Compute model center
    occ_indices = np.argwhere(occupancy)
    model_center = origin + np.mean(occ_indices, axis=0) * pitch + pitch * 0.5

    unique_rids = sorted(set(int(region_grid[tuple(idx)]) for idx in occ_indices) - {0})

    pl = pv.Plotter()
    cmap = pv.LookupTable(cmap="tab20", n_values=20)

    explode_factor = 0.3  # fraction of model size

    for rid in unique_rids:
        mask = (region_grid == rid) & occupancy
        region_indices = np.argwhere(mask)
        if len(region_indices) == 0:
            continue

        # Region center
        region_center = origin + np.mean(region_indices, axis=0) * pitch + pitch * 0.5

        # Explode offset: push away from model center
        offset_dir = region_center - model_center
        norm = np.linalg.norm(offset_dir)
        if norm > 1e-9:
            offset_dir = offset_dir / norm
        offset = offset_dir * norm * explode_factor

        # Build mini-grid for this region
        region_occ = mask.astype(bool)
        region_grid_pv = make_voxel_grid(region_occ, pitch, transform)
        threshed = region_grid_pv.threshold(0.5, scalars="occupancy")
        threshed.points += offset

        color = cmap.map_value((rid - 1) % 20)[:3]
        color_f = tuple(c / 255.0 for c in color)

        # Find region label
        label = f"Region {rid}"
        for r in occ_regions:
            if r["occ_face_index"] == rid:
                label = f"R{rid}: {r['surface_type']}"
                if r.get("radius") is not None:
                    label += f" (r={r['radius']:.1f})"
                break

        pl.add_mesh(threshed, color=color_f, opacity=0.85, show_edges=True,
                     edge_color="gray", line_width=0.3, label=label)

    pl.add_legend(face="rectangle", bcolor="white", size=(0.25, 0.35))
    pl.add_axes()
    pl.add_text("Exploded Region View", font_size=10)
    pl.show()


def _region_centroid_world(rid: int, occupancy: np.ndarray, region_grid: np.ndarray,
                           pitch: float, origin: np.ndarray) -> Optional[np.ndarray]:
    """Compute the world-space centroid of a voxel region."""
    mask = (region_grid == rid) & occupancy
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
    return origin + np.mean(indices, axis=0) * pitch + pitch * 0.5


def _region_normal(rid: int, occ_regions: list) -> Optional[np.ndarray]:
    """Get the face normal direction for a region from the JSON metadata."""
    for r in occ_regions:
        if r["occ_face_index"] == rid and r.get("direction"):
            d = np.array(r["direction"], dtype=np.float64)
            n = np.linalg.norm(d)
            return d / n if n > 1e-9 else None
    return None


def _offset_point(center: np.ndarray, normal: Optional[np.ndarray], model_size: float,
                  factor: float = 0.35) -> np.ndarray:
    """Push a label point outward from the surface so it floats above the face."""
    if normal is not None:
        return center + normal * model_size * factor
    return center + np.array([0, 0, model_size * factor])


def show_mbd(
    occupancy: np.ndarray,
    region_grid: np.ndarray,
    pitch: float,
    transform: np.ndarray,
    json_data: dict,
):
    """MBD annotation view: part geometry with PMI labels, leader lines, and dimension lines."""
    pmi_records = json_data.get("pmi_records", [])
    occ_regions = json_data.get("occ_regions", [])
    r2v = json_data.get("region_to_voxel_count", {})
    origin = transform[:3, 3]

    # -- Render the base geometry (regions colored) --
    grid = make_voxel_grid(occupancy, pitch, transform)
    grid.cell_data["region_id"] = region_grid.flatten(order="F").astype(np.float32)
    threshed = grid.threshold(0.5, scalars="occupancy")

    # Identify PMI-linked regions for highlighting
    pmi_region_set = set()
    for rec in pmi_records:
        for rid in rec.get("linked_region_ids", []):
            pmi_region_set.add(rid)

    # Build color array: PMI faces get saturated color, rest is light gray
    n_cells = threshed.n_cells
    rids = threshed.cell_data["region_id"]
    rgba = np.full((n_cells, 4), 220, dtype=np.uint8)  # default light gray
    rgba[:, 3] = 40  # mostly transparent

    cmap_tab = pv.LookupTable(cmap="tab20", n_values=20)
    for i in range(n_cells):
        rid = int(rids[i])
        if rid in pmi_region_set:
            c = cmap_tab.map_value((rid - 1) % 20)
            rgba[i, :3] = c[:3]
            rgba[i, 3] = 200

    threshed.cell_data["rgba"] = rgba

    pl = pv.Plotter()
    pl.set_background("white")
    pl.add_mesh(threshed, scalars="rgba", rgb=True, show_edges=False, opacity=1.0)

    # Compute model size for scaling offsets
    occ_indices = np.argwhere(occupancy)
    model_min = origin + np.min(occ_indices, axis=0) * pitch
    model_max = origin + np.max(occ_indices, axis=0) * pitch + pitch
    model_size = np.linalg.norm(model_max - model_min)

    # Annotation colors per category
    CATEGORY_COLORS = {
        "dimension": "#0066CC",
        "datum": "#CC0000",
        "geometric_tolerance": "#007700",
    }

    annotation_idx = 0
    for rec in pmi_records:
        rids = rec.get("linked_region_ids", [])
        if not rids:
            continue

        category = rec.get("category", "")
        color = CATEGORY_COLORS.get(category, "#333333")

        # Build annotation text
        label = rec.get("label") or rec.get("subtype", "")
        value = rec.get("value")
        unit = rec.get("unit", "")
        if category == "datum":
            # Datum flag style: just the letter
            datum_letter = ""
            lbl = rec.get("label", "")
            if lbl:
                parts = lbl.split()
                datum_letter = parts[-1] if parts else lbl
            annotation_text = f"[{datum_letter}]"
        elif value is not None:
            annotation_text = f"{value} {unit}".strip()
            if label:
                annotation_text = f"{label}\n{annotation_text}"
        else:
            annotation_text = label or rec["pmi_id"]

        # Compute face centroids for all linked regions
        face_centers = []
        face_normals = []
        for rid in rids:
            c = _region_centroid_world(rid, occupancy, region_grid, pitch, origin)
            n = _region_normal(rid, occ_regions)
            if c is not None:
                face_centers.append(c)
                face_normals.append(n)

        if not face_centers:
            continue

        if category == "dimension" and len(face_centers) >= 2:
            # -- Dimension: line between two faces with label at midpoint --
            p1 = face_centers[0]
            p2 = face_centers[1]
            midpoint = (p1 + p2) / 2.0

            # Choose an offset direction (perpendicular to the dimension line)
            dim_dir = p2 - p1
            dim_len = np.linalg.norm(dim_dir)
            if dim_len > 1e-9:
                dim_dir = dim_dir / dim_len

            # Find a perpendicular direction for the offset
            up = np.array([0, 0, 1.0])
            perp = np.cross(dim_dir, up)
            if np.linalg.norm(perp) < 1e-9:
                up = np.array([0, 1.0, 0])
                perp = np.cross(dim_dir, up)
            perp = perp / np.linalg.norm(perp)

            offset = perp * model_size * 0.25
            label_pos = midpoint + offset

            # Extension lines from faces to the offset plane
            ext1 = p1 + offset
            ext2 = p2 + offset

            # Draw extension lines (face to offset plane)
            ext_line1 = pv.Line(p1, ext1)
            ext_line2 = pv.Line(p2, ext2)
            pl.add_mesh(ext_line1, color=color, line_width=1.5, style="wireframe")
            pl.add_mesh(ext_line2, color=color, line_width=1.5, style="wireframe")

            # Draw dimension line (between extension line endpoints)
            dim_line = pv.Line(ext1, ext2)
            pl.add_mesh(dim_line, color=color, line_width=2.5, style="wireframe")

            # Arrowheads at both ends
            arrow_scale = model_size * 0.04
            arrow1 = pv.Arrow(start=ext1, direction=(ext2 - ext1), scale=arrow_scale,
                              tip_length=0.4, tip_radius=0.15, shaft_radius=0.0)
            arrow2 = pv.Arrow(start=ext2, direction=(ext1 - ext2), scale=arrow_scale,
                              tip_length=0.4, tip_radius=0.15, shaft_radius=0.0)
            pl.add_mesh(arrow1, color=color)
            pl.add_mesh(arrow2, color=color)

            # Label at midpoint
            pl.add_point_labels(
                pv.PolyData(label_pos.reshape(1, 3)),
                [annotation_text],
                font_size=14,
                text_color=color,
                bold=True,
                point_size=0,
                shape=None,
                render_points_as_spheres=False,
                always_visible=True,
            )

        elif category == "datum":
            # -- Datum flag: box label connected by leader line --
            face_center = face_centers[0]
            normal = face_normals[0]
            flag_pos = _offset_point(face_center, normal, model_size, factor=0.4)

            # Leader line
            leader = pv.Line(face_center, flag_pos)
            pl.add_mesh(leader, color=color, line_width=2.0, style="wireframe")

            # Small triangle at the face end
            tri_size = model_size * 0.02
            if normal is not None:
                tri_tip = face_center
                tri_base_center = face_center + normal * tri_size * 2
                # Create a small cone as the datum triangle
                cone = pv.Cone(center=(tri_tip + tri_base_center) / 2,
                               direction=normal, height=tri_size * 2,
                               radius=tri_size, resolution=3)
                pl.add_mesh(cone, color=color)

            # Datum flag label
            pl.add_point_labels(
                pv.PolyData(flag_pos.reshape(1, 3)),
                [annotation_text],
                font_size=18,
                text_color="white",
                bold=True,
                point_size=0,
                shape="rounded_rect",
                fill_shape=True,
                shape_color=color,
                shape_opacity=0.95,
                margin=5,
                always_visible=True,
            )

        else:
            # -- Generic PMI: label with leader line to face --
            face_center = face_centers[0]
            normal = face_normals[0]
            label_pos = _offset_point(face_center, normal, model_size, factor=0.35)

            leader = pv.Line(face_center, label_pos)
            pl.add_mesh(leader, color=color, line_width=1.5, style="wireframe")

            pl.add_point_labels(
                pv.PolyData(label_pos.reshape(1, 3)),
                [annotation_text],
                font_size=12,
                text_color=color,
                bold=True,
                point_size=0,
                shape="rounded_rect",
                fill_shape=True,
                shape_color="white",
                shape_opacity=0.85,
                margin=3,
                always_visible=True,
            )

        annotation_idx += 1

    if annotation_idx == 0:
        pl.add_text("No PMI with linked regions found", font_size=12, color="red")

    pl.add_axes()
    pl.add_text("MBD Annotation View", font_size=10, position="upper_left", color="black")
    pl.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D voxel viewer")
    parser.add_argument("npz_file", type=Path)
    parser.add_argument("--mode", choices=["occupancy", "regions", "pmi", "explode", "mbd"], default="regions")
    parser.add_argument("--json", type=Path, default=None, help="JSON sidecar (for pmi/explode modes)")
    args = parser.parse_args()

    if not args.npz_file.exists():
        raise SystemExit(f"NPZ file not found: {args.npz_file}")

    occupancy, region_grid, pitch, transform, bounds = load_npz(args.npz_file)
    print(f"Loaded: shape={occupancy.shape} pitch={pitch} occupied={np.count_nonzero(occupancy):,}")

    if args.mode == "occupancy":
        show_occupancy(occupancy, pitch, transform)

    elif args.mode == "regions":
        show_regions(occupancy, region_grid, pitch, transform)

    elif args.mode in ("pmi", "explode", "mbd"):
        json_path = args.json or args.npz_file.with_suffix(".json")
        if not json_path.exists():
            raise SystemExit(f"JSON sidecar not found: {json_path}")
        json_data = json.loads(json_path.read_text(encoding="utf-8"))

        if args.mode == "pmi":
            show_pmi(occupancy, region_grid, pitch, transform, json_data)
        elif args.mode == "explode":
            show_exploded(occupancy, region_grid, pitch, transform, json_data)
        else:
            show_mbd(occupancy, region_grid, pitch, transform, json_data)


if __name__ == "__main__":
    main()
