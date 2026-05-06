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
import math
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


def show_debug_mapping(
    occupancy: np.ndarray,
    region_grid: np.ndarray,
    pitch: float,
    transform: np.ndarray,
    json_data: dict,
):
    grid = make_voxel_grid(occupancy, pitch, transform)
    grid.cell_data["region_id"] = region_grid.flatten(order="F").astype(np.float32)
    threshed = grid.threshold(0.5, scalars="occupancy")

    region_to_step = {}
    for face in json_data.get("step_faces", []):
        rid = face.get("matched_occ_face_index")
        if rid is not None:
            region_to_step.setdefault(int(rid), []).append(face)

    region_to_pmi = {}
    for rec in json_data.get("pmi_records", []):
        for rid in rec.get("linked_region_ids", []):
            region_to_pmi.setdefault(int(rid), []).append(rec)

    pl = pv.Plotter()
    pl.add_mesh(
        threshed,
        scalars="region_id",
        cmap="tab20",
        show_edges=True,
        edge_color="gray",
        line_width=0.2,
        opacity=0.78,
        scalar_bar_args={"title": "OCC Region ID"},
    )

    labels = []
    points = []
    for rid in sorted(set(int(v) for v in np.unique(region_grid) if v > 0)):
        center = _region_centroid_world(rid, occupancy, region_grid, pitch, transform)
        if center is None:
            continue
        faces = region_to_step.get(rid, [])
        pmi = region_to_pmi.get(rid, [])
        face_bits = []
        for face in faces[:3]:
            score = face.get("match_score")
            conf = face.get("match_confidence", "?")
            score_txt = f"{score:.3f}" if score is not None else "-"
            face_bits.append(f"STEP #{face.get('step_face_id')} {conf} {score_txt}")
        pmi_bits = [rec.get("pmi_id", "?") for rec in pmi[:3]]
        label = f"R{rid}"
        if face_bits:
            label += "\n" + "\n".join(face_bits)
        if pmi_bits:
            label += "\nPMI: " + ", ".join(pmi_bits)
        points.append(center)
        labels.append(label)

    if points:
        pl.add_point_labels(
            pv.PolyData(np.asarray(points)),
            labels,
            font_size=9,
            text_color="black",
            point_size=0,
            shape="rounded_rect",
            fill_shape=True,
            shape_color="white",
            shape_opacity=0.82,
            margin=2,
            always_visible=False,
        )

    bad_pmi = [
        rec.get("pmi_id", "?")
        for rec in json_data.get("pmi_records", [])
        if not rec.get("valid_association", False)
    ]
    if bad_pmi:
        pl.add_text(f"Unmapped/ambiguous PMI: {', '.join(bad_pmi[:8])}", font_size=10, color="red", position="upper_left")
    else:
        pl.add_text("Debug Mapping | all PMI records marked valid", font_size=10, color="black", position="upper_left")
    pl.add_axes()
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


def _occ_region_lookup(occ_regions: list) -> Dict[int, dict]:
    return {int(r["occ_face_index"]): r for r in occ_regions}


def _dimension_display_geometry(
    rec: dict,
    face_centers: List[np.ndarray],
    face_normals: List[Optional[np.ndarray]],
    occ_regions: list,
    model_size: float,
) -> dict:
    """Display-only reconstruction. Does not affect semantic mapping confidence."""
    p1 = face_centers[0]
    p2 = face_centers[1]
    display_start = p1
    display_end = p2
    expected = None
    status = "approximate"
    reason = "dimension line uses mapped region centroids; AP242 witness/annotation geometry is not parsed."

    rids = rec.get("linked_region_ids", [])
    lookup = _occ_region_lookup(occ_regions)
    mapped = [lookup[rid] for rid in rids if rid in lookup]
    if len(mapped) >= 2 and mapped[0].get("surface_type") == "plane" and mapped[1].get("surface_type") == "plane":
        n1 = face_normals[0]
        n2 = face_normals[1]
        if n1 is not None and n2 is not None and abs(float(np.dot(n1, n2))) > 0.98:
            expected = n1.copy()
            if float(np.dot(p2 - p1, expected)) < 0:
                expected = -expected
            display_end = p1 + expected * float(np.dot(p2 - p1, expected))
            status = "valid"
            reason = "two parallel planar regions; display vector reconstructed along plane normal."
    elif any(r.get("surface_type") == "cylinder" for r in mapped):
        reason = "dimension references a cylinder; displayed approximately from mapped region centers."

    dim_vec = display_end - display_start
    dim_len = np.linalg.norm(dim_vec)
    if dim_len > 1e-9:
        dim_dir = dim_vec / dim_len
    else:
        dim_dir = np.array([1.0, 0.0, 0.0])

    up = np.array([0.0, 0.0, 1.0])
    perp = np.cross(dim_dir, up)
    if np.linalg.norm(perp) < 1e-9:
        up = np.array([0.0, 1.0, 0.0])
        perp = np.cross(dim_dir, up)
    perp = perp / np.linalg.norm(perp)
    offset = perp * model_size * 0.25

    angle = None
    if expected is not None:
        angle = float(math.degrees(math.acos(float(np.clip(abs(np.dot(dim_dir, expected)), -1.0, 1.0)))))

    return {
        "display_start": display_start,
        "display_end": display_end,
        "display_vector": dim_vec,
        "offset": offset,
        "label_pos": (display_start + display_end) / 2.0 + offset,
        "expected": expected,
        "status": status,
        "reason": reason,
        "angle_degrees": angle,
    }


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
            # -- Dimension: display-only reconstruction. For two parallel
            # planar faces, draw along the shared normal. Other cases remain
            # approximate and do not affect semantic mapping confidence.
            geom = _dimension_display_geometry(rec, face_centers, face_normals, occ_regions, model_size)
            p1 = face_centers[0]
            p2 = face_centers[1]
            ext1 = geom["display_start"] + geom["offset"]
            ext2 = geom["display_end"] + geom["offset"]
            label_pos = geom["label_pos"]

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

            if geom["expected"] is not None:
                expected_line = pv.Line(geom["display_start"], geom["display_start"] + geom["expected"] * model_size * 0.18)
                pl.add_mesh(expected_line, color="black", line_width=1.0, style="wireframe")

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


def _rec_regions(rec: dict) -> List[int]:
    return rec.get("linked_region_ids") or rec.get("mapped_voxel_regions") or rec.get("selected_voxel_region_ids") or []


def show_pmi_qa(
    occupancy: np.ndarray,
    region_grid: np.ndarray,
    pitch: float,
    transform: np.ndarray,
    json_data: dict,
    pmi_index: int = 0,
):
    records = json_data.get("pmi_records", [])
    if not records:
        print("No PMI records found.")
        show_regions(occupancy, region_grid, pitch, transform)
        return

    pmi_index = max(0, min(pmi_index, len(records) - 1))
    rec = records[pmi_index]
    rids = set(int(v) for v in _rec_regions(rec))

    grid = make_voxel_grid(occupancy, pitch, transform)
    selected = np.zeros_like(region_grid, dtype=np.float32)
    for rid in rids:
        selected[region_grid == rid] = 1.0
    grid.cell_data["selected_pmi"] = selected.flatten(order="F")
    grid.cell_data["occupancy_f"] = occupancy.flatten(order="F").astype(np.float32)
    threshed = grid.threshold(0.5, scalars="occupancy_f")

    pl = pv.Plotter()
    base = threshed.threshold(0.5, scalars="selected_pmi", invert=True)
    if base.n_cells > 0:
        pl.add_mesh(base, color="lightgray", opacity=0.12, show_edges=False)
    picked = threshed.threshold(0.5, scalars="selected_pmi")
    if picked.n_cells > 0:
        pl.add_mesh(picked, color="#0066CC", opacity=0.9, show_edges=True, edge_color="black", line_width=0.3)

    origin = transform[:3, 3]
    occ_regions = json_data.get("occ_regions", [])
    centers = []
    normals = []
    for rid in sorted(rids):
        c = _region_centroid_world(rid, occupancy, region_grid, pitch, origin)
        n = _region_normal(rid, occ_regions)
        if c is not None:
            centers.append(c)
            normals.append(n)
            pl.add_point_labels(pv.PolyData(c.reshape(1, 3)), [f"R{rid}"], font_size=12, point_size=0, always_visible=True)

    occ_indices = np.argwhere(occupancy)
    model_min = origin + np.min(occ_indices, axis=0) * pitch
    model_max = origin + np.max(occ_indices, axis=0) * pitch + pitch
    model_size = np.linalg.norm(model_max - model_min)
    if (rec.get("category") == "dimension" or rec.get("pmi_type") == "dimension") and len(centers) >= 2:
        geom = _dimension_display_geometry(rec, centers, normals, occ_regions, model_size)
        line = pv.Line(geom["display_start"] + geom["offset"], geom["display_end"] + geom["offset"])
        pl.add_mesh(line, color="#0066CC", line_width=3)
        if geom["expected"] is not None:
            pl.add_mesh(pv.Line(geom["display_start"], geom["display_start"] + geom["expected"] * model_size * 0.18), color="black", line_width=2)

    value = rec.get("value", rec.get("parsed_value", rec.get("tolerance_value")))
    unit = rec.get("unit", rec.get("parsed_units", ""))
    text = "\n".join(
        [
            f"PMI {pmi_index + 1}/{len(records)}: {rec.get('pmi_id')}",
            f"type: {rec.get('category', rec.get('pmi_type'))} / {rec.get('subtype', rec.get('pmi_subtype'))}",
            f"value: {value} {unit}".strip(),
            f"STEP faces: {rec.get('linked_step_face_ids', rec.get('referenced_step_faces', []))}",
            f"OCC regions: {rec.get('linked_occ_face_indices', rec.get('mapped_occ_regions', []))}",
            f"voxel regions: {sorted(rids)}",
            f"semantic: {rec.get('semantic_mapping_status', rec.get('confidence', '-'))}",
            f"display: {rec.get('display_status', '-')}",
            f"current-scope usable: {rec.get('usable_for_current_project_scope', rec.get('semantic_record_valid', rec.get('valid_association', '-')))}",
            "Use --pmi-index N to inspect another record.",
        ]
    )
    pl.add_text(text, font_size=9, position="upper_left", color="black")
    pl.add_axes()
    pl.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D voxel viewer")
    parser.add_argument("npz_file", type=Path)
    parser.add_argument("--mode", choices=["occupancy", "regions", "pmi", "explode", "mbd", "debug-mapping", "pmi-qa"], default="regions")
    parser.add_argument("--json", type=Path, default=None, help="JSON sidecar or diagnostics/QA report (for pmi/explode/debug-mapping/pmi-qa modes)")
    parser.add_argument("--pmi-index", type=int, default=0, help="PMI record index for --mode pmi-qa")
    args = parser.parse_args()

    if not args.npz_file.exists():
        raise SystemExit(f"NPZ file not found: {args.npz_file}")

    occupancy, region_grid, pitch, transform, bounds = load_npz(args.npz_file)
    print(f"Loaded: shape={occupancy.shape} pitch={pitch} occupied={np.count_nonzero(occupancy):,}")

    if args.mode == "occupancy":
        show_occupancy(occupancy, pitch, transform)

    elif args.mode == "regions":
        show_regions(occupancy, region_grid, pitch, transform)

    elif args.mode in ("pmi", "explode", "mbd", "debug-mapping", "pmi-qa"):
        json_path = args.json or args.npz_file.with_suffix(".json")
        if not json_path.exists():
            raise SystemExit(f"JSON sidecar not found: {json_path}")
        json_data = json.loads(json_path.read_text(encoding="utf-8"))

        if args.mode == "pmi":
            show_pmi(occupancy, region_grid, pitch, transform, json_data)
        elif args.mode == "explode":
            show_exploded(occupancy, region_grid, pitch, transform, json_data)
        elif args.mode == "mbd":
            show_mbd(occupancy, region_grid, pitch, transform, json_data)
        elif args.mode == "debug-mapping":
            show_debug_mapping(occupancy, region_grid, pitch, transform, json_data)
        else:
            show_pmi_qa(occupancy, region_grid, pitch, transform, json_data, args.pmi_index)


if __name__ == "__main__":
    main()
