#!/usr/bin/env python3
"""
Pretty-print and interpret a voxel semantic JSON sidecar.

Usage:
    python inspect_json.py Square_Hole_voxel_semantic.json
    python inspect_json.py Square_Hole_voxel_semantic.json --full
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich import box


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def section_header(console: Console, title: str):
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    console.print()


def show_overview(console: Console, data: dict):
    section_header(console, "Overview")
    occ = data.get("occ_regions", [])
    sf = data.get("step_faces", [])
    pmi = data.get("pmi_records", [])
    r2v = data.get("region_to_voxel_count", {})

    matched_sf = sum(1 for f in sf if f.get("matched_occ_face_index") is not None)
    linked_pmi = sum(1 for r in pmi if r.get("linked_region_ids"))
    total_voxels = sum(int(v) for v in r2v.values())

    table = Table(box=box.SIMPLE_HEAVY, show_header=False, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("OCC face regions", str(len(occ)))
    table.add_row("STEP faces parsed", str(len(sf)))
    table.add_row("STEP faces matched to OCC", f"{matched_sf} / {len(sf)}")
    table.add_row("PMI records", str(len(pmi)))
    table.add_row("PMI with region links", f"{linked_pmi} / {len(pmi)}")
    table.add_row("Total region-labeled voxels", f"{total_voxels:,}")
    console.print(table)


def show_occ_regions(console: Console, data: dict, full: bool):
    section_header(console, "OCC Face Regions")
    regions = data.get("occ_regions", [])
    r2v = data.get("region_to_voxel_count", {})

    table = Table(box=box.ROUNDED, title="Reconstructed OCC face regions")
    table.add_column("Region ID", justify="center", style="bold yellow")
    table.add_column("Surface Type", style="green")
    table.add_column("Area", justify="right")
    table.add_column("Centroid", justify="center")
    table.add_column("Triangles", justify="right")
    table.add_column("Voxels", justify="right", style="magenta")

    if full:
        table.add_column("Direction")
        table.add_column("Radius", justify="right")
        table.add_column("BBox Min")
        table.add_column("BBox Max")

    for r in regions:
        idx = str(r["occ_face_index"])
        vcount = str(r2v.get(idx, 0))
        c = r.get("centroid")
        centroid_str = f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})" if c else "-"
        row = [
            idx,
            r["surface_type"],
            f"{r['area']:.1f}",
            centroid_str,
            str(r["triangle_count"]),
            vcount,
        ]
        if full:
            d = r.get("direction")
            dir_str = f"({d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f})" if d else "-"
            rad = f"{r['radius']:.3f}" if r.get("radius") is not None else "-"
            bmin = r.get("bbox_min")
            bmax = r.get("bbox_max")
            bmin_str = f"({bmin[0]:.2f}, {bmin[1]:.2f}, {bmin[2]:.2f})" if bmin else "-"
            bmax_str = f"({bmax[0]:.2f}, {bmax[1]:.2f}, {bmax[2]:.2f})" if bmax else "-"
            row.extend([dir_str, rad, bmin_str, bmax_str])
        table.add_row(*row)

    console.print(table)


def show_step_faces(console: Console, data: dict, full: bool):
    section_header(console, "STEP Face Matching")
    faces = data.get("step_faces", [])

    table = Table(box=box.ROUNDED, title="STEP ADVANCED_FACE -> OCC region matching")
    table.add_column("STEP Face ID", justify="center", style="bold")
    table.add_column("Surface Type", style="green")
    table.add_column("Matched OCC Region", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="center")
    table.add_column("Orientation", justify="center")

    if full:
        table.add_column("STEP Centroid")
        table.add_column("STEP Origin")

    for f in faces:
        match = f.get("matched_occ_face_index")
        score = f.get("match_score")
        match_str = str(match) if match is not None else "[red]unmatched[/red]"
        score_str = f"{score:.4f}" if score is not None else "-"

        if score is not None and score < 0.3:
            score_str = f"[green]{score_str}[/green]"
        elif score is not None:
            score_str = f"[yellow]{score_str}[/yellow]"

        row = [
            str(f["step_face_id"]),
            f["surface_type"],
            match_str,
            score_str,
            f.get("match_confidence", "-"),
            f.get("orientation_flag", "-"),
        ]
        if full:
            c = f.get("centroid")
            o = f.get("origin")
            c_str = f"({c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f})" if c else "-"
            o_str = f"({o[0]:.6f}, {o[1]:.6f}, {o[2]:.6f})" if o else "-"
            row.extend([c_str, o_str])
        table.add_row(*row)

    console.print(table)


def show_pmi(console: Console, data: dict):
    section_header(console, "PMI Records (Semantic Layer)")
    records = data.get("pmi_records", [])

    if not records:
        console.print("[dim]No PMI records found.[/dim]")
        return

    for rec in records:
        valid = rec.get("valid_association", False)
        status = "[bold green]LINKED[/bold green]" if valid else "[bold red]UNLINKED[/bold red]"
        category = rec.get("category", "unknown").upper()
        cat_color = {"DIMENSION": "cyan", "DATUM": "yellow", "GEOMETRIC_TOLERANCE": "magenta"}.get(category, "white")

        tree = Tree(
            f"[bold]{rec['pmi_id']}[/bold]  [{cat_color}]{category}[/{cat_color}]  {status}"
        )

        # Basic info
        info = tree.add("[dim]Info[/dim]")
        info.add(f"Subtype: {rec.get('subtype', '-')}")
        if rec.get("label"):
            info.add(f"Label: [bold]{rec['label']}[/bold]")
        if rec.get("value") is not None:
            unit = rec.get("unit", "")
            info.add(f"Value: [bold green]{rec['value']} {unit}[/bold green]")
        if rec.get("datum_refs"):
            info.add(f"Datum refs: {rec['datum_refs']}")
        if rec.get("modifiers"):
            info.add(f"Modifiers: {len(rec['modifiers'])}")

        # Association chain
        chain = tree.add("[dim]Association Chain[/dim]")
        chain.add(f"Method: {rec.get('association_method', '-')}")

        step_faces = rec.get("linked_step_face_ids", [])
        occ_faces = rec.get("linked_occ_face_indices", [])
        regions = rec.get("linked_region_ids", [])

        sf_style = "green" if step_faces else "red"
        of_style = "green" if occ_faces else "red"
        rg_style = "green" if regions else "red"

        chain.add(f"STEP faces:   [{sf_style}]{step_faces}[/{sf_style}]")
        chain.add(f"OCC faces:    [{of_style}]{occ_faces}[/{of_style}]")
        chain.add(f"Voxel regions:[{rg_style}]{regions}[/{rg_style}]")

        if rec.get("notes"):
            notes = tree.add("[dim]Notes[/dim]")
            for n in rec["notes"]:
                notes.add(f"[dim italic]{n}[/dim italic]")

        console.print(tree)
        console.print()


def show_chain_summary(console: Console, data: dict):
    """Show the full PMI -> STEP face -> OCC region -> voxel chain in one view."""
    section_header(console, "End-to-End Mapping Chain")
    records = data.get("pmi_records", [])
    r2v = data.get("region_to_voxel_count", {})

    if not records:
        console.print("[dim]No PMI records.[/dim]")
        return

    table = Table(box=box.HEAVY_EDGE, title="PMI -> STEP Face -> OCC Region -> Voxels")
    table.add_column("PMI", style="bold")
    table.add_column("Label / Value")
    table.add_column("STEP Faces", justify="center")
    table.add_column("OCC Regions", justify="center")
    table.add_column("Voxels Covered", justify="right", style="magenta")
    table.add_column("Status", justify="center")

    for rec in records:
        label = rec.get("label") or rec["pmi_id"]
        val = rec.get("value")
        unit = rec.get("unit", "")
        label_val = label
        if val is not None:
            label_val += f" = {val} {unit}"

        step_faces = rec.get("linked_step_face_ids", [])
        regions = rec.get("linked_region_ids", [])

        voxel_total = sum(int(r2v.get(str(r), 0)) for r in regions)

        if rec.get("valid_association"):
            status = "[green]OK[/green]"
        elif step_faces:
            status = "[yellow]PARTIAL[/yellow]"
        else:
            status = "[red]NONE[/red]"

        table.add_row(
            rec["pmi_id"],
            label_val,
            ", ".join(str(f) for f in step_faces) or "-",
            ", ".join(str(r) for r in regions) or "-",
            f"{voxel_total:,}" if voxel_total else "-",
            status,
        )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Inspect voxel semantic JSON sidecar")
    parser.add_argument("json_file", type=Path)
    parser.add_argument("--full", action="store_true", help="Show all fields including origins, directions, bboxes")
    args = parser.parse_args()

    if not args.json_file.exists():
        sys.exit(f"File not found: {args.json_file}")

    data = load(args.json_file)
    console = Console()

    console.print(Panel(
        f"[bold]Semantic Layer Inspector[/bold]\n{args.json_file}",
        border_style="blue",
    ))

    show_overview(console, data)
    show_occ_regions(console, data, args.full)
    show_step_faces(console, data, args.full)
    show_pmi(console, data)
    show_chain_summary(console, data)


if __name__ == "__main__":
    main()
