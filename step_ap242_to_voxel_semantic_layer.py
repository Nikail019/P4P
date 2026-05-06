#!/usr/bin/env python3
"""
Read a STEP AP242 B-rep file, triangulate it with Open CASCADE,
voxelize the resulting mesh, and create a SECONDARY semantic layer that keeps
PMI/semantic records separate from the dense voxel occupancy grid.

This script intentionally does NOT embed PMI directly into voxel channels.
Instead it produces:
1) a voxel occupancy grid from the meshing route,
2) a boundary-region grid keyed by reconstructed OCC face regions, and
3) a JSON sidecar containing normalized PMI records and their mappings to
   source STEP ADVANCED_FACE ids, OCC face regions, and voxel regions.

What this script attempts:
- Geometry: STEP -> OCC shape -> triangulated mesh -> voxels
- Semantics: parse a subset of AP242 PMI directly from the STEP text
- Mapping: infer STEP ADVANCED_FACE -> OCC face-region correspondences using
  geometric signatures (best-effort for plane/cylinder-centric models)

Important limitations:
- AP242 PMI is vendor-dependent and often incomplete/inconsistent.
- The parser here is intentionally lightweight and focuses on common semantic
  PMI patterns (dimensions, dimensional tolerances, geometric tolerances,
  datum references) rather than the full AP242 schema.
- STEP ADVANCED_FACE to OCC face matching is heuristic. It is strongest for
  planar and cylindrical faces and may fail on repeated/symmetric faces or
  complex/freeform geometry.
- If the source STEP file does not actually contain semantic PMI, the semantic
  layer will simply be sparse or empty.

Dependencies:
    pip install pythonocc-core trimesh matplotlib numpy

Example:
    python step_ap242_to_voxel_semantic_layer.py part.step \
        --pitch 1.0 --fill --save-prefix part_semantic
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from OCC.Core.TopAbs import TopAbs_REVERSED

import numpy as np
import trimesh
import matplotlib.pyplot as plt

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GProp import GProp_GProps
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import pythonocc-core / Open CASCADE bindings.\n"
        "Install them first, for example:\n"
        "    pip install pythonocc-core\n\n"
        f"Import error: {exc}"
    )


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class OccFaceRegion:
    occ_face_index: int
    surface_type: str
    origin: Optional[Tuple[float, float, float]]
    direction: Optional[Tuple[float, float, float]]
    radius: Optional[float]
    area: float
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]
    centroid: Tuple[float, float, float]
    triangle_count: int


@dataclass
class StepFaceSignature:
    step_face_id: int
    surface_type: str
    origin: Optional[Tuple[float, float, float]]
    direction: Optional[Tuple[float, float, float]]
    radius: Optional[float]
    orientation_flag: Optional[str]
    centroid: Optional[Tuple[float, float, float]] = None
    area_estimate: Optional[float] = None
    matched_occ_face_index: Optional[int] = None
    match_score: Optional[float] = None


@dataclass
class PMIRecord:
    pmi_id: str
    category: str
    subtype: str
    label: Optional[str]
    value: Optional[float]
    unit: Optional[str]
    modifiers: List[str]
    datum_refs: List[str]
    source_entity_ids: List[int]
    linked_shape_aspect_ids: List[int]
    linked_step_face_ids: List[int]
    linked_occ_face_indices: List[int]
    linked_region_ids: List[int]
    valid_association: bool
    association_method: str
    notes: List[str]


# -----------------------------------------------------------------------------
# STEP reading / meshing / voxelization
# -----------------------------------------------------------------------------


def read_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Could not read STEP file: {step_path}")

    ok = reader.TransferRoots()
    if ok == 0:
        raise RuntimeError("STEP file loaded, but no shapes were transferred.")

    shape = reader.OneShape()
    if shape.IsNull():
        raise RuntimeError("Transferred shape is null.")
    return shape



def triangulate_shape(shape, linear_deflection: float, angular_deflection: float) -> None:
    mesher = BRepMesh_IncrementalMesh(
        shape,
        float(linear_deflection),
        False,
        float(angular_deflection),
        True,
    )
    mesher.Perform()



def _safe_direction_to_tuple(direction_obj) -> Tuple[float, float, float]:
    return (float(direction_obj.X()), float(direction_obj.Y()), float(direction_obj.Z()))



def _safe_point_to_tuple(point_obj) -> Tuple[float, float, float]:
    return (float(point_obj.X()), float(point_obj.Y()), float(point_obj.Z()))



def occ_shape_to_trimesh_with_regions(shape) -> Tuple[trimesh.Trimesh, np.ndarray, List[OccFaceRegion]]:
    """
    Extract triangle soup from OCC, while keeping a mapping from each triangle to
    its originating OCC face index and storing face-region geometric signatures.
    """
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    triangle_region_ids: List[int] = []
    face_regions: List[OccFaceRegion] = []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    vertex_offset = 0
    occ_face_index = 0

    while explorer.More():
        occ_face_index += 1
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation is None or triangulation.NbNodes() == 0:
            explorer.Next()
            continue

        transform = location.Transformation()
        nb_nodes = triangulation.NbNodes()
        face_vertices: List[List[float]] = []
        for i in range(1, nb_nodes + 1):
            pnt = triangulation.Node(i).Transformed(transform)
            face_vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

        fv = np.asarray(face_vertices, dtype=np.float64)
        if fv.size == 0:
            explorer.Next()
            continue

        vertex_offset = len(vertices)
        vertices.extend(face_vertices)

        face_orientation = face.Orientation()
        reversed_face = face_orientation == TopAbs_REVERSED
        triangle_count = 0

        nb_tri = triangulation.NbTriangles()
        for i in range(1, nb_tri + 1):
            tri = triangulation.Triangle(i)
            n1, n2, n3 = tri.Get()
            a = vertex_offset + (n1 - 1)
            b = vertex_offset + (n2 - 1)
            c = vertex_offset + (n3 - 1)
            if reversed_face:
                faces.append([a, c, b])
            else:
                faces.append([a, b, c])
            triangle_region_ids.append(occ_face_index)
            triangle_count += 1

     

        area_props = GProp_GProps()
        try:
            brepgprop.SurfaceProperties(face, area_props)
            area = float(area_props.Mass())
        except Exception:
            area = 0.0

        bbox_min = tuple(np.min(fv, axis=0).tolist())
        bbox_max = tuple(np.max(fv, axis=0).tolist())
        centroid = tuple(np.mean(fv, axis=0).tolist())

        surface_type = "other"
        origin = None
        direction = None
        radius = None
        try:
            adaptor = BRepAdaptor_Surface(face, True)
            stype = adaptor.GetType()
            if stype == GeomAbs_Plane:
                plane = adaptor.Plane()
                surface_type = "plane"
                origin = _safe_point_to_tuple(plane.Location())
                direction = _safe_direction_to_tuple(plane.Axis().Direction())
            elif stype == GeomAbs_Cylinder:
                cyl = adaptor.Cylinder()
                surface_type = "cylinder"
                origin = _safe_point_to_tuple(cyl.Location())
                direction = _safe_direction_to_tuple(cyl.Axis().Direction())
                radius = float(cyl.Radius())
        except Exception:
            pass

        face_regions.append(
            OccFaceRegion(
                occ_face_index=occ_face_index,
                surface_type=surface_type,
                origin=origin,
                direction=direction,
                radius=radius,
                area=area,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                centroid=centroid,
                triangle_count=triangle_count,
            )
        )
        explorer.Next()

    if not vertices or not faces:
        raise RuntimeError(
            "No triangulated faces were extracted. "
            "Try increasing the meshing quality (smaller deflection)."
        )

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
        validate=True,
    )
    return mesh, np.asarray(triangle_region_ids, dtype=np.int32), face_regions



def voxelize_mesh(mesh: trimesh.Trimesh, pitch: float, fill: bool = False) -> trimesh.voxel.VoxelGrid:
    if pitch <= 0:
        raise ValueError("Voxel pitch must be > 0.")
    vox = mesh.voxelized(pitch)
    if fill:
        vox = vox.fill(method="base")
    return vox


# -----------------------------------------------------------------------------
# Lightweight STEP Part 21 parsing utilities
# -----------------------------------------------------------------------------

_ENTITY_RE = re.compile(r"#(\d+)\s*=\s*(.*?);", re.DOTALL)
_REF_RE = re.compile(r"#(\d+)")
_FLOAT_RE = re.compile(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?")



def strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)



def parse_step_entities(step_path: Path) -> Dict[int, str]:
    text = strip_comments(step_path.read_text(encoding="utf-8", errors="ignore"))
    entities: Dict[int, str] = {}
    for m in _ENTITY_RE.finditer(text):
        entities[int(m.group(1))] = m.group(2).strip()
    return entities



def entity_refs(raw: str) -> List[int]:
    return [int(v) for v in _REF_RE.findall(raw)]



def entity_contains(raw: Optional[str], token: str) -> bool:
    return raw is not None and token in raw



def entity_head_type(raw: str) -> Optional[str]:
    raw = raw.strip()
    m = re.match(r"\(?\s*([A-Z0-9_]+)\s*\(", raw)
    if not m:
        return None
    return m.group(1)



def split_top_level_args(s: str) -> List[str]:
    args: List[str] = []
    current: List[str] = []
    depth = 0
    in_string = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            in_string = not in_string
            current.append(ch)
        elif not in_string and ch == '(':
            depth += 1
            current.append(ch)
        elif not in_string and ch == ')':
            depth -= 1
            current.append(ch)
        elif not in_string and ch == ',' and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
        i += 1
    if current:
        args.append("".join(current).strip())
    return args



def parse_simple_entity(raw: str) -> Tuple[Optional[str], List[str]]:
    raw = raw.strip()
    m = re.match(r"([A-Z0-9_]+)\((.*)\)$", raw, flags=re.DOTALL)
    if not m:
        return None, []
    return m.group(1), split_top_level_args(m.group(2).strip())



def parse_tuple_floats(raw: str) -> Optional[Tuple[float, ...]]:
    m = re.search(r"\(([^()]*)\)\s*\)*\s*$", raw.strip())
    if not m:
        return None
    nums = _FLOAT_RE.findall(m.group(1))
    if not nums:
        return None
    return tuple(float(v) for v in nums)



def parse_cartesian_point(entity_id: int, entities: Dict[int, str]) -> Optional[Tuple[float, float, float]]:
    raw = entities.get(entity_id)
    if raw is None or "CARTESIAN_POINT" not in raw:
        return None
    vals = parse_tuple_floats(raw)
    if vals is None or len(vals) < 3:
        return None
    return (float(vals[0]), float(vals[1]), float(vals[2]))



def parse_direction(entity_id: int, entities: Dict[int, str]) -> Optional[Tuple[float, float, float]]:
    raw = entities.get(entity_id)
    if raw is None or "DIRECTION" not in raw:
        return None
    vals = parse_tuple_floats(raw)
    if vals is None or len(vals) < 3:
        return None
    vec = np.asarray(vals[:3], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm <= 0:
        return None
    vec = vec / norm
    return (float(vec[0]), float(vec[1]), float(vec[2]))



def parse_axis2_placement(entity_id: int, entities: Dict[int, str]) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    raw = entities.get(entity_id)
    if raw is None or "AXIS2_PLACEMENT_3D" not in raw:
        return None, None
    _, args = parse_simple_entity(raw)
    refs = [int(v[1:]) for v in args if v.startswith("#")]
    if not refs:
        return None, None
    point = parse_cartesian_point(refs[0], entities)
    axis = parse_direction(refs[1], entities) if len(refs) >= 2 else None
    return point, axis



def parse_surface_signature(surface_entity_id: int, entities: Dict[int, str]) -> Tuple[str, Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[float]]:
    raw = entities.get(surface_entity_id)
    if raw is None:
        return "other", None, None, None

    if "PLANE" in raw:
        _, args = parse_simple_entity(raw)
        refs = [int(v[1:]) for v in args if v.startswith("#")]
        axis_id = refs[-1] if refs else None
        origin, direction = parse_axis2_placement(axis_id, entities) if axis_id is not None else (None, None)
        return "plane", origin, direction, None

    if "CYLINDRICAL_SURFACE" in raw:
        _, args = parse_simple_entity(raw)
        refs = [int(v[1:]) for v in args if v.startswith("#")]
        axis_id = refs[-1] if refs else None
        origin, direction = parse_axis2_placement(axis_id, entities) if axis_id is not None else (None, None)
        nums = [float(v) for v in _FLOAT_RE.findall(raw)]
        radius = nums[-1] if nums else None
        return "cylinder", origin, direction, radius

    return "other", None, None, None



def compute_step_face_centroid(face_entity_id: int, entities: Dict[int, str]) -> Optional[Tuple[float, float, float]]:
    """Walk ADVANCED_FACE -> FACE_BOUND -> EDGE_LOOP -> vertices to compute centroid."""
    raw = entities.get(face_entity_id)
    if raw is None:
        return None

    points: List[Tuple[float, float, float]] = []
    seen: set = set()

    # Topology keywords that form the face-boundary traversal path
    _TOPO_KEYS = (
        "ADVANCED_FACE(", "FACE_BOUND(", "FACE_OUTER_BOUND(",
        "EDGE_LOOP(", "ORIENTED_EDGE(", "EDGE_CURVE(",
        "VERTEX_POINT(",
    )

    queue: List[Tuple[int, int]] = [(face_entity_id, 0)]
    while queue:
        eid, depth = queue.pop(0)
        if eid in seen or depth > 7:
            continue
        seen.add(eid)
        eraw = entities.get(eid, "")

        if "CARTESIAN_POINT(" in eraw and depth >= 2:
            pt = parse_cartesian_point(eid, entities)
            if pt is not None:
                points.append(pt)
            continue

        if any(k in eraw for k in _TOPO_KEYS):
            for ref in entity_refs(eraw):
                if ref not in seen:
                    queue.append((ref, depth + 1))

    if not points:
        return None
    arr = np.array(points, dtype=np.float64)
    unique = np.unique(arr, axis=0)
    if len(unique) == 0:
        return None
    return tuple(np.mean(unique, axis=0).tolist())



def extract_step_face_signatures(entities: Dict[int, str]) -> Dict[int, StepFaceSignature]:
    step_faces: Dict[int, StepFaceSignature] = {}
    for entity_id, raw in entities.items():
        if "ADVANCED_FACE(" not in raw:
            continue
        _, args = parse_simple_entity(raw)
        if len(args) < 3:
            continue
        surface_ref = args[2].strip()
        if not surface_ref.startswith("#"):
            continue
        surface_entity_id = int(surface_ref[1:])
        surface_type, origin, direction, radius = parse_surface_signature(surface_entity_id, entities)
        orientation_flag = args[3].strip() if len(args) >= 4 else None
        centroid = compute_step_face_centroid(entity_id, entities)
        step_faces[entity_id] = StepFaceSignature(
            step_face_id=entity_id,
            surface_type=surface_type,
            origin=origin,
            direction=direction,
            radius=radius,
            orientation_flag=orientation_flag,
            centroid=centroid,
        )
    return step_faces


# -----------------------------------------------------------------------------
# PMI extraction from STEP entities
# -----------------------------------------------------------------------------


def collect_measure_info(start_ids: Iterable[int], entities: Dict[int, str], max_depth: int = 4) -> Tuple[Optional[float], Optional[str], List[str]]:
    """Best-effort extraction of one value, one unit, and textual modifiers near an entity subtree."""
    value = None
    unit = None
    modifiers: List[str] = []

    queue = [(sid, 0) for sid in start_ids]
    seen = set()
    while queue:
        eid, depth = queue.pop(0)
        if eid in seen or depth > max_depth:
            continue
        seen.add(eid)
        raw = entities.get(eid)
        if raw is None:
            continue

        if value is None:
            m = re.search(r"(?:POSITIVE_LENGTH_MEASURE|LENGTH_MEASURE|PLANE_ANGLE_MEASURE|NUMERIC_MEASURE)\(([-+0-9.Ee]+)\)", raw)
            if m:
                value = float(m.group(1))

        if unit is None:
            if "SI_UNIT(" in raw:
                if ".MILLI." in raw and ".METRE." in raw:
                    unit = "mm"
                elif ".METRE." in raw:
                    unit = "m"
                elif ".RADIAN." in raw:
                    unit = "rad"
                elif ".DEGREE" in raw or ".DEGREE_CELSIUS" in raw:
                    unit = "deg"
            elif "NAMED_UNIT" in raw and "DEGREE" in raw:
                unit = "deg"

        if "MODIFIER" in raw or "LIMITS_AND_FITS" in raw or "PLUS_MINUS_TOLERANCE" in raw:
            modifiers.append(raw)

        for ref in entity_refs(raw):
            if ref not in seen:
                queue.append((ref, depth + 1))
    return value, unit, modifiers



def build_shape_aspect_to_step_faces(entities: Dict[int, str]) -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = defaultdict(list)
    for _eid, raw in entities.items():
        if "GEOMETRIC_ITEM_SPECIFIC_USAGE(" not in raw:
            continue
        refs = entity_refs(raw)
        if len(refs) < 2:
            continue
        shape_aspect_id = refs[0]
        target_face_id = refs[-1]
        if entity_contains(entities.get(target_face_id), "ADVANCED_FACE("):
            mapping[shape_aspect_id].append(target_face_id)
    return {k: sorted(set(v)) for k, v in mapping.items()}



def extract_dimensions(entities: Dict[int, str], shape_aspect_to_faces: Dict[int, List[int]]) -> List[PMIRecord]:
    records: List[PMIRecord] = []
    for entity_id, raw in entities.items():
        if "DIMENSIONAL_CHARACTERISTIC_REPRESENTATION(" not in raw:
            continue
        refs = entity_refs(raw)
        dim_loc_id = next((r for r in refs if entity_contains(entities.get(r), "DIMENSIONAL_")), None)
        shape_dim_id = next((r for r in refs if entity_contains(entities.get(r), "SHAPE_DIMENSION_REPRESENTATION(")), None)
        if dim_loc_id is None:
            continue

        dim_loc_raw = entities.get(dim_loc_id, "")
        label_match = re.search(r"DIMENSIONAL_[A-Z_]+\('([^']*)'", dim_loc_raw)
        label = label_match.group(1) if label_match else None
        linked_shape_aspects = entity_refs(dim_loc_raw)
        linked_step_face_ids = []
        for sid in linked_shape_aspects:
            linked_step_face_ids.extend(shape_aspect_to_faces.get(sid, []))
        linked_step_face_ids = sorted(set(linked_step_face_ids))

        value, unit, modifiers = collect_measure_info([shape_dim_id] if shape_dim_id else [dim_loc_id], entities)

        records.append(
            PMIRecord(
                pmi_id=f"dimension_{entity_id}",
                category="dimension",
                subtype=entity_head_type(dim_loc_raw) or "dimension",
                label=label,
                value=value,
                unit=unit,
                modifiers=modifiers,
                datum_refs=[],
                source_entity_ids=sorted(set([entity_id, dim_loc_id] + ([shape_dim_id] if shape_dim_id else []))),
                linked_shape_aspect_ids=linked_shape_aspects,
                linked_step_face_ids=linked_step_face_ids,
                linked_occ_face_indices=[],
                linked_region_ids=[],
                valid_association=bool(linked_step_face_ids),
                association_method="direct_shape_aspect_lookup" if linked_step_face_ids else "unresolved",
                notes=[],
            )
        )
    return records



def extract_geometric_tolerances(entities: Dict[int, str], shape_aspect_to_faces: Dict[int, List[int]]) -> List[PMIRecord]:
    records: List[PMIRecord] = []
    tolerance_keywords = (
        "STRAIGHTNESS_TOLERANCE(",
        "FLATNESS_TOLERANCE(",
        "ROUNDNESS_TOLERANCE(",
        "CYLINDRICITY_TOLERANCE(",
        "POSITION_TOLERANCE(",
        "PARALLELISM_TOLERANCE(",
        "PERPENDICULARITY_TOLERANCE(",
        "ANGULARITY_TOLERANCE(",
        "PROFILE_OF_A_LINE_TOLERANCE(",
        "PROFILE_OF_A_SURFACE_TOLERANCE(",
        "CIRCULAR_RUNOUT_TOLERANCE(",
        "TOTAL_RUNOUT_TOLERANCE(",
        "GEOMETRIC_TOLERANCE(",
    )
    seen_root_ids = set()

    for entity_id, raw in entities.items():
        if not any(k in raw for k in tolerance_keywords):
            continue
        if entity_id in seen_root_ids:
            continue
        seen_root_ids.add(entity_id)

        refs = entity_refs(raw)
        linked_shape_aspects = [r for r in refs if entity_contains(entities.get(r), "SHAPE_ASPECT(")]
        linked_step_face_ids: List[int] = []
        for sid in linked_shape_aspects:
            linked_step_face_ids.extend(shape_aspect_to_faces.get(sid, []))
        linked_step_face_ids = sorted(set(linked_step_face_ids))

        value, unit, modifiers = collect_measure_info([entity_id], entities)

        datum_refs: List[str] = []
        subtree_refs = set(refs)
        for rid in refs:
            subtree_refs.update(entity_refs(entities.get(rid, "")))
        for rid in subtree_refs:
            raw_r = entities.get(rid, "")
            if "DATUM_REFERENCE_COMPARTMENT(" in raw_r or entity_contains(raw_r, "DATUM("):
                labels = re.findall(r"'([^']+)'", raw_r)
                datum_refs.extend([lab for lab in labels if lab])

        head = entity_head_type(raw) or "geometric_tolerance"
        records.append(
            PMIRecord(
                pmi_id=f"gtol_{entity_id}",
                category="geometric_tolerance",
                subtype=head.lower(),
                label=None,
                value=value,
                unit=unit,
                modifiers=modifiers,
                datum_refs=sorted(set(datum_refs)),
                source_entity_ids=[entity_id],
                linked_shape_aspect_ids=linked_shape_aspects,
                linked_step_face_ids=linked_step_face_ids,
                linked_occ_face_indices=[],
                linked_region_ids=[],
                valid_association=bool(linked_step_face_ids),
                association_method="direct_shape_aspect_lookup" if linked_step_face_ids else "unresolved",
                notes=[],
            )
        )
    return records



def extract_standalone_datums(entities: Dict[int, str], shape_aspect_to_faces: Dict[int, List[int]]) -> List[PMIRecord]:
    records: List[PMIRecord] = []
    for entity_id, raw in entities.items():
        head = entity_head_type(raw)
        if head not in {"DATUM", "DATUM_FEATURE", "DATUM_TARGET"}:
            continue
        labels = re.findall(r"'([^']+)'", raw)
        label = labels[0] if labels else None

        # A DATUM/DATUM_FEATURE is itself a SHAPE_ASPECT — look up its face links
        linked_step_face_ids: List[int] = list(shape_aspect_to_faces.get(entity_id, []))
        linked_shape_aspect_ids: List[int] = [entity_id]

        # Also chase direct #refs that point to other SHAPE_ASPECTs
        refs = entity_refs(raw)
        for ref in refs:
            ref_raw = entities.get(ref, "")
            if "SHAPE_ASPECT(" in ref_raw:
                linked_shape_aspect_ids.append(ref)
                linked_step_face_ids.extend(shape_aspect_to_faces.get(ref, []))

        # Check SHAPE_ASPECT_RELATIONSHIP where this datum is related_element
        for _eid, sar_raw in entities.items():
            if "SHAPE_ASPECT_RELATIONSHIP(" not in sar_raw:
                continue
            sar_refs = entity_refs(sar_raw)
            if entity_id in sar_refs:
                for r in sar_refs:
                    if r != entity_id:
                        linked_step_face_ids.extend(shape_aspect_to_faces.get(r, []))
                        ref_raw2 = entities.get(r, "")
                        if "SHAPE_ASPECT(" in ref_raw2 or "COMPOSITE_SHAPE_ASPECT(" in ref_raw2:
                            linked_shape_aspect_ids.append(r)

        linked_step_face_ids = sorted(set(linked_step_face_ids))
        linked_shape_aspect_ids = sorted(set(linked_shape_aspect_ids))

        records.append(
            PMIRecord(
                pmi_id=f"datum_{entity_id}",
                category="datum",
                subtype=head.lower(),
                label=label,
                value=None,
                unit=None,
                modifiers=[],
                datum_refs=[],
                source_entity_ids=[entity_id],
                linked_shape_aspect_ids=linked_shape_aspect_ids,
                linked_step_face_ids=linked_step_face_ids,
                linked_occ_face_indices=[],
                linked_region_ids=[],
                valid_association=bool(linked_step_face_ids),
                association_method="direct_shape_aspect_lookup" if linked_step_face_ids else "none",
                notes=[],
            )
        )
    return records


# -----------------------------------------------------------------------------
# STEP face <-> OCC face-region matching
# -----------------------------------------------------------------------------


def _norm_vec(v: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(arr)
    if n <= 0:
        return None
    return arr / n



def angle_between_dirs(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    va = _norm_vec(a)
    vb = _norm_vec(b)
    if va is None or vb is None:
        return math.pi
    dot = float(np.clip(np.abs(np.dot(va, vb)), -1.0, 1.0))
    return float(math.acos(dot))



def point_to_plane_distance(point: Optional[Sequence[float]], plane_origin: Optional[Sequence[float]], plane_normal: Optional[Sequence[float]]) -> float:
    if point is None or plane_origin is None or plane_normal is None:
        return 1.0e9
    p = np.asarray(point, dtype=np.float64)
    o = np.asarray(plane_origin, dtype=np.float64)
    n = _norm_vec(plane_normal)
    if n is None:
        return 1.0e9
    return float(abs(np.dot(p - o, n)))



def point_to_axis_distance(point: Optional[Sequence[float]], axis_origin: Optional[Sequence[float]], axis_dir: Optional[Sequence[float]]) -> float:
    if point is None or axis_origin is None or axis_dir is None:
        return 1.0e9
    p = np.asarray(point, dtype=np.float64)
    o = np.asarray(axis_origin, dtype=np.float64)
    d = _norm_vec(axis_dir)
    if d is None:
        return 1.0e9
    v = p - o
    perp = v - np.dot(v, d) * d
    return float(np.linalg.norm(perp))



def _centroid_distance(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if a is None or b is None:
        return 1.0e9
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))



def _bbox_diagonal(occ: OccFaceRegion) -> float:
    return float(np.linalg.norm(np.asarray(occ.bbox_max) - np.asarray(occ.bbox_min)))



def match_step_faces_to_occ_regions(step_faces: Dict[int, StepFaceSignature], occ_regions: List[OccFaceRegion]) -> Dict[int, int]:
    """
    Best-effort matching using centroid distance as the primary discriminator,
    with surface type filtering and direction angle as secondary checks.

    Returns
    -------
    dict
        step_face_id -> occ_face_index for matched pairs.
    """
    matches: Dict[int, int] = {}
    used_occ: set = set()

    # Pre-compute a characteristic length for relative thresholding
    all_diags = [_bbox_diagonal(r) for r in occ_regions if _bbox_diagonal(r) > 0]
    char_length = float(np.median(all_diags)) if all_diags else 1.0

    candidates: List[Tuple[float, int, int]] = []
    for step_face_id, step_sig in step_faces.items():
        for occ in occ_regions:
            # Surface type must agree (or both "other")
            if step_sig.surface_type != occ.surface_type:
                continue

            # Direction filter: reject clearly misaligned faces early
            dir_angle = angle_between_dirs(step_sig.direction, occ.direction)
            if step_sig.direction is not None and occ.direction is not None:
                if dir_angle > 0.35:  # ~20 degrees tolerance
                    continue

            # Primary score: centroid-to-centroid distance (normalized)
            cdist = _centroid_distance(step_sig.centroid, occ.centroid)
            score = cdist / max(char_length, 1.0e-9)

            # Secondary: small direction penalty
            score += 0.2 * dir_angle

            # For cylinders: penalize radius mismatch
            if step_sig.surface_type == "cylinder":
                if step_sig.radius is not None and occ.radius is not None:
                    rdiff = abs(step_sig.radius - occ.radius)
                    score += rdiff / max(char_length, 1.0e-9)

            # For planes without centroid, fall back to plane-offset comparison
            if step_sig.centroid is None and step_sig.surface_type == "plane":
                score = dir_angle + 0.1 * point_to_plane_distance(
                    occ.centroid, step_sig.origin, step_sig.direction
                )

            candidates.append((score, step_face_id, occ.occ_face_index))

    candidates.sort(key=lambda x: x[0])
    for score, step_face_id, occ_face_index in candidates:
        if step_face_id in matches or occ_face_index in used_occ:
            continue
        # Reject matches that are clearly too far (> 1.5x characteristic length)
        if score > 1.5:
            continue
        matches[step_face_id] = occ_face_index
        used_occ.add(occ_face_index)
        step_faces[step_face_id].matched_occ_face_index = occ_face_index
        step_faces[step_face_id].match_score = score

    return matches


# -----------------------------------------------------------------------------
# Voxel secondary layer creation
# -----------------------------------------------------------------------------


def sample_triangle_points(a: np.ndarray, b: np.ndarray, c: np.ndarray, pitch: float) -> np.ndarray:
    """Adaptive barycentric sampling to cover boundary voxels from one triangle."""
    edges = [np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)]
    max_edge = max(edges)
    n = max(1, int(math.ceil(max_edge / max(pitch * 0.75, 1.0e-9))))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            u = i / n
            v = j / n
            w = 1.0 - u - v
            pts.append(w * a + u * b + v * c)
    return np.asarray(pts, dtype=np.float64)



def build_region_id_grid(
    vox: trimesh.voxel.VoxelGrid,
    mesh: trimesh.Trimesh,
    triangle_region_ids: np.ndarray,
    region_count: int,
    pitch: float,
) -> Tuple[np.ndarray, Dict[int, List[Tuple[int, int, int]]]]:
    """
    Build a boundary-region voxel grid where each occupied voxel is optionally
    labeled with the dominant OCC face-region sampled on that voxel.
    """
    vote_map: Dict[Tuple[int, int, int], Counter] = defaultdict(Counter)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.faces, dtype=np.int64)

    for tri_idx, tri in enumerate(tris):
        rid = int(triangle_region_ids[tri_idx])
        a, b, c = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        pts = sample_triangle_points(a, b, c, pitch)
        try:
            idx = vox.points_to_indices(pts)
        except Exception:
            continue
        if idx is None or len(idx) == 0:
            continue
        idx = np.asarray(idx, dtype=np.int64)
        for ix, iy, iz in idx:
            if (
                0 <= ix < vox.matrix.shape[0]
                and 0 <= iy < vox.matrix.shape[1]
                and 0 <= iz < vox.matrix.shape[2]
                and bool(vox.matrix[ix, iy, iz])
            ):
                vote_map[(int(ix), int(iy), int(iz))][rid] += 1

    region_grid = np.zeros(vox.matrix.shape, dtype=np.int32)
    region_to_voxels: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for voxel_idx, counter in vote_map.items():
        rid, _count = counter.most_common(1)[0]
        region_grid[voxel_idx] = rid
        region_to_voxels[rid].append(voxel_idx)

    return region_grid, dict(region_to_voxels)



def annotate_pmi_records(
    records: List[PMIRecord],
    step_to_occ: Dict[int, int],
) -> None:
    for rec in records:
        occ_ids = sorted({step_to_occ[sid] for sid in rec.linked_step_face_ids if sid in step_to_occ})
        rec.linked_occ_face_indices = occ_ids
        rec.linked_region_ids = occ_ids[:]  # region ids are OCC face ids in this script
        if rec.linked_step_face_ids and not occ_ids:
            rec.notes.append("Step-face association found, but no OCC face-region match succeeded.")
            rec.association_method = "step_face_found_occ_match_failed"
            rec.valid_association = False
        elif occ_ids:
            rec.association_method = "step_face_to_occ_face_signature_match"
            rec.valid_association = True


# -----------------------------------------------------------------------------
# Output / visualization
# -----------------------------------------------------------------------------


def save_outputs(
    save_prefix: Path,
    vox: trimesh.voxel.VoxelGrid,
    region_grid: np.ndarray,
    occ_regions: List[OccFaceRegion],
    step_faces: Dict[int, StepFaceSignature],
    pmi_records: List[PMIRecord],
    region_to_voxels: Dict[int, List[Tuple[int, int, int]]],
) -> Tuple[Path, Path]:
    npz_path = save_prefix.with_suffix(".npz")
    json_path = save_prefix.with_suffix(".json")

    pitch_value = float(np.asarray(vox.pitch).reshape(-1)[0])
    np.savez_compressed(
        npz_path,
        occupancy=vox.matrix.astype(np.uint8),
        region_id_grid=region_grid.astype(np.int32),
        transform=np.asarray(vox.transform),
        pitch=pitch_value,
        bounds=np.asarray(vox.bounds, dtype=np.float64),
    )

    payload = {
        "metadata": {
            "format": "step_ap242_voxel_semantic_layer_v1",
            "notes": [
                "occupancy grid stores geometry",
                "region_id_grid stores best-effort boundary face-region labels",
                "PMI remains in a sidecar semantic layer rather than dense voxel channels",
            ],
        },
        "occ_regions": [asdict(r) for r in occ_regions],
        "step_faces": [asdict(v) for v in step_faces.values()],
        "region_to_voxel_count": {str(k): len(v) for k, v in region_to_voxels.items()},
        "pmi_records": [asdict(r) for r in pmi_records],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return npz_path, json_path



def visualize_voxels_matplotlib(vox: trimesh.voxel.VoxelGrid, region_grid: Optional[np.ndarray] = None, max_dim: int = 160) -> None:
    grid = np.asarray(vox.matrix, dtype=bool)
    shape = np.array(grid.shape)
    if shape.max() > max_dim:
        raise RuntimeError(
            f"Voxel grid is too dense for a quick matplotlib preview: shape={tuple(shape)}. "
            f"Increase --pitch or reduce model size."
        )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    if region_grid is None:
        ax.voxels(grid, edgecolor="k", linewidth=0.03)
        ax.set_title(f"Voxelized STEP model | shape={tuple(grid.shape)} | pitch={float(vox.pitch):g}")
    else:
        colors = np.empty(grid.shape, dtype=object)
        labels = np.asarray(region_grid, dtype=np.int32)
        occ = np.argwhere(grid)
        unique_regions = sorted(set(int(labels[tuple(idx)]) for idx in occ if int(labels[tuple(idx)]) > 0))
        cmap = plt.get_cmap("tab20")
        for idx in occ:
            rid = int(labels[tuple(idx)])
            if rid > 0:
                colors[tuple(idx)] = cmap((rid - 1) % 20)
            else:
                colors[tuple(idx)] = (0.8, 0.8, 0.8, 1.0)
        ax.voxels(grid, facecolors=colors, edgecolor="k", linewidth=0.02)
        ax.set_title(f"Voxelized STEP model + boundary region layer | regions={len(unique_regions)}")
    ax.set_xlabel("X index")
    ax.set_ylabel("Y index")
    ax.set_zlabel("Z index")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Voxelize STEP AP242 geometry via meshing and produce a SECONDARY semantic layer "
            "mapping PMI/semantics to voxel regions."
        )
    )
    parser.add_argument("step_file", type=Path, help="Input STEP / STP file")
    parser.add_argument("--pitch", type=float, default=1.0, help="Voxel side length in model units (default: 1.0)")
    parser.add_argument("--linear-deflection", type=float, default=0.1, help="OCC meshing linear deflection (default: 0.1)")
    parser.add_argument("--angular-deflection", type=float, default=0.3, help="OCC meshing angular deflection in radians (default: 0.3)")
    parser.add_argument("--fill", action="store_true", help="Fill watertight voxel shells into solids")
    parser.add_argument("--save-prefix", type=Path, default=None, help="Output prefix for .npz and .json files")
    parser.add_argument("--no-preview", action="store_true", help="Skip matplotlib preview")
    return parser



def main() -> None:
    args = build_argparser().parse_args()
    step_path: Path = args.step_file
    if not step_path.exists():
        raise SystemExit(f"STEP file not found: {step_path}")

    save_prefix = args.save_prefix or step_path.with_name(step_path.stem + "_voxel_semantic")

    print(f"[1/7] Reading STEP shape: {step_path}")
    shape = read_step_shape(step_path)

    print("[2/7] Triangulating B-rep...")
    triangulate_shape(shape, args.linear_deflection, args.angular_deflection)

    print("[3/7] Extracting triangle mesh + OCC face regions...")
    mesh, triangle_region_ids, occ_regions = occ_shape_to_trimesh_with_regions(shape)
    print(f"      mesh vertices={len(mesh.vertices)} faces={len(mesh.faces)} occ_face_regions={len(occ_regions)}")

    print("[4/7] Voxelizing mesh...")
    vox = voxelize_mesh(mesh, args.pitch, fill=args.fill)
    print(f"      voxel grid shape={tuple(int(v) for v in vox.matrix.shape)} filled={int(np.count_nonzero(vox.matrix))}")

    print("[5/7] Parsing STEP AP242 semantic layer...")
    entities = parse_step_entities(step_path)
    shape_aspect_to_faces = build_shape_aspect_to_step_faces(entities)
    step_faces = extract_step_face_signatures(entities)
    pmi_records: List[PMIRecord] = []
    pmi_records.extend(extract_dimensions(entities, shape_aspect_to_faces))
    pmi_records.extend(extract_geometric_tolerances(entities, shape_aspect_to_faces))
    pmi_records.extend(extract_standalone_datums(entities, shape_aspect_to_faces))
    print(f"      parsed entities={len(entities)} step_faces={len(step_faces)} pmi_records={len(pmi_records)}")

    print("[6/7] Matching STEP faces to OCC face regions and building region grid...")
    step_to_occ = match_step_faces_to_occ_regions(step_faces, occ_regions)
    annotate_pmi_records(pmi_records, step_to_occ)
    region_grid, region_to_voxels = build_region_id_grid(vox, mesh, triangle_region_ids, len(occ_regions), args.pitch)
    matched_pmi = sum(1 for r in pmi_records if r.linked_region_ids)
    print(f"      step_face_matches={len(step_to_occ)} pmi_with_region_links={matched_pmi}")

    print("[7/7] Saving outputs...")
    npz_path, json_path = save_outputs(save_prefix, vox, region_grid, occ_regions, step_faces, pmi_records, region_to_voxels)
    print(f"      saved voxel+region grid: {npz_path}")
    print(f"      saved semantic sidecar: {json_path}")

    if not args.no_preview:
        try:
            visualize_voxels_matplotlib(vox, region_grid=region_grid)
        except Exception as exc:
            print(f"Preview skipped: {exc}")


if __name__ == "__main__":
    main()
