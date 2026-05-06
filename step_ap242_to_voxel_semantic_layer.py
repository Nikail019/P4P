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
import copy
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import trimesh

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
    OCC_IMPORT_ERROR = exc
else:
    OCC_IMPORT_ERROR = None


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
    bbox_min: Optional[Tuple[float, float, float]] = None
    bbox_max: Optional[Tuple[float, float, float]] = None
    matched_occ_face_index: Optional[int] = None
    match_score: Optional[float] = None
    match_confidence: str = "failed"
    match_reason: str = "not evaluated"


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


@dataclass
class FaceMatchCandidate:
    step_face_id: int
    occ_face_index: int
    step_surface_type: str
    occ_surface_type: str
    surface_type_match: bool
    centroid_distance: Optional[float]
    centroid_score: float
    area_difference: Optional[float]
    area_score: float
    angular_difference: Optional[float]
    angular_score: float
    radius_difference: Optional[float]
    radius_score: float
    bbox_difference: Optional[float]
    bbox_score: float
    adjacency_signature: Optional[str]
    final_score: float
    selected: bool = False
    rejected: bool = True
    rejection_reason: str = ""


@dataclass
class FaceMatchResult:
    step_face_id: int
    selected_occ_face_index: Optional[int]
    selected_score: Optional[float]
    confidence: str
    reason: str
    candidate_occ_face_indices: List[int]


@dataclass
class GeometryAlignment:
    step_bbox_min_raw: Optional[Tuple[float, float, float]]
    step_bbox_max_raw: Optional[Tuple[float, float, float]]
    occ_bbox_min: Optional[Tuple[float, float, float]]
    occ_bbox_max: Optional[Tuple[float, float, float]]
    mesh_bbox_min: Optional[Tuple[float, float, float]]
    mesh_bbox_max: Optional[Tuple[float, float, float]]
    scale_step_to_occ: float
    translation_step_to_occ: Tuple[float, float, float]
    scale_ratio_per_axis: Optional[Tuple[float, float, float]]
    alignment_applied: bool
    reason: str
    step_length_unit: Optional[str]
    step_length_unit_to_metre: Optional[float]


# -----------------------------------------------------------------------------
# STEP reading / meshing / voxelization
# -----------------------------------------------------------------------------


def read_step_shape(step_path: Path):
    if OCC_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Failed to import pythonocc-core / Open CASCADE bindings.\n"
            "Install them first, for example:\n"
            "    pip install pythonocc-core\n\n"
            f"Import error: {OCC_IMPORT_ERROR}"
        )
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



def collect_step_face_points(face_entity_id: int, entities: Dict[int, str]) -> np.ndarray:
    """Walk ADVANCED_FACE -> FACE_BOUND -> EDGE_LOOP -> vertices."""
    raw = entities.get(face_entity_id)
    if raw is None:
        return np.empty((0, 3), dtype=np.float64)

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
        return np.empty((0, 3), dtype=np.float64)
    arr = np.array(points, dtype=np.float64)
    return np.unique(arr, axis=0)



def collect_step_face_boundary_trace(face_entity_id: int, entities: Dict[int, str]) -> Dict[str, List[int]]:
    trace: Dict[str, List[int]] = defaultdict(list)
    seen: set = set()
    queue: List[Tuple[int, int]] = [(face_entity_id, 0)]
    while queue:
        eid, depth = queue.pop(0)
        if eid in seen or depth > 7:
            continue
        seen.add(eid)
        raw = entities.get(eid, "")
        head = entity_head_type(raw) or "unknown"
        key = head.lower() + "_ids"
        trace[key].append(eid)
        if "CARTESIAN_POINT(" in raw:
            continue
        for ref in entity_refs(raw):
            if ref not in seen:
                queue.append((ref, depth + 1))
    return {k: sorted(set(v)) for k, v in trace.items()}



def compute_step_face_centroid(face_entity_id: int, entities: Dict[int, str]) -> Optional[Tuple[float, float, float]]:
    """Compute a STEP face centroid from explicit boundary vertices when present."""
    unique = collect_step_face_points(face_entity_id, entities)
    if len(unique) == 0:
        return None
    return tuple(np.mean(unique, axis=0).tolist())



def compute_step_face_bbox(face_entity_id: int, entities: Dict[int, str]) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    unique = collect_step_face_points(face_entity_id, entities)
    if len(unique) == 0:
        return None, None
    return tuple(np.min(unique, axis=0).tolist()), tuple(np.max(unique, axis=0).tolist())



def compute_all_step_points(step_faces: Dict[int, StepFaceSignature], entities: Dict[int, str]) -> np.ndarray:
    arrays = [collect_step_face_points(fid, entities) for fid in step_faces]
    arrays = [a for a in arrays if len(a) > 0]
    if not arrays:
        return np.empty((0, 3), dtype=np.float64)
    return np.unique(np.vstack(arrays), axis=0)



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
        bbox_min, bbox_max = compute_step_face_bbox(entity_id, entities)
        step_faces[entity_id] = StepFaceSignature(
            step_face_id=entity_id,
            surface_type=surface_type,
            origin=origin,
            direction=direction,
            radius=radius,
            orientation_flag=orientation_flag,
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )
    return step_faces



def detect_step_length_unit(entities: Dict[int, str]) -> Tuple[Optional[str], Optional[float]]:
    """Best-effort STEP model length unit from GLOBAL_UNIT_ASSIGNED_CONTEXT."""
    context_refs: List[int] = []
    for _eid, raw in entities.items():
        if "GLOBAL_UNIT_ASSIGNED_CONTEXT(" in raw:
            context_refs.extend(entity_refs(raw))
    candidates = context_refs or list(entities.keys())
    for eid in candidates:
        raw = entities.get(eid, "")
        if "LENGTH_UNIT" not in raw:
            continue
        subtree = raw
        for ref in entity_refs(raw):
            subtree += " " + entities.get(ref, "")
        if ".MILLI." in subtree and ".METRE." in subtree:
            return "millimetre", 0.001
        if ".CENTI." in subtree and ".METRE." in subtree:
            return "centimetre", 0.01
        if ".METRE." in subtree:
            return "metre", 1.0
        if "INCH" in subtree.upper():
            return "inch", 0.0254
    return None, None


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


def _bbox_from_points(points: np.ndarray) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    if points.size == 0:
        return None, None
    return tuple(np.min(points, axis=0).tolist()), tuple(np.max(points, axis=0).tolist())



def _bbox_from_occ_regions(occ_regions: List[OccFaceRegion]) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    if not occ_regions:
        return None, None
    mins = np.asarray([r.bbox_min for r in occ_regions], dtype=np.float64)
    maxs = np.asarray([r.bbox_max for r in occ_regions], dtype=np.float64)
    return tuple(np.min(mins, axis=0).tolist()), tuple(np.max(maxs, axis=0).tolist())



def _bbox_extents(bmin: Optional[Sequence[float]], bmax: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if bmin is None or bmax is None:
        return None
    return np.asarray(bmax, dtype=np.float64) - np.asarray(bmin, dtype=np.float64)



def infer_geometry_alignment(
    step_faces: Dict[int, StepFaceSignature],
    entities: Dict[int, str],
    occ_regions: List[OccFaceRegion],
    mesh: trimesh.Trimesh,
) -> GeometryAlignment:
    step_points = compute_all_step_points(step_faces, entities)
    step_min, step_max = _bbox_from_points(step_points)
    occ_min, occ_max = _bbox_from_occ_regions(occ_regions)
    mesh_min = tuple(np.min(np.asarray(mesh.vertices), axis=0).tolist()) if len(mesh.vertices) else None
    mesh_max = tuple(np.max(np.asarray(mesh.vertices), axis=0).tolist()) if len(mesh.vertices) else None
    unit_name, unit_to_metre = detect_step_length_unit(entities)

    step_ext = _bbox_extents(step_min, step_max)
    occ_ext = _bbox_extents(occ_min, occ_max)
    scale = 1.0
    ratios_tuple = None
    applied = False
    reason = "insufficient bbox data; no alignment applied"
    translation = np.zeros(3, dtype=np.float64)

    if step_ext is not None and occ_ext is not None:
        valid = step_ext > 1.0e-12
        ratios = occ_ext[valid] / step_ext[valid]
        ratios_tuple_values = np.ones(3, dtype=np.float64)
        ratios_tuple_values[valid] = occ_ext[valid] / step_ext[valid]
        ratios_tuple = tuple(float(v) for v in ratios_tuple_values)
        if len(ratios) > 0:
            median_ratio = float(np.median(ratios))
            spread = float(np.max(np.abs(ratios - median_ratio))) if len(ratios) else 0.0
            if median_ratio > 0 and spread <= max(abs(median_ratio) * 0.02, 1.0e-9):
                scale = median_ratio
                translation = np.asarray(occ_min, dtype=np.float64) - np.asarray(step_min, dtype=np.float64) * scale
                applied = not math.isclose(scale, 1.0, rel_tol=1.0e-9, abs_tol=1.0e-12) or np.linalg.norm(translation) > 1.0e-9
                reason = "consistent bbox scale/translation inferred"
            else:
                reason = f"inconsistent bbox scale ratios; no alignment applied: {ratios_tuple}"

    return GeometryAlignment(
        step_bbox_min_raw=step_min,
        step_bbox_max_raw=step_max,
        occ_bbox_min=occ_min,
        occ_bbox_max=occ_max,
        mesh_bbox_min=mesh_min,
        mesh_bbox_max=mesh_max,
        scale_step_to_occ=float(scale),
        translation_step_to_occ=tuple(float(v) for v in translation),
        scale_ratio_per_axis=ratios_tuple,
        alignment_applied=applied,
        reason=reason,
        step_length_unit=unit_name,
        step_length_unit_to_metre=unit_to_metre,
    )



def _scale_point(
    value: Optional[Tuple[float, float, float]],
    scale: float,
    translation: Sequence[float],
) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64) * scale + np.asarray(translation, dtype=np.float64)
    return tuple(float(v) for v in arr)



def apply_geometry_alignment_to_step_faces(
    step_faces: Dict[int, StepFaceSignature],
    alignment: GeometryAlignment,
) -> None:
    scale = alignment.scale_step_to_occ
    translation = alignment.translation_step_to_occ
    if not alignment.alignment_applied:
        return
    for sig in step_faces.values():
        sig.origin = _scale_point(sig.origin, scale, translation)
        sig.centroid = _scale_point(sig.centroid, scale, translation)
        sig.bbox_min = _scale_point(sig.bbox_min, scale, translation)
        sig.bbox_max = _scale_point(sig.bbox_max, scale, translation)
        if sig.radius is not None:
            sig.radius = float(sig.radius * scale)
        if sig.area_estimate is not None:
            sig.area_estimate = float(sig.area_estimate * scale * scale)



def _bbox_difference(
    step_min: Optional[Sequence[float]],
    step_max: Optional[Sequence[float]],
    occ_min: Sequence[float],
    occ_max: Sequence[float],
) -> Optional[float]:
    if step_min is None or step_max is None:
        return None
    s0 = np.asarray(step_min, dtype=np.float64)
    s1 = np.asarray(step_max, dtype=np.float64)
    o0 = np.asarray(occ_min, dtype=np.float64)
    o1 = np.asarray(occ_max, dtype=np.float64)
    center_gap = np.linalg.norm(((s0 + s1) * 0.5) - ((o0 + o1) * 0.5))
    size_gap = np.linalg.norm((s1 - s0) - (o1 - o0))
    return float(center_gap + 0.5 * size_gap)



def _confidence_from_score(score: Optional[float]) -> str:
    if score is None:
        return "failed"
    if score <= 0.03:
        return "exact"
    if score <= 0.15:
        return "high"
    if score <= 0.35:
        return "medium"
    return "failed"



def _build_face_match_candidates(
    step_faces: Dict[int, StepFaceSignature],
    occ_regions: List[OccFaceRegion],
) -> List[FaceMatchCandidate]:
    all_diags = [_bbox_diagonal(r) for r in occ_regions if _bbox_diagonal(r) > 0]
    char_length = max(float(np.median(all_diags)) if all_diags else 1.0, 1.0e-9)
    candidates: List[FaceMatchCandidate] = []

    for step_face_id, step_sig in step_faces.items():
        for occ in occ_regions:
            surface_type_match = step_sig.surface_type == occ.surface_type
            if not surface_type_match and step_sig.surface_type != "other" and occ.surface_type != "other":
                candidates.append(
                    FaceMatchCandidate(
                        step_face_id=step_face_id,
                        occ_face_index=occ.occ_face_index,
                        step_surface_type=step_sig.surface_type,
                        occ_surface_type=occ.surface_type,
                        surface_type_match=False,
                        centroid_distance=None,
                        centroid_score=999.0,
                        area_difference=None,
                        area_score=999.0,
                        angular_difference=None,
                        angular_score=999.0,
                        radius_difference=None,
                        radius_score=999.0,
                        bbox_difference=None,
                        bbox_score=999.0,
                        adjacency_signature=None,
                        final_score=999.0,
                        rejection_reason="surface_type_mismatch",
                    )
                )
                continue

            cdist = _centroid_distance(step_sig.centroid, occ.centroid)
            if cdist >= 1.0e8 and step_sig.surface_type == "plane":
                cdist = point_to_plane_distance(occ.centroid, step_sig.origin, step_sig.direction)
            centroid_score = cdist / char_length if cdist < 1.0e8 else 10.0

            area_difference = None
            area_score = 0.0
            if step_sig.area_estimate is not None and occ.area > 0:
                area_difference = abs(float(step_sig.area_estimate) - float(occ.area))
                area_score = area_difference / max(float(occ.area), 1.0e-9)

            dir_angle = angle_between_dirs(step_sig.direction, occ.direction)
            angular_difference = None if dir_angle >= math.pi else dir_angle
            angular_score = 0.0 if angular_difference is None else angular_difference / math.pi

            radius_difference = None
            radius_score = 0.0
            if step_sig.surface_type == "cylinder":
                if step_sig.radius is None or occ.radius is None:
                    radius_score = 1.0
                else:
                    radius_difference = abs(step_sig.radius - occ.radius)
                    radius_score = radius_difference / max(abs(occ.radius), 1.0e-9)

            bbox_difference = _bbox_difference(step_sig.bbox_min, step_sig.bbox_max, occ.bbox_min, occ.bbox_max)
            bbox_score = bbox_difference / char_length if bbox_difference is not None else 0.0

            if step_sig.surface_type == "plane":
                score = 0.60 * centroid_score + 0.25 * angular_score + 0.15 * bbox_score + 0.10 * area_score
            elif step_sig.surface_type == "cylinder":
                score = 0.45 * centroid_score + 0.20 * angular_score + 0.25 * radius_score + 0.10 * bbox_score
            else:
                score = 0.70 * centroid_score + 0.20 * bbox_score + 0.10 * area_score

            rejection_reason = ""
            if angular_difference is not None and angular_difference > 0.35:
                score += 2.0
                rejection_reason = "axis_or_normal_misaligned"

            candidates.append(
                FaceMatchCandidate(
                    step_face_id=step_face_id,
                    occ_face_index=occ.occ_face_index,
                    step_surface_type=step_sig.surface_type,
                    occ_surface_type=occ.surface_type,
                    surface_type_match=surface_type_match,
                    centroid_distance=None if cdist >= 1.0e8 else float(cdist),
                    centroid_score=float(centroid_score),
                    area_difference=area_difference,
                    area_score=float(area_score),
                    angular_difference=angular_difference,
                    angular_score=float(angular_score),
                    radius_difference=radius_difference,
                    radius_score=float(radius_score),
                    bbox_difference=bbox_difference,
                    bbox_score=float(bbox_score),
                    adjacency_signature=None,
                    final_score=float(score),
                    rejection_reason=rejection_reason,
                )
            )
    return candidates



def _min_cost_one_to_one_assignment(
    step_ids: List[int],
    occ_ids: List[int],
    candidate_costs: Dict[Tuple[int, int], float],
    fail_cost: float,
) -> Dict[int, int]:
    source = 0
    step_offset = 1
    occ_offset = step_offset + len(step_ids)
    sink = occ_offset + len(occ_ids)
    graph: List[List[Dict[str, Any]]] = [[] for _ in range(sink + 1)]

    def add_edge(u: int, v: int, cap: int, cost: int, tag: Optional[Tuple[int, int]] = None) -> None:
        graph[u].append({"v": v, "cap": cap, "cost": cost, "rev": len(graph[v]), "tag": tag})
        graph[v].append({"v": u, "cap": 0, "cost": -cost, "rev": len(graph[u]) - 1, "tag": None})

    step_index = {sid: i for i, sid in enumerate(step_ids)}
    occ_index = {oid: i for i, oid in enumerate(occ_ids)}
    for sid, i in step_index.items():
        add_edge(source, step_offset + i, 1, 0)
        add_edge(step_offset + i, sink, 1, int(fail_cost * 1_000_000), tag=(sid, -1))
    for oid, j in occ_index.items():
        add_edge(occ_offset + j, sink, 1, 0)
    for (sid, oid), cost in candidate_costs.items():
        if sid in step_index and oid in occ_index:
            add_edge(step_offset + step_index[sid], occ_offset + occ_index[oid], 1, int(cost * 1_000_000), tag=(sid, oid))

    for _ in range(len(step_ids)):
        dist = [math.inf] * len(graph)
        parent: List[Optional[Tuple[int, int]]] = [None] * len(graph)
        in_queue = [False] * len(graph)
        dist[source] = 0
        queue = [source]
        in_queue[source] = True
        while queue:
            u = queue.pop(0)
            in_queue[u] = False
            for ei, edge in enumerate(graph[u]):
                if edge["cap"] <= 0:
                    continue
                v = edge["v"]
                nd = dist[u] + edge["cost"]
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = (u, ei)
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        if parent[sink] is None:
            break
        v = sink
        while v != source:
            u, ei = parent[v]
            edge = graph[u][ei]
            edge["cap"] -= 1
            graph[v][edge["rev"]]["cap"] += 1
            v = u

    matches: Dict[int, int] = {}
    for sid, i in step_index.items():
        for edge in graph[step_offset + i]:
            tag = edge.get("tag")
            if tag and tag[0] == sid and tag[1] != -1 and edge["cap"] == 0:
                matches[sid] = tag[1]
    return matches



def match_step_faces_to_occ_regions(
    step_faces: Dict[int, StepFaceSignature],
    occ_regions: List[OccFaceRegion],
    fail_threshold: float = 0.75,
    ambiguity_margin: float = 0.08,
) -> Tuple[Dict[int, int], Dict[int, FaceMatchResult], List[FaceMatchCandidate]]:
    """
    Conservative, explainable STEP ADVANCED_FACE -> OCC face-region matching.

    Candidate scores are globally assigned one-to-one. Matches are rejected if
    weak or if multiple candidates are too close to distinguish safely.
    """
    candidates = _build_face_match_candidates(step_faces, occ_regions)
    by_step: Dict[int, List[FaceMatchCandidate]] = defaultdict(list)
    for cand in candidates:
        by_step[cand.step_face_id].append(cand)

    results: Dict[int, FaceMatchResult] = {}
    eligible_costs: Dict[Tuple[int, int], float] = {}
    ambiguous_step_ids: set[int] = set()

    for sid in step_faces:
        ranked = sorted(by_step.get(sid, []), key=lambda c: c.final_score)
        viable = [c for c in ranked if c.final_score <= fail_threshold and c.rejection_reason != "surface_type_mismatch"]
        if not viable:
            reason = "no candidate below failure threshold"
            if ranked and ranked[0].rejection_reason:
                reason = ranked[0].rejection_reason
            results[sid] = FaceMatchResult(sid, None, None, "failed", reason, [c.occ_face_index for c in ranked[:5]])
            continue
        if len(viable) > 1 and (viable[1].final_score - viable[0].final_score) < ambiguity_margin:
            ambiguous_step_ids.add(sid)
            results[sid] = FaceMatchResult(
                sid,
                None,
                viable[0].final_score,
                "ambiguous",
                f"best and second-best scores differ by {viable[1].final_score - viable[0].final_score:.4f}",
                [c.occ_face_index for c in viable[:5]],
            )
            continue
        for cand in viable:
            eligible_costs[(sid, cand.occ_face_index)] = cand.final_score

    assignable_step_ids = [sid for sid in step_faces if sid not in results and sid not in ambiguous_step_ids]
    occ_ids = [r.occ_face_index for r in occ_regions]
    assigned = _min_cost_one_to_one_assignment(assignable_step_ids, occ_ids, eligible_costs, fail_threshold)

    candidate_by_pair = {(c.step_face_id, c.occ_face_index): c for c in candidates}
    matches: Dict[int, int] = {}
    for sid in assignable_step_ids:
        oid = assigned.get(sid)
        if oid is None:
            results[sid] = FaceMatchResult(sid, None, None, "failed", "global assignment selected fail option", [])
            continue
        selected = candidate_by_pair[(sid, oid)]
        alt_scores = sorted(
            c.final_score for c in by_step.get(sid, [])
            if c.occ_face_index != oid and c.final_score <= fail_threshold
        )
        if alt_scores and (alt_scores[0] - selected.final_score) < ambiguity_margin:
            results[sid] = FaceMatchResult(
                sid,
                None,
                selected.final_score,
                "ambiguous",
                f"selected and next candidate scores differ by {alt_scores[0] - selected.final_score:.4f}",
                [c.occ_face_index for c in sorted(by_step.get(sid, []), key=lambda c: c.final_score)[:5]],
            )
            continue
        confidence = _confidence_from_score(selected.final_score)
        if confidence == "failed":
            results[sid] = FaceMatchResult(sid, None, selected.final_score, "failed", "selected score above confidence threshold", [oid])
            continue
        matches[sid] = oid
        results[sid] = FaceMatchResult(sid, oid, selected.final_score, confidence, "selected by global one-to-one assignment", [oid])

    for cand in candidates:
        result = results.get(cand.step_face_id)
        if result and result.selected_occ_face_index == cand.occ_face_index:
            cand.selected = True
            cand.rejected = False
            cand.rejection_reason = ""
        elif not cand.rejection_reason:
            cand.rejection_reason = "not selected"

    for sid, result in results.items():
        sig = step_faces[sid]
        sig.matched_occ_face_index = result.selected_occ_face_index
        sig.match_score = result.selected_score
        sig.match_confidence = result.confidence
        sig.match_reason = result.reason

    return matches, results, candidates


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
    match_results: Optional[Dict[int, FaceMatchResult]] = None,
) -> None:
    for rec in records:
        occ_ids = sorted({step_to_occ[sid] for sid in rec.linked_step_face_ids if sid in step_to_occ})
        rec.linked_occ_face_indices = occ_ids
        rec.linked_region_ids = occ_ids[:]  # region ids are OCC face ids in this script
        linked_results = [match_results[sid] for sid in rec.linked_step_face_ids if match_results and sid in match_results]
        failed_or_ambiguous = [r for r in linked_results if r.confidence in {"failed", "ambiguous"}]
        if not rec.linked_step_face_ids:
            rec.valid_association = False
            rec.association_method = "no_step_face_association"
            rec.notes.append("PMI record did not resolve to any STEP ADVANCED_FACE ids.")
        elif failed_or_ambiguous:
            reasons = "; ".join(f"STEP face {r.step_face_id}: {r.confidence} ({r.reason})" for r in failed_or_ambiguous)
            rec.notes.append(f"Step-face association found, but safe OCC face-region mapping failed: {reasons}")
            rec.association_method = "step_face_found_occ_match_failed"
            rec.valid_association = False
        elif occ_ids:
            rec.association_method = "step_face_to_occ_face_signature_match"
            rec.valid_association = True
        else:
            rec.notes.append("Step-face association found, but no OCC face-region match succeeded.")
            rec.association_method = "step_face_found_occ_match_failed"
            rec.valid_association = False


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



def _pmi_confidence(
    rec: PMIRecord,
    match_results: Dict[int, FaceMatchResult],
) -> Tuple[str, Optional[float], str]:
    if not rec.linked_step_face_ids:
        return "failed", None, "PMI record has no referenced STEP faces"
    results = [match_results.get(sid) for sid in rec.linked_step_face_ids]
    missing = [sid for sid, result in zip(rec.linked_step_face_ids, results) if result is None]
    if missing:
        return "failed", None, f"missing STEP-face match results for {missing}"
    bad = [r for r in results if r is not None and r.confidence in {"failed", "ambiguous"}]
    if bad:
        reason = "; ".join(f"STEP face {r.step_face_id}: {r.confidence} ({r.reason})" for r in bad)
        return ("ambiguous" if any(r.confidence == "ambiguous" for r in bad) else "failed"), None, reason
    scores = [r.selected_score for r in results if r is not None and r.selected_score is not None]
    if not scores:
        return "failed", None, "no selected match scores"
    worst_score = max(scores)
    return _confidence_from_score(worst_score), worst_score, "all referenced STEP faces mapped"


def _root_pmi_entity_id(rec: PMIRecord) -> Optional[int]:
    if not rec.source_entity_ids:
        return None
    try:
        suffix = int(rec.pmi_id.rsplit("_", 1)[-1])
    except ValueError:
        suffix = None
    if suffix in rec.source_entity_ids:
        return suffix
    return rec.source_entity_ids[-1]



def _occ_region_by_id(occ_regions: List[OccFaceRegion]) -> Dict[int, OccFaceRegion]:
    return {r.occ_face_index: r for r in occ_regions}



def _world_region_centroid_from_voxels(
    rid: int,
    vox: trimesh.voxel.VoxelGrid,
    region_grid: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    mask = np.asarray(region_grid == rid)
    idx = np.argwhere(mask & np.asarray(vox.matrix, dtype=bool))
    if len(idx) == 0:
        return None
    pitch_value = float(np.asarray(vox.pitch).reshape(-1)[0])
    origin = np.asarray(vox.transform, dtype=np.float64)[:3, 3]
    center = origin + np.mean(idx, axis=0) * pitch_value + pitch_value * 0.5
    return tuple(float(v) for v in center)



def _pmi_parsed_status(rec: PMIRecord) -> Tuple[str, str]:
    if rec.category == "datum":
        return ("valid", "datum label parsed") if rec.label else ("partial", "datum entity parsed but label missing")
    if rec.category == "dimension":
        return ("valid", "dimension value and unit parsed") if rec.value is not None and rec.unit else ("partial", "dimension entity parsed but value/unit incomplete")
    if rec.category == "geometric_tolerance":
        return ("valid", "geometric tolerance value parsed") if rec.value is not None else ("partial", "geometric tolerance entity parsed but value missing")
    return "partial", "generic PMI record parsed"



def _semantic_status(rec: PMIRecord, match_results: Dict[int, FaceMatchResult]) -> Tuple[str, Optional[float], str]:
    confidence, score, reason = _pmi_confidence(rec, match_results)
    if confidence in {"exact", "high", "medium"} and rec.linked_region_ids:
        return "valid", score, reason
    if confidence == "ambiguous":
        return "ambiguous", score, reason
    return "failed", score, reason



def assess_pmi_display(
    rec: PMIRecord,
    occ_regions: List[OccFaceRegion],
    vox: trimesh.voxel.VoxelGrid,
    region_grid: np.ndarray,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """Classify viewer display support without changing semantic confidence."""
    if not rec.linked_region_ids:
        return "not_supported", "PMI has no mapped voxel regions to display.", None

    regions = _occ_region_by_id(occ_regions)
    mapped = [regions[rid] for rid in rec.linked_region_ids if rid in regions]
    model_size = float(np.linalg.norm(np.asarray(vox.bounds[1]) - np.asarray(vox.bounds[0]))) if hasattr(vox, "bounds") else 1.0

    if rec.category == "dimension":
        points = [_world_region_centroid_from_voxels(rid, vox, region_grid) for rid in rec.linked_region_ids]
        points_arr = [np.asarray(p, dtype=np.float64) for p in points if p is not None]
        if len(points_arr) >= 2:
            start = points_arr[0]
            end = points_arr[1]
            display_vec = end - start
            expected = None
            status = "approximate"
            reason = "dimension display uses mapped region centroids; AP242 annotation/witness geometry is not parsed."

            if len(mapped) >= 2 and mapped[0].surface_type == "plane" and mapped[1].surface_type == "plane":
                n1 = _norm_vec(mapped[0].direction)
                n2 = _norm_vec(mapped[1].direction)
                if n1 is not None and n2 is not None and abs(float(np.dot(n1, n2))) > 0.98:
                    expected = n1
                    if float(np.dot(end - start, expected)) < 0:
                        expected = -expected
                    projected_end = start + expected * float(np.dot(end - start, expected))
                    end = projected_end
                    display_vec = end - start
                    status = "valid"
                    reason = "two parallel planar faces; display vector reconstructed along plane normal."
            elif len(mapped) >= 2:
                types = sorted({m.surface_type for m in mapped})
                if "cylinder" in types:
                    status = "approximate"
                    reason = "dimension involves a cylinder; centre/diameter/radius semantics are not fully reconstructed from AP242."

            angle = None
            if expected is not None and np.linalg.norm(display_vec) > 1.0e-9:
                dv = display_vec / np.linalg.norm(display_vec)
                angle = float(math.degrees(math.acos(float(np.clip(abs(np.dot(dv, expected)), -1.0, 1.0)))))

            diag = {
                "pmi_id": rec.pmi_id,
                "dimension_value": rec.value,
                "referenced_regions": rec.linked_region_ids,
                "display_start_point": start.tolist(),
                "display_end_point": end.tolist(),
                "display_vector": display_vec.tolist(),
                "expected_measurement_direction": expected.tolist() if expected is not None else None,
                "angle_between_display_and_expected_degrees": angle,
                "display_status": status,
                "reason": reason,
            }
            return status, reason, diag
        return "not_supported", "dimension display needs at least two mapped regions.", None

    if rec.category == "datum":
        return "approximate", "datum label/leader is displayed, but formal datum feature symbol geometry is approximate.", None
    if rec.category == "geometric_tolerance":
        return "approximate", "geometric tolerance is displayed as a generic label/leader, not a standards-complete feature control frame.", None
    return "not_supported", "PMI category is not displayed by the MBD viewer.", None


_FIT_RE = re.compile(r"\b([A-Z])\s*(?:IT)?([6-9]|10)\b", re.IGNORECASE)
_IT_RE = re.compile(r"\bIT\s*([6-9]|10)\b", re.IGNORECASE)


def extract_fit_info(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text or ""
    it_match = _IT_RE.search(raw_text)
    fit_match = _FIT_RE.search(raw_text)
    fit_letter = fit_grade = fit_class = None
    if fit_match and fit_match.group(1).upper() != "I":
        fit_letter = fit_match.group(1).upper()
        fit_grade = int(fit_match.group(2))
        fit_class = f"{fit_letter}{fit_grade}"
    return {
        "it_grade": int(it_match.group(1)) if it_match else None,
        "fit_class": fit_class,
        "fit_letter": fit_letter,
        "fit_grade": fit_grade,
        "raw_fit_text": fit_match.group(0) if fit_match else (it_match.group(0) if it_match else None),
    }


def extract_tolerance_bounds(modifiers: List[str]) -> Dict[str, Optional[float]]:
    raw = " ".join(modifiers)
    nums = [float(v) for v in _FLOAT_RE.findall(raw)]
    plus = minus = symmetric = None
    if "PLUS_MINUS_TOLERANCE" in raw and nums:
        symmetric = abs(nums[0])
    elif len(nums) >= 2:
        plus = max(nums[0], nums[1])
        minus = min(nums[0], nums[1])
    return {"tolerance_plus": plus, "tolerance_minus": minus, "tolerance_symmetric": symmetric}


def classify_dimension_record(rec: PMIRecord, occ_regions: List[OccFaceRegion]) -> Dict[str, Any]:
    text = " ".join([rec.subtype or "", rec.label or "", " ".join(rec.modifiers)]).upper()
    mapped_regions = _occ_region_by_id(occ_regions)
    mapped = [mapped_regions[rid] for rid in rec.linked_region_ids if rid in mapped_regions]
    cyls = [r for r in mapped if r.surface_type == "cylinder"]
    subtype = "unknown_dimension"
    confidence = "low"
    reason = "dimension subtype is not explicit enough to classify conservatively"
    if "ANGLE" in text or rec.unit in {"deg", "rad"}:
        subtype, confidence, reason = "angular_dimension", "medium", "angle keyword or angular unit found"
    elif "DIAMETER" in text or "DIAM" in text or "Ø" in text:
        subtype, confidence, reason = "diameter_dimension", "medium", "diameter keyword/symbol found"
    elif "RADIUS" in text:
        subtype, confidence, reason = "radius_dimension", "medium", "radius keyword found"
    elif cyls and rec.value is not None:
        for cyl in cyls:
            if cyl.radius is not None and abs(rec.value - 2.0 * cyl.radius) <= max(abs(rec.value) * 0.02, 1.0e-6):
                subtype, confidence, reason = "hole_size_dimension", "medium", "dimension value matches mapped cylinder diameter"
                break
            if cyl.radius is not None and abs(rec.value - cyl.radius) <= max(abs(rec.value) * 0.02, 1.0e-6):
                subtype, confidence, reason = "radius_dimension", "medium", "dimension value matches mapped cylinder radius"
                break
        if subtype == "unknown_dimension":
            subtype, confidence, reason = "linear_distance", "medium", "dimension links to cylinder but value does not match cylinder diameter/radius; likely centre/offset distance"
    elif "DIMENSIONAL_LOCATION" in text or "DISTANCE" in text or len(rec.linked_region_ids) >= 2:
        subtype = "linear_distance"
        confidence = "high" if len(rec.linked_region_ids) >= 2 else "medium"
        reason = "DIMENSIONAL_LOCATION/distance relationship with mapped references"
    return {
        "dimension_subtype": subtype,
        "nominal_value": rec.value,
        "units": rec.unit,
        **extract_tolerance_bounds(rec.modifiers),
        "raw_modifiers": rec.modifiers,
        "referenced_step_faces": rec.linked_step_face_ids,
        "mapped_occ_regions": rec.linked_occ_face_indices,
        "classification_confidence": confidence,
        "classification_reason": reason,
        **extract_fit_info(" ".join([rec.label or "", " ".join(rec.modifiers)])),
    }


def cylinder_qa_for_record(rec: PMIRecord, occ_regions: List[OccFaceRegion]) -> List[Dict[str, Any]]:
    lookup = _occ_region_by_id(occ_regions)
    dim_class = classify_dimension_record(rec, occ_regions) if rec.category == "dimension" else None
    out = []
    for rid in rec.linked_region_ids:
        region = lookup.get(rid)
        if region is None or region.surface_type != "cylinder":
            continue
        radius = region.radius
        out.append(
            {
                "mapped_cylindrical_region_id": rid,
                "cylinder_radius": radius,
                "cylinder_diameter": 2.0 * radius if radius is not None else None,
                "cylinder_axis": region.direction,
                "cylinder_length_depth_estimate": abs(region.bbox_max[2] - region.bbox_min[2]) if region.bbox_min and region.bbox_max else None,
                "appears_to_describe": dim_class["dimension_subtype"] if dim_class else "not_dimension",
                "confidence": dim_class["classification_confidence"] if dim_class else "low",
                "reason": dim_class["classification_reason"] if dim_class else "cylinder linked to non-dimension PMI",
            }
        )
    return out


def classify_geometric_tolerance_record(rec: PMIRecord) -> Dict[str, Any]:
    mapping = {
        "parallelism_tolerance": "parallelism",
        "perpendicularity_tolerance": "perpendicularity",
        "position_tolerance": "position",
        "flatness_tolerance": "flatness",
        "straightness_tolerance": "straightness",
        "roundness_tolerance": "circularity",
        "cylindricity_tolerance": "cylindricity",
        "angularity_tolerance": "angularity",
        "profile_of_a_line_tolerance": "profile",
        "profile_of_a_surface_tolerance": "profile",
        "circular_runout_tolerance": "runout",
        "total_runout_tolerance": "runout",
    }
    gt_type = mapping.get(rec.subtype, "generic_geometric_tolerance")
    limitations = ["tolerance zone/modifiers are not fully normalized"]
    if gt_type == "generic_geometric_tolerance":
        limitations.append("specific GD&T type was not identified")
    if not rec.datum_refs:
        limitations.append("datum reference frame not fully reconstructed")
    completeness = "value_and_type" if gt_type != "generic_geometric_tolerance" and rec.value is not None else "value_only"
    if rec.datum_refs and gt_type != "generic_geometric_tolerance":
        completeness = "partial_feature_control_frame"
    return {
        "geometric_tolerance_type": gt_type,
        "tolerance_value": rec.value,
        "units": rec.unit,
        "datum_references": rec.datum_refs,
        "referenced_step_faces": rec.linked_step_face_ids,
        "mapped_occ_regions": rec.linked_occ_face_indices,
        "semantic_completeness": completeness,
        "limitations": limitations,
    }


def semantic_classification_status(rec: PMIRecord, dim: Optional[Dict[str, Any]], gt: Optional[Dict[str, Any]]) -> str:
    if rec.category == "datum":
        return "partial" if rec.label else "low"
    if rec.category == "dimension":
        if not dim or dim["dimension_subtype"] == "unknown_dimension":
            return "low"
        return dim["classification_confidence"]
    if rec.category == "geometric_tolerance":
        return "partial"
    return "low"


def current_scope_usable(rec: PMIRecord, semantic_status: str, class_status: str) -> Tuple[bool, Optional[str]]:
    if semantic_status != "valid":
        return False, "semantic mapping is not valid"
    if rec.category == "datum":
        return (True, None) if rec.label else (False, "datum label was not parsed")
    if rec.category == "dimension":
        return (class_status in {"high", "medium"}, None if class_status in {"high", "medium"} else "dimension subtype is unknown")
    if rec.category == "geometric_tolerance":
        return False, "geometric tolerance is detected/mapped but not normalized enough for the current supported subset"
    return False, "PMI category is out of current scope"



def build_pmi_qa_records(
    entities: Dict[int, str],
    pmi_records: List[PMIRecord],
    occ_regions: List[OccFaceRegion],
    vox: trimesh.voxel.VoxelGrid,
    region_grid: np.ndarray,
    match_results: Dict[int, FaceMatchResult],
) -> List[Dict[str, Any]]:
    records = []
    for rec in pmi_records:
        root_id = _root_pmi_entity_id(rec)
        parsed_status, parsed_reason = _pmi_parsed_status(rec)
        semantic_status, score, semantic_reason = _semantic_status(rec, match_results)
        display_status, display_issue, linear_display = assess_pmi_display(rec, occ_regions, vox, region_grid)
        dim_class = classify_dimension_record(rec, occ_regions) if rec.category == "dimension" else None
        gt_class = classify_geometric_tolerance_record(rec) if rec.category == "geometric_tolerance" else None
        class_status = semantic_classification_status(rec, dim_class, gt_class)
        usable, excluded_reason = current_scope_usable(rec, semantic_status, class_status)
        records.append(
            {
                "pmi_id": rec.pmi_id,
                "pmi_type": rec.category,
                "pmi_subtype": rec.subtype,
                "raw_step_entity_id": root_id,
                "raw_step_entity": entities.get(root_id) if root_id is not None else None,
                "parsed_value": rec.value,
                "parsed_units": rec.unit,
                "parsed_correctly": parsed_status,
                "parsed_issue": parsed_reason,
                "referenced_step_faces": rec.linked_step_face_ids,
                "linked_to_step_faces_correctly": "valid" if rec.linked_step_face_ids else "failed",
                "mapped_occ_regions": rec.linked_occ_face_indices,
                "mapped_to_occ_regions_correctly": "valid" if rec.linked_occ_face_indices and semantic_status == "valid" else semantic_status,
                "mapped_voxel_regions": rec.linked_region_ids,
                "mapped_to_voxel_regions_correctly": "valid" if rec.linked_region_ids and semantic_status == "valid" else semantic_status,
                "semantic_mapping_status": semantic_status,
                "semantic_classification_status": class_status,
                "semantic_mapping_score": score,
                "semantic_mapping_reason": semantic_reason,
                "dimension_classification": dim_class,
                "cylinder_qa": cylinder_qa_for_record(rec, occ_regions),
                "geometric_tolerance_classification": gt_class,
                "display_status": display_status,
                "display_issue": display_issue,
                "linear_distance_display_diagnostics": linear_display,
                "usable_for_current_project_scope": usable,
                "semantic_record_valid": usable,
                "excluded_reason": excluded_reason,
            }
        )
    return records



def save_diagnostics(
    save_prefix: Path,
    step_path: Path,
    entities: Dict[int, str],
    vox: trimesh.voxel.VoxelGrid,
    region_grid: np.ndarray,
    step_faces: Dict[int, StepFaceSignature],
    occ_regions: List[OccFaceRegion],
    pmi_records: List[PMIRecord],
    step_to_occ: Dict[int, int],
    match_results: Dict[int, FaceMatchResult],
    match_candidates: List[FaceMatchCandidate],
) -> Path:
    diagnostics_path = save_prefix.with_name(save_prefix.name + "_diagnostics.json")
    occupancy = np.asarray(vox.matrix, dtype=bool)
    labelled = np.asarray(region_grid, dtype=np.int32) > 0
    occupied_voxels = int(np.count_nonzero(occupancy))
    region_labelled_voxels = int(np.count_nonzero(occupancy & labelled))
    unlabeled_occupied_voxels = int(np.count_nonzero(occupancy & ~labelled))

    candidates_by_step: Dict[int, List[FaceMatchCandidate]] = defaultdict(list)
    for cand in match_candidates:
        candidates_by_step[cand.step_face_id].append(cand)

    pmi_payload = []
    for rec in pmi_records:
        confidence, score, reason = _pmi_confidence(rec, match_results)
        qa_record = build_pmi_qa_records(entities, [rec], occ_regions, vox, region_grid, match_results)[0]
        candidate_occ = []
        for sid in rec.linked_step_face_ids:
            ranked = sorted(candidates_by_step.get(sid, []), key=lambda c: c.final_score)
            candidate_occ.extend(
                {
                    "step_face_id": sid,
                    "occ_region_id": c.occ_face_index,
                    "score": c.final_score,
                    "selected": c.selected,
                    "rejection_reason": c.rejection_reason,
                }
                for c in ranked[:10]
            )
        pmi_payload.append(
            {
                "pmi_id": rec.pmi_id,
                "source_entity_ids": rec.source_entity_ids,
                "pmi_type": rec.category,
                "pmi_subtype": rec.subtype,
                "raw_step_entity": qa_record["raw_step_entity"],
                "parsed_correctly": qa_record["parsed_correctly"],
                "linked_to_step_faces_correctly": qa_record["linked_to_step_faces_correctly"],
                "mapped_to_occ_regions_correctly": qa_record["mapped_to_occ_regions_correctly"],
                "mapped_to_voxel_regions_correctly": qa_record["mapped_to_voxel_regions_correctly"],
                "label": rec.label,
                "tolerance_value": rec.value,
                "unit": rec.unit,
                "it_grade": next((m for m in rec.modifiers if "IT" in m.upper()), None),
                "referenced_step_face_ids": rec.linked_step_face_ids,
                "candidate_occ_region_matches": candidate_occ,
                "selected_occ_region_ids": rec.linked_occ_face_indices,
                "selected_voxel_region_ids": rec.linked_region_ids,
                "match_score": score,
                "confidence": confidence,
                "reason": reason,
                "semantic_mapping_status": qa_record["semantic_mapping_status"],
                "semantic_classification_status": qa_record["semantic_classification_status"],
                "dimension_classification": qa_record["dimension_classification"],
                "cylinder_qa": qa_record["cylinder_qa"],
                "geometric_tolerance_classification": qa_record["geometric_tolerance_classification"],
                "display_status": qa_record["display_status"],
                "display_issue": qa_record["display_issue"],
                "linear_distance_display_diagnostics": qa_record["linear_distance_display_diagnostics"],
                "semantic_record_valid": bool(rec.valid_association and confidence not in {"failed", "ambiguous"}),
                "usable_for_current_project_scope": qa_record["usable_for_current_project_scope"],
                "usable_for_visual_QA": qa_record["display_status"] in {"valid", "approximate"},
                "excluded_reason": qa_record["excluded_reason"],
                "notes": rec.notes,
            }
        )

    payload = {
        "input_step_filename": str(step_path),
        "grid_shape": [int(v) for v in occupancy.shape],
        "counts": {
            "step_faces_parsed": len(step_faces),
            "occ_regions_extracted": len(occ_regions),
            "pmi_records_parsed": len(pmi_records),
            "pmi_records_with_linked_step_faces": sum(1 for r in pmi_records if r.linked_step_face_ids),
            "step_faces_matched_to_occ_regions": len(step_to_occ),
            "pmi_records_mapped_to_occ_regions": sum(1 for r in pmi_records if r.linked_occ_face_indices),
            "pmi_records_mapped_to_voxel_regions": sum(1 for r in pmi_records if r.linked_region_ids),
            "occupied_voxels": occupied_voxels,
            "region_labelled_voxels": region_labelled_voxels,
            "unlabeled_occupied_voxels": unlabeled_occupied_voxels,
        },
        "semantic_qa_gate": {
            "usable_for_current_project_scope": all(item["usable_for_current_project_scope"] for item in pmi_payload),
            "excluded_pmi_ids": [item["pmi_id"] for item in pmi_payload if not item["usable_for_current_project_scope"]],
            "policy": "PMI records with failed, ambiguous, or insufficiently classified semantics are excluded from the current conservative PMI/GD&T QA scope.",
        },
        "step_face_match_results": [asdict(match_results[sid]) for sid in sorted(match_results)],
        "face_match_candidates": [asdict(c) for c in sorted(match_candidates, key=lambda c: (c.step_face_id, c.final_score, c.occ_face_index))],
        "pmi_records": pmi_payload,
    }
    diagnostics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return diagnostics_path


def save_pmi_qa_report(
    save_prefix: Path,
    step_path: Path,
    entities: Dict[int, str],
    pmi_records: List[PMIRecord],
    occ_regions: List[OccFaceRegion],
    vox: trimesh.voxel.VoxelGrid,
    region_grid: np.ndarray,
    match_results: Dict[int, FaceMatchResult],
) -> Path:
    report_path = save_prefix.with_name(save_prefix.name + "_pmi_qa_report.json")
    records = build_pmi_qa_records(entities, pmi_records, occ_regions, vox, region_grid, match_results)
    by_category = Counter(rec.category for rec in pmi_records)
    display_counts = Counter(r["display_status"] for r in records)
    dim_counts = Counter((r.get("dimension_classification") or {}).get("dimension_subtype") for r in records if r["pmi_type"] == "dimension")
    gt_counts = Counter((r.get("geometric_tolerance_classification") or {}).get("geometric_tolerance_type") for r in records if r["pmi_type"] == "geometric_tolerance")
    semantic_counts = Counter(r["semantic_mapping_status"] for r in records)
    fit_classes = sorted({(r.get("dimension_classification") or {}).get("fit_class") for r in records if (r.get("dimension_classification") or {}).get("fit_class")})
    it_grades = sorted({(r.get("dimension_classification") or {}).get("it_grade") for r in records if (r.get("dimension_classification") or {}).get("it_grade") is not None})
    unsupported = sorted({r["pmi_subtype"] for r in records if r["display_status"] == "not_supported"})
    payload = {
        "input_step_filename": str(step_path),
        "summary": {
            "total_pmi_records_parsed": len(pmi_records),
            "number_of_datums": by_category.get("datum", 0),
            "number_of_dimensions": by_category.get("dimension", 0),
            "number_of_geometric_tolerances": by_category.get("geometric_tolerance", 0),
            "pmi_records_linked_to_step_faces": sum(1 for r in pmi_records if r.linked_step_face_ids),
            "pmi_records_mapped_to_occ_regions": sum(1 for r in pmi_records if r.linked_occ_face_indices),
            "pmi_records_mapped_to_voxel_regions": sum(1 for r in pmi_records if r.linked_region_ids),
            "usable_for_current_project_scope": sum(1 for r in records if r["usable_for_current_project_scope"]),
            "excluded_from_current_project_scope": sum(1 for r in records if not r["usable_for_current_project_scope"]),
            "displayed_correctly": display_counts.get("valid", 0),
            "displayed_approximately": display_counts.get("approximate", 0),
            "displayed_incorrectly": display_counts.get("incorrect", 0),
            "display_not_supported": display_counts.get("not_supported", 0),
            "unsupported_pmi_types_encountered": unsupported,
        },
        "current_model_qa_summary": {
            "file": str(step_path),
            "pmi_total": len(pmi_records),
            "datums": by_category.get("datum", 0),
            "linear_distances": dim_counts.get("linear_distance", 0),
            "diameter_dimensions": dim_counts.get("diameter_dimension", 0),
            "radius_dimensions": dim_counts.get("radius_dimension", 0),
            "hole_size_dimensions": dim_counts.get("hole_size_dimension", 0),
            "generic_dimensions": dim_counts.get("unknown_dimension", 0),
            "geometric_tolerances": by_category.get("geometric_tolerance", 0),
            "classified_geometric_tolerances": sum(v for k, v in gt_counts.items() if k and k != "generic_geometric_tolerance"),
            "generic_geometric_tolerances": gt_counts.get("generic_geometric_tolerance", 0),
            "it_grades_found": it_grades,
            "fit_classes_found": fit_classes,
            "semantic_mapping_valid": semantic_counts.get("valid", 0),
            "semantic_mapping_partial": semantic_counts.get("ambiguous", 0),
            "semantic_mapping_failed": semantic_counts.get("failed", 0),
            "display_valid": display_counts.get("valid", 0),
            "display_approximate": display_counts.get("approximate", 0),
            "display_not_supported": display_counts.get("not_supported", 0),
        },
        "pmi_records": records,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def build_pmi_capability_matrix() -> List[Dict[str, str]]:
    return [
        {"pmi_type": "datums", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "DATUM/DATUM_FEATURE/DATUM_TARGET entities are parsed; label and face mapping can work, but formal datum feature symbol semantics are incomplete."},
        {"pmi_type": "datum labels / datum features", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Datum label extraction is simple string-based; display is approximate flag/leader."},
        {"pmi_type": "linear distance dimensions", "parsed_from_step": "partial", "semantic_value_extracted": "yes", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "DIMENSIONAL_CHARACTERISTIC_REPRESENTATION is parsed. Two-plane display can be reconstructed along plane normal; cylinder/axis cases remain approximate."},
        {"pmi_type": "diameter dimensions", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "May be parsed as a generic dimension if represented through DIMENSIONAL_*; diameter semantics are not explicitly classified yet."},
        {"pmi_type": "radius dimensions", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "May be parsed as a generic dimension; radius-specific semantics are not fully separated from generic dimensions."},
        {"pmi_type": "hole tolerances / hole size tolerances", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Hole size may be represented as dimensions/modifiers; fit semantics need more parsing."},
        {"pmi_type": "ISO IT grades", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "no", "mapped_to_occ_voxel_region": "no", "displayed_in_viewer": "no", "current_project_support": "partial", "notes": "Current code only searches modifiers for IT text; no robust ISO fit/grade parser."},
        {"pmi_type": "general dimensional tolerances", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "PLUS_MINUS_TOLERANCE and modifiers are retained raw, not normalized into upper/lower tolerance fields."},
        {"pmi_type": "geometric tolerances", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Known tolerance entity names are recognized, but display is generic label/leader, not full FCF."},
        {"pmi_type": "flatness", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Recognized when entity is FLATNESS_TOLERANCE; not standards-complete."},
        {"pmi_type": "straightness", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Recognized when entity is STRAIGHTNESS_TOLERANCE; not standards-complete."},
        {"pmi_type": "circularity", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Code currently recognizes ROUNDNESS_TOLERANCE rather than explicitly naming circularity."},
        {"pmi_type": "cylindricity", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Recognized when entity is CYLINDRICITY_TOLERANCE; not standards-complete."},
        {"pmi_type": "parallelism", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Entity name recognized; datum reference handling is lightweight."},
        {"pmi_type": "perpendicularity", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Entity name recognized; datum reference handling is lightweight."},
        {"pmi_type": "angularity", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "Entity name recognized; datum reference handling is lightweight."},
        {"pmi_type": "position", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "POSITION_TOLERANCE recognized; feature control frame semantics are not normalized."},
        {"pmi_type": "concentricity/coaxiality", "parsed_from_step": "no", "semantic_value_extracted": "no", "linked_to_step_faces": "no", "mapped_to_occ_voxel_region": "no", "displayed_in_viewer": "no", "current_project_support": "no", "notes": "Not explicitly in the tolerance keyword list."},
        {"pmi_type": "symmetry", "parsed_from_step": "no", "semantic_value_extracted": "no", "linked_to_step_faces": "no", "mapped_to_occ_voxel_region": "no", "displayed_in_viewer": "no", "current_project_support": "no", "notes": "Not explicitly in the tolerance keyword list."},
        {"pmi_type": "profile of a line", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "PROFILE_OF_A_LINE_TOLERANCE recognized generically."},
        {"pmi_type": "profile of a surface", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "PROFILE_OF_A_SURFACE_TOLERANCE recognized generically."},
        {"pmi_type": "runout / total runout", "parsed_from_step": "partial", "semantic_value_extracted": "partial", "linked_to_step_faces": "partial", "mapped_to_occ_voxel_region": "partial", "displayed_in_viewer": "partial", "current_project_support": "partial", "notes": "CIRCULAR_RUNOUT_TOLERANCE and TOTAL_RUNOUT_TOLERANCE recognized generically."},
        {"pmi_type": "surface finish", "parsed_from_step": "no", "semantic_value_extracted": "no", "linked_to_step_faces": "no", "mapped_to_occ_voxel_region": "no", "displayed_in_viewer": "no", "current_project_support": "no", "notes": "Surface texture/finish entities are not parsed."},
    ]


def save_pmi_capability_matrix(base_dir: Path) -> Tuple[Path, Path]:
    matrix = build_pmi_capability_matrix()
    json_path = base_dir / "pmi_capability_matrix.json"
    md_path = base_dir / "pmi_capability_matrix.md"
    json_path.write_text(json.dumps({"capabilities": matrix}, indent=2), encoding="utf-8")
    headers = [
        "PMI type", "Parsed", "Value", "STEP link", "Mapped", "Displayed", "Current scope", "Notes"
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in matrix:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["pmi_type"],
                    row["parsed_from_step"],
                    row["semantic_value_extracted"],
                    row["linked_to_step_faces"],
                    row["mapped_to_occ_voxel_region"],
                    row["displayed_in_viewer"],
                    row["current_project_support"],
                    row["notes"].replace("|", "/"),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def build_project_pmi_support_profile() -> Dict[str, Any]:
    return {
        "level_1_supported_for_current_project_qa": [
            {"pmi_type": "datum labels", "reason": "label can be parsed, linked, mapped, and audited; display is approximate."},
            {"pmi_type": "datum features", "reason": "datum feature face mapping can be audited when AP242 references resolve."},
            {"pmi_type": "linear distance dimensions", "reason": "DIMENSIONAL_LOCATION records with mapped references are classified conservatively; two-plane displays can be QA-checked."},
        ],
        "level_2_partially_supported": [
            {"pmi_type": "diameter dimensions", "reason": "detected by diameter text/symbol or cylinder diameter match; AP242 diameter semantics are not fully normalized."},
            {"pmi_type": "radius dimensions", "reason": "detected by radius text or cylinder radius match; not fully normalized."},
            {"pmi_type": "hole size dimensions", "reason": "detected when nominal value matches mapped cylinder diameter; hole fit semantics are partial."},
            {"pmi_type": "general dimensional tolerances", "reason": "raw modifiers are retained; upper/lower bounds are best-effort only."},
            {"pmi_type": "upper/lower tolerance bounds", "reason": "simple numeric extraction only; no full tolerance model."},
            {"pmi_type": "ISO IT grades / IT6-IT10", "reason": "IT/fit text can be detected if present; ISO 286 meaning is not interpreted."},
            {"pmi_type": "generic geometric tolerances", "reason": "detected and mapped, but not considered normalized enough for Level 1."},
            {"pmi_type": "flatness", "reason": "entity name classified; FCF semantics incomplete."},
            {"pmi_type": "straightness", "reason": "entity name classified; FCF semantics incomplete."},
            {"pmi_type": "circularity", "reason": "ROUNDNESS_TOLERANCE classified as circularity; FCF semantics incomplete."},
            {"pmi_type": "cylindricity", "reason": "entity name classified; FCF semantics incomplete."},
            {"pmi_type": "parallelism", "reason": "entity name classified; datum frame reconstruction is partial."},
            {"pmi_type": "perpendicularity", "reason": "entity name classified; datum frame reconstruction is partial."},
            {"pmi_type": "angularity", "reason": "entity name classified; datum frame reconstruction is partial."},
            {"pmi_type": "position", "reason": "entity name classified; target feature/tolerance zone semantics are partial."},
            {"pmi_type": "profile", "reason": "profile entity names are grouped; details incomplete."},
            {"pmi_type": "runout", "reason": "runout entity names are grouped; datum/axis semantics incomplete."},
        ],
        "level_3_out_of_current_scope": [
            {"pmi_type": "surface finish", "reason": "surface texture entities are not parsed."},
            {"pmi_type": "full feature-control-frame rendering", "reason": "viewer is QA-oriented, not standards-complete MBD reconstruction."},
            {"pmi_type": "full ISO 286 fit interpretation", "reason": "fit text may be detected, but fit meaning is not interpreted."},
        ],
    }


def save_project_pmi_support_profile(base_dir: Path) -> Tuple[Path, Path]:
    profile = build_project_pmi_support_profile()
    json_path = base_dir / "project_pmi_support_profile.json"
    md_path = base_dir / "project_pmi_support_profile.md"
    json_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    lines = ["# Project PMI Support Profile", ""]
    labels = [
        ("level_1_supported_for_current_project_qa", "Level 1 - Supported for Current Project QA"),
        ("level_2_partially_supported", "Level 2 - Partially Supported"),
        ("level_3_out_of_current_scope", "Level 3 - Out of Current Scope"),
    ]
    for key, title in labels:
        lines.extend([f"## {title}", ""])
        for item in profile[key]:
            lines.append(f"- **{item['pmi_type']}**: {item['reason']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path



def _surface_ref_for_step_face(face_id: int, entities: Dict[int, str]) -> Optional[int]:
    raw = entities.get(face_id, "")
    _head, args = parse_simple_entity(raw)
    if len(args) < 3:
        return None
    surface_ref = args[2].strip()
    return int(surface_ref[1:]) if surface_ref.startswith("#") else None



def _orientation_reversed(face_id: int, entities: Dict[int, str]) -> Optional[bool]:
    raw = entities.get(face_id, "")
    _head, args = parse_simple_entity(raw)
    if len(args) < 4:
        return None
    return args[3].strip().upper() in {".F.", "F", "FALSE", ".FALSE."}



def _entity_payload(entity_id: int, entities: Dict[int, str]) -> Dict[str, Any]:
    raw = entities.get(entity_id)
    return {
        "id": entity_id,
        "type": entity_head_type(raw or "") if raw else None,
        "raw": raw,
        "refs": entity_refs(raw or ""),
    }



def _collect_pmi_chain_entity_ids(rec: PMIRecord, entities: Dict[int, str]) -> List[int]:
    ids = set(rec.source_entity_ids)
    ids.update(rec.linked_shape_aspect_ids)
    ids.update(rec.linked_step_face_ids)
    for _eid, raw in entities.items():
        refs = set(entity_refs(raw))
        if refs.intersection(rec.source_entity_ids) or refs.intersection(rec.linked_shape_aspect_ids) or refs.intersection(rec.linked_step_face_ids):
            if (
                "GEOMETRIC_ITEM_SPECIFIC_USAGE(" in raw
                or "SHAPE_ASPECT_RELATIONSHIP(" in raw
                or "DIMENSIONAL_" in raw
                or "DATUM" in raw
                or "GEOMETRIC_TOLERANCE" in raw
            ):
                ids.add(_eid)
    for fid in rec.linked_step_face_ids:
        ids.update(entity_refs(entities.get(fid, "")))
        sid = _surface_ref_for_step_face(fid, entities)
        if sid is not None:
            ids.add(sid)
            ids.update(entity_refs(entities.get(sid, "")))
    return sorted(ids)



def save_face_signature_audit(
    save_prefix: Path,
    step_path: Path,
    entities: Dict[int, str],
    step_faces_raw: Dict[int, StepFaceSignature],
    step_faces_aligned: Dict[int, StepFaceSignature],
    occ_regions: List[OccFaceRegion],
    mesh: trimesh.Trimesh,
    vox: trimesh.voxel.VoxelGrid,
    pmi_records: List[PMIRecord],
    match_candidates: List[FaceMatchCandidate],
    alignment: GeometryAlignment,
) -> Path:
    audit_path = save_prefix.with_name(save_prefix.name + "_face_signature_audit.json")
    candidates_by_step: Dict[int, List[FaceMatchCandidate]] = defaultdict(list)
    for cand in match_candidates:
        candidates_by_step[cand.step_face_id].append(cand)

    step_payload = []
    for fid in sorted(step_faces_raw):
        raw_sig = step_faces_raw[fid]
        aligned_sig = step_faces_aligned[fid]
        surface_id = _surface_ref_for_step_face(fid, entities)
        boundary_trace = collect_step_face_boundary_trace(fid, entities)
        points = collect_step_face_points(fid, entities)
        step_payload.append(
            {
                "step_face_id": fid,
                "raw_advanced_face_entity": entities.get(fid),
                "referenced_surface_entity_id": surface_id,
                "raw_surface_entity": entities.get(surface_id) if surface_id is not None else None,
                "parsed_raw": {
                    "surface_type": raw_sig.surface_type,
                    "centroid": raw_sig.centroid,
                    "bbox_min": raw_sig.bbox_min,
                    "bbox_max": raw_sig.bbox_max,
                    "normal_or_axis": raw_sig.direction,
                    "radius": raw_sig.radius,
                    "area": raw_sig.area_estimate,
                },
                "parsed_aligned_to_occ": {
                    "surface_type": aligned_sig.surface_type,
                    "centroid": aligned_sig.centroid,
                    "bbox_min": aligned_sig.bbox_min,
                    "bbox_max": aligned_sig.bbox_max,
                    "normal_or_axis": aligned_sig.direction,
                    "radius": aligned_sig.radius,
                    "area": aligned_sig.area_estimate,
                },
                "boundary_loop_edge_ids_used": boundary_trace,
                "boundary_vertices_used_raw": points.tolist(),
                "units_assumed": alignment.step_length_unit,
                "orientation_reversed": _orientation_reversed(fid, entities),
                "candidate_occ_regions": [asdict(c) for c in sorted(candidates_by_step.get(fid, []), key=lambda c: c.final_score)],
            }
        )

    occ_payload = []
    pitch_value = float(np.asarray(vox.pitch).reshape(-1)[0])
    for region in occ_regions:
        occ_payload.append(
            {
                "occ_face_index": region.occ_face_index,
                "region_id": region.occ_face_index,
                "surface_type": region.surface_type,
                "centroid": region.centroid,
                "bbox_min": region.bbox_min,
                "bbox_max": region.bbox_max,
                "normal_or_axis": region.direction,
                "radius": region.radius,
                "area": region.area,
                "triangle_count": region.triangle_count,
                "units_assumed": "OCC transferred model units",
                "voxel_transform": np.asarray(vox.transform).tolist(),
                "voxel_bounds": np.asarray(vox.bounds, dtype=np.float64).tolist(),
                "voxel_pitch": pitch_value,
            }
        )

    pmi_traces = []
    for rec in pmi_records:
        ids = _collect_pmi_chain_entity_ids(rec, entities)
        pmi_traces.append(
            {
                "pmi_id": rec.pmi_id,
                "valid_association": rec.valid_association,
                "pmi_record": asdict(rec),
                "entity_chain": [_entity_payload(eid, entities) for eid in ids],
                "referenced_step_faces": [
                    {
                        "step_face_id": fid,
                        "advanced_face": entities.get(fid),
                        "surface_entity_id": _surface_ref_for_step_face(fid, entities),
                        "surface_entity": entities.get(_surface_ref_for_step_face(fid, entities) or -1),
                        "candidate_occ_regions": [asdict(c) for c in sorted(candidates_by_step.get(fid, []), key=lambda c: c.final_score)],
                    }
                    for fid in rec.linked_step_face_ids
                ],
            }
        )

    payload = {
        "input_step_filename": str(step_path),
        "geometry_alignment": asdict(alignment),
        "step_faces": step_payload,
        "occ_regions": occ_payload,
        "pmi_entity_chains": pmi_traces,
        "failed_or_ambiguous_pmi_entity_chains": [t for t in pmi_traces if not t["valid_association"]],
    }
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return audit_path



def visualize_voxels_matplotlib(vox: trimesh.voxel.VoxelGrid, region_grid: Optional[np.ndarray] = None, max_dim: int = 160) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"matplotlib preview is unavailable: {exc}") from exc
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
    parser.add_argument("--pmi-qa", action="store_true", help="Explicitly request PMI QA artifacts (currently generated for every conversion)")
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
    step_faces_raw = copy.deepcopy(step_faces)
    pmi_records: List[PMIRecord] = []
    pmi_records.extend(extract_dimensions(entities, shape_aspect_to_faces))
    pmi_records.extend(extract_geometric_tolerances(entities, shape_aspect_to_faces))
    pmi_records.extend(extract_standalone_datums(entities, shape_aspect_to_faces))
    print(f"      parsed entities={len(entities)} step_faces={len(step_faces)} pmi_records={len(pmi_records)}")

    alignment = infer_geometry_alignment(step_faces, entities, occ_regions, mesh)
    apply_geometry_alignment_to_step_faces(step_faces, alignment)
    if alignment.alignment_applied:
        print(
            "      geometry alignment applied: "
            f"scale={alignment.scale_step_to_occ:g} translation={alignment.translation_step_to_occ}"
        )
    else:
        print(f"      geometry alignment not applied: {alignment.reason}")

    print("[6/7] Matching STEP faces to OCC face regions and building region grid...")
    step_to_occ, match_results, match_candidates = match_step_faces_to_occ_regions(step_faces, occ_regions)
    annotate_pmi_records(pmi_records, step_to_occ, match_results)
    region_grid, region_to_voxels = build_region_id_grid(vox, mesh, triangle_region_ids, len(occ_regions), args.pitch)
    matched_pmi = sum(1 for r in pmi_records if r.linked_region_ids)
    ambiguous_faces = sum(1 for r in match_results.values() if r.confidence == "ambiguous")
    failed_faces = sum(1 for r in match_results.values() if r.confidence == "failed")
    invalid_pmi = sum(1 for r in pmi_records if not r.valid_association)
    print(
        f"      step_face_matches={len(step_to_occ)} "
        f"ambiguous_faces={ambiguous_faces} failed_faces={failed_faces} "
        f"pmi_with_region_links={matched_pmi} invalid_pmi={invalid_pmi}"
    )

    print("[7/7] Saving outputs...")
    npz_path, json_path = save_outputs(save_prefix, vox, region_grid, occ_regions, step_faces, pmi_records, region_to_voxels)
    diagnostics_path = save_diagnostics(
        save_prefix,
        step_path,
        entities,
        vox,
        region_grid,
        step_faces,
        occ_regions,
        pmi_records,
        step_to_occ,
        match_results,
        match_candidates,
    )
    audit_path = save_face_signature_audit(
        save_prefix,
        step_path,
        entities,
        step_faces_raw,
        step_faces,
        occ_regions,
        mesh,
        vox,
        pmi_records,
        match_candidates,
        alignment,
    )
    pmi_qa_path = save_pmi_qa_report(
        save_prefix,
        step_path,
        entities,
        pmi_records,
        occ_regions,
        vox,
        region_grid,
        match_results,
    )
    capability_json_path, capability_md_path = save_pmi_capability_matrix(Path(__file__).parent)
    support_json_path, support_md_path = save_project_pmi_support_profile(Path(__file__).parent)
    print(f"      saved voxel+region grid: {npz_path}")
    print(f"      saved semantic sidecar: {json_path}")
    print(f"      saved mapping diagnostics: {diagnostics_path}")
    print(f"      saved face signature audit: {audit_path}")
    print(f"      saved PMI QA report: {pmi_qa_path}")
    print(f"      saved PMI capability matrix: {capability_json_path}, {capability_md_path}")
    print(f"      saved project PMI support profile: {support_json_path}, {support_md_path}")
    if invalid_pmi:
        print(f"      WARNING: {invalid_pmi} PMI records failed semantic QA; see diagnostics.")

    if not args.no_preview:
        try:
            visualize_voxels_matplotlib(vox, region_grid=region_grid)
        except Exception as exc:
            print(f"Preview skipped: {exc}")


if __name__ == "__main__":
    main()
