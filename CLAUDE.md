# Step2Vox Project Context

This file is shared project memory for Claude/Codex-style coding sessions. Keep it current when architecture, commands, dependencies, or workflow assumptions change.

## Project Purpose

Step2Vox converts STEP/AP242 CAD files into voxel occupancy data plus a separate semantic/PMI sidecar.

The current pipeline aims to:

- Read STEP files with Open CASCADE via `pythonocc-core`.
- Triangulate B-rep geometry and voxelize it with `trimesh`/`numpy`.
- Reconstruct face regions and map them back to STEP `ADVANCED_FACE` records where possible.
- Parse a lightweight subset of AP242 semantic PMI into JSON.
- Visualize occupancy, regions, PMI, exploded views, and MBD-style overlays.

## Important Limitations

- AP242 PMI is vendor-dependent; parsing is intentionally best-effort and incomplete.
- STEP face to OCC face matching is heuristic, strongest for planar and cylindrical models.
- Symmetric/repeated features and freeform surfaces can produce ambiguous mappings.
- Generated `.npz` and `.json` outputs are artifacts, not source-of-truth code.

## Main Files

- `step_ap242_to_voxel_semantic_layer.py`: core STEP -> mesh -> voxel -> semantic JSON pipeline.
- `gui.py`: Tkinter GUI wrapper for voxelization, inspection, and visualization.
- `view3d.py`: interactive 3D viewer with modes `occupancy`, `regions`, `pmi`, `explode`, and `mbd`.
- `visualize_voxel.py`: simpler voxel visualization utility.
- `inspect_json.py`: terminal inspection tool for semantic JSON sidecars.
- `Instructions.txt`: original quick-start commands.
- `Archives/`, `DevonTest/`, `*.step`, `*_voxel_semantic.*`: local fixtures and generated examples.

## Run Commands

Preferred environment from the existing notes:

```bash
conda run -n ap242 python step_ap242_to_voxel_semantic_layer.py YourFile.step
conda run -n ap242 python inspect_json.py YourFile_voxel_semantic.json
conda run -n ap242 python view3d.py YourFile_voxel_semantic.npz --mode mbd --json YourFile_voxel_semantic.json
conda run -n ap242 python gui.py
```

Direct local virtualenv may also exist at `.venv/`, but verify dependencies before relying on it:

```bash
.venv/bin/python -c "import OCC, trimesh, numpy, matplotlib; print('deps ok')"
```

Example fixture commands:

```bash
conda run -n ap242 python step_ap242_to_voxel_semantic_layer.py "Square_Hole.step" --no-preview
conda run -n ap242 python view3d.py "Square_Hole_voxel_semantic.npz" --mode mbd --json "Square_Hole_voxel_semantic.json"
```

## Development Notes

- Prefer small, focused edits; this is currently a script-oriented prototype rather than a packaged app.
- Preserve generated fixture files unless the task explicitly involves regenerating or replacing them.
- Be careful with filenames containing spaces, for example `Double Datum.step`.
- GUI subprocesses currently use `sys.executable`, so behavior depends on the Python used to launch `gui.py`.
- `gui.py` uses macOS `osascript` to open inspect output in Terminal.
- Visualization and preview commands may require a local display/session.
- The git worktree may include user changes; do not reset or revert unrelated files.

## Semantic Mapping Chain

Current semantic mapping is:

```text
AP242 PMI entity
  -> SHAPE_ASPECT / GEOMETRIC_ITEM_SPECIFIC_USAGE references
  -> STEP ADVANCED_FACE ids
  -> scored STEP-face-to-OCC-region candidates
  -> conservative one-to-one OCC region assignment
  -> voxel region_id_grid labels
  -> future tensor channels only if diagnostics mark the PMI valid
```

The core implementation locations are:

- PMI extraction: `extract_dimensions`, `extract_geometric_tolerances`, and `extract_standalone_datums` in `step_ap242_to_voxel_semantic_layer.py`.
- STEP `ADVANCED_FACE` extraction: `extract_step_face_signatures`.
- OCC face-region extraction: `occ_shape_to_trimesh_with_regions`.
- STEP-to-OCC matching: `match_step_faces_to_occ_regions` and `_build_face_match_candidates`.
- Voxel region labelling: `build_region_id_grid`.
- PMI overlay/debug visualization: `show_pmi`, `show_mbd`, and `show_debug_mapping` in `view3d.py`.

Every conversion now writes `<save-prefix>_diagnostics.json`. This file is the source of truth for mapping correctness. The viewer is only a debugging aid.

Every conversion also writes `<save-prefix>_face_signature_audit.json`. This captures raw/aligned STEP face signatures, OCC region signatures, geometry bbox/unit alignment, PMI entity chains, and per-face candidate matches.

Every conversion also writes `<save-prefix>_pmi_qa_report.json`, plus repo-level `pmi_capability_matrix.json` and `pmi_capability_matrix.md`. These separate semantic mapping readiness from MBD display quality. Display issues must not be treated as semantic mapping failures without diagnostics evidence.

Fail-safe policy: PMI records with `failed` or `ambiguous` mapping confidence must not be embedded into voxel tensor channels or used as valid ML experiment labels.

Current real-fixture root cause found on 2026-05-06: parsed STEP coordinates/radii were in metres while OCC/voxel geometry was in millimetres. The matcher now infers a consistent bbox scale/translation, applies it to STEP face signatures before matching, and records the evidence in the face signature audit. Do not loosen thresholds to compensate for unit/coordinate mismatch.

## Quality Priorities

Near-term refinements should focus on:

- Reliable setup/run instructions and dependency detection.
- Better error messages when OCC, `trimesh`, display backends, or input files fail.
- Cleaner GUI behavior around long-running subprocesses, invalid numeric inputs, and output path discovery.
- Tests or smoke checks for parsing, output schemas, and fixture conversions.
- Reducing duplication between `view3d.py` and `visualize_voxel.py` if their responsibilities continue to overlap.
- Synthetic AP242 fixtures with ground-truth metadata are still needed for end-to-end mapping validation; no synthetic STEP generator is currently present in this repo.

## Session Handoff

When ending a substantial session, update this file with:

- Commands that now work or fail.
- Any new dependencies or environment assumptions.
- Important behavior changes.
- Known bugs or follow-up tasks.
