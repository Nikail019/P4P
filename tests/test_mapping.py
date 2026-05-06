import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from step_ap242_to_voxel_semantic_layer import (
    FaceMatchResult,
    OccFaceRegion,
    PMIRecord,
    StepFaceSignature,
    annotate_pmi_records,
    apply_geometry_alignment_to_step_faces,
    extract_fit_info,
    infer_geometry_alignment,
    match_step_faces_to_occ_regions,
    parse_surface_signature,
    save_diagnostics,
)


def occ_region(region_id, surface_type, centroid, direction=(0.0, 0.0, 1.0), radius=None):
    x, y, z = centroid
    return OccFaceRegion(
        occ_face_index=region_id,
        surface_type=surface_type,
        origin=centroid,
        direction=direction,
        radius=radius,
        area=100.0,
        bbox_min=(x - 1.0, y - 1.0, z - 0.1),
        bbox_max=(x + 1.0, y + 1.0, z + 0.1),
        centroid=centroid,
        triangle_count=2,
    )


def step_face(face_id, surface_type, centroid, direction=(0.0, 0.0, 1.0), radius=None):
    x, y, z = centroid
    return StepFaceSignature(
        step_face_id=face_id,
        surface_type=surface_type,
        origin=centroid,
        direction=direction,
        radius=radius,
        orientation_flag=".T.",
        centroid=centroid,
        bbox_min=(x - 1.0, y - 1.0, z - 0.1),
        bbox_max=(x + 1.0, y + 1.0, z + 0.1),
    )


class MappingMatcherTests(unittest.TestCase):
    def test_confident_single_planar_face_match(self):
        step_faces = {101: step_face(101, "plane", (0.0, 0.0, 0.0))}
        occ_regions = [
            occ_region(1, "plane", (0.0, 0.0, 0.0)),
            occ_region(2, "plane", (20.0, 0.0, 0.0)),
        ]

        matches, results, candidates = match_step_faces_to_occ_regions(step_faces, occ_regions)

        self.assertEqual(matches, {101: 1})
        self.assertEqual(results[101].confidence, "exact")
        self.assertTrue(any(c.selected and c.occ_face_index == 1 for c in candidates))

    def test_repeated_geometry_is_marked_ambiguous(self):
        step_faces = {101: step_face(101, "plane", (0.0, 0.0, 0.0))}
        occ_regions = [
            occ_region(1, "plane", (0.0, 0.0, 0.0)),
            occ_region(2, "plane", (0.01, 0.0, 0.0)),
        ]

        matches, results, _candidates = match_step_faces_to_occ_regions(step_faces, occ_regions)

        self.assertEqual(matches, {})
        self.assertEqual(results[101].confidence, "ambiguous")
        self.assertIsNone(step_faces[101].matched_occ_face_index)

    def test_cylinder_radius_contributes_to_selection(self):
        step_faces = {201: step_face(201, "cylinder", (0.0, 0.0, 0.0), radius=5.0)}
        occ_regions = [
            occ_region(11, "cylinder", (0.0, 0.0, 0.0), radius=5.0),
            occ_region(12, "cylinder", (0.0, 0.0, 0.0), radius=8.0),
        ]

        matches, results, _candidates = match_step_faces_to_occ_regions(step_faces, occ_regions)

        self.assertEqual(matches, {201: 11})
        self.assertIn(results[201].confidence, {"exact", "high"})

    def test_pmi_is_invalid_when_step_face_match_is_ambiguous(self):
        pmi = PMIRecord(
            pmi_id="gtol_1",
            category="geometric_tolerance",
            subtype="position_tolerance",
            label=None,
            value=0.02,
            unit="mm",
            modifiers=[],
            datum_refs=[],
            source_entity_ids=[1],
            linked_shape_aspect_ids=[10],
            linked_step_face_ids=[101],
            linked_occ_face_indices=[],
            linked_region_ids=[],
            valid_association=True,
            association_method="direct_shape_aspect_lookup",
            notes=[],
        )
        results = {
            101: FaceMatchResult(
                step_face_id=101,
                selected_occ_face_index=None,
                selected_score=0.01,
                confidence="ambiguous",
                reason="synthetic ambiguity",
                candidate_occ_face_indices=[1, 2],
            )
        }

        annotate_pmi_records([pmi], {}, results)

        self.assertFalse(pmi.valid_association)
        self.assertEqual(pmi.linked_region_ids, [])
        self.assertIn("safe OCC face-region mapping failed", pmi.notes[0])

    def test_diagnostics_file_contains_semantic_qa_gate_and_counts(self):
        class FakeVoxelGrid:
            matrix = np.array([[[1, 0], [1, 1]]], dtype=bool)
            pitch = 1.0
            transform = np.eye(4)
            bounds = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])

        step_faces = {101: step_face(101, "plane", (0.0, 0.0, 0.0))}
        occ_regions = [occ_region(1, "plane", (0.0, 0.0, 0.0))]
        matches, results, candidates = match_step_faces_to_occ_regions(step_faces, occ_regions)
        pmi = PMIRecord(
            pmi_id="dimension_1",
            category="dimension",
            subtype="dimensional_location",
            label="test",
            value=1.0,
            unit="mm",
            modifiers=[],
            datum_refs=[],
            source_entity_ids=[1],
            linked_shape_aspect_ids=[10],
            linked_step_face_ids=[101],
            linked_occ_face_indices=[],
            linked_region_ids=[],
            valid_association=True,
            association_method="direct_shape_aspect_lookup",
            notes=[],
        )
        annotate_pmi_records([pmi], matches, results)
        region_grid = np.array([[[1, 0], [1, 0]]], dtype=np.int32)

        with TemporaryDirectory() as tmp:
            out = save_diagnostics(
                Path(tmp) / "fixture",
                Path("synthetic.step"),
                {},
                FakeVoxelGrid(),
                region_grid,
                step_faces,
                occ_regions,
                [pmi],
                matches,
                results,
                candidates,
            )

            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn('"semantic_qa_gate"', text)
            self.assertIn('"unlabeled_occupied_voxels": 1', text)

    def test_cylindrical_surface_radius_parses_last_surface_argument(self):
        entities = {
            1: "CARTESIAN_POINT('',(0.,0.,0.))",
            2: "DIRECTION('',(0.,0.,1.))",
            3: "DIRECTION('',(1.,0.,0.))",
            4: "AXIS2_PLACEMENT_3D('',#1,#2,#3)",
            5: "CYLINDRICAL_SURFACE('',#4,0.015)",
        }

        surface_type, origin, direction, radius = parse_surface_signature(5, entities)

        self.assertEqual(surface_type, "cylinder")
        self.assertEqual(origin, (0.0, 0.0, 0.0))
        self.assertEqual(direction, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(radius, 0.015)

    def test_geometry_alignment_scales_step_metres_to_occ_millimetres(self):
        class FakeMesh:
            vertices = np.array([[-25.0, -25.0, 0.0], [25.0, 25.0, 50.0]])

        entities = {
            1: "GLOBAL_UNIT_ASSIGNED_CONTEXT((#2))",
            2: "(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.))",
            10: "ADVANCED_FACE('',(#11),#20,.F.)",
            11: "FACE_BOUND('',#12,.T.)",
            12: "EDGE_LOOP('',(#13,#14,#15,#16))",
            13: "ORIENTED_EDGE('',*,*,#23,.T.)",
            14: "ORIENTED_EDGE('',*,*,#24,.T.)",
            15: "ORIENTED_EDGE('',*,*,#25,.T.)",
            16: "ORIENTED_EDGE('',*,*,#26,.T.)",
            23: "EDGE_CURVE('',#33,#34,#43,.T.)",
            24: "EDGE_CURVE('',#34,#35,#44,.T.)",
            25: "EDGE_CURVE('',#35,#36,#45,.T.)",
            26: "EDGE_CURVE('',#36,#33,#46,.T.)",
            33: "VERTEX_POINT('',#53)",
            34: "VERTEX_POINT('',#54)",
            35: "VERTEX_POINT('',#55)",
            36: "VERTEX_POINT('',#56)",
            43: "LINE('',#53,#63)",
            44: "LINE('',#54,#64)",
            45: "LINE('',#55,#65)",
            46: "LINE('',#56,#66)",
            53: "CARTESIAN_POINT('',(-0.025,-0.025,0.0))",
            54: "CARTESIAN_POINT('',(0.025,-0.025,0.0))",
            55: "CARTESIAN_POINT('',(0.025,0.025,0.05))",
            56: "CARTESIAN_POINT('',(-0.025,0.025,0.05))",
            63: "VECTOR('',#73,1.)",
            64: "VECTOR('',#74,1.)",
            65: "VECTOR('',#75,1.)",
            66: "VECTOR('',#76,1.)",
            73: "DIRECTION('',(1.,0.,0.))",
            74: "DIRECTION('',(0.,1.,0.))",
            75: "DIRECTION('',(-1.,0.,0.))",
            76: "DIRECTION('',(0.,-1.,0.))",
        }
        step_faces = {
            10: StepFaceSignature(
                step_face_id=10,
                surface_type="cylinder",
                origin=(0.0, 0.0, 0.0),
                direction=(0.0, 0.0, 1.0),
                radius=0.015,
                orientation_flag=".F.",
                centroid=(0.015, 0.0, 0.025),
                bbox_min=(-0.025, -0.025, 0.0),
                bbox_max=(0.025, 0.025, 0.05),
            )
        }
        occ_regions = [occ_region(1, "cylinder", (15.0, 0.0, 25.0), radius=15.0)]
        occ_regions[0].bbox_min = (-25.0, -25.0, 0.0)
        occ_regions[0].bbox_max = (25.0, 25.0, 50.0)

        alignment = infer_geometry_alignment(step_faces, entities, occ_regions, FakeMesh())
        apply_geometry_alignment_to_step_faces(step_faces, alignment)

        self.assertTrue(alignment.alignment_applied)
        self.assertAlmostEqual(alignment.scale_step_to_occ, 1000.0)
        self.assertEqual(step_faces[10].centroid, (15.0, 0.0, 25.0))
        self.assertAlmostEqual(step_faces[10].radius, 15.0)

    def test_fit_info_extracts_it_and_h7_patterns(self):
        self.assertEqual(extract_fit_info("hole diameter 10 H7")["fit_class"], "H7")
        self.assertEqual(extract_fit_info("IT8 tolerance")["it_grade"], 8)


if __name__ == "__main__":
    unittest.main()
