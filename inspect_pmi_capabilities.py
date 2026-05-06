#!/usr/bin/env python3
"""Run Step2Vox conversion and emit PMI QA/capability artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Step2Vox PMI QA diagnostics for one STEP file.")
    parser.add_argument("step_file", type=Path)
    parser.add_argument("--pitch", default="1.0")
    parser.add_argument("--linear-deflection", default="0.1")
    parser.add_argument("--angular-deflection", default="0.3")
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--save-prefix", type=Path, default=None)
    args = parser.parse_args()

    script = Path(__file__).with_name("step_ap242_to_voxel_semantic_layer.py")
    cmd = [
        sys.executable,
        str(script),
        str(args.step_file),
        "--pitch",
        args.pitch,
        "--linear-deflection",
        args.linear_deflection,
        "--angular-deflection",
        args.angular_deflection,
        "--no-preview",
    ]
    if args.fill:
        cmd.append("--fill")
    if args.save_prefix:
        cmd.extend(["--save-prefix", str(args.save_prefix)])

    subprocess.run(cmd, check=True)

    save_prefix = args.save_prefix or args.step_file.with_name(args.step_file.stem + "_voxel_semantic")
    print("\nPMI QA artifacts:")
    print(f"  diagnostics:           {save_prefix.with_name(save_prefix.name + '_diagnostics.json')}")
    print(f"  face signature audit:  {save_prefix.with_name(save_prefix.name + '_face_signature_audit.json')}")
    print(f"  PMI QA report:         {save_prefix.with_name(save_prefix.name + '_pmi_qa_report.json')}")
    print(f"  capability matrix:     {Path(__file__).with_name('pmi_capability_matrix.json')}")
    print(f"  capability matrix md:  {Path(__file__).with_name('pmi_capability_matrix.md')}")


if __name__ == "__main__":
    main()
