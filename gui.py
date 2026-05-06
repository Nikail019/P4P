#!/usr/bin/env python3
"""
GUI for the Step2Vox pipeline.
Run with: conda run -n ap242 python gui.py
"""

import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


class Step2VoxGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Step2Vox")
        self.resizable(True, True)
        self.minsize(700, 580)

        self.step_path = tk.StringVar()
        self.pitch = tk.StringVar(value="1.0")
        self.fill = tk.BooleanVar(value=False)
        self.linear_def = tk.StringVar(value="0.1")
        self.angular_def = tk.StringVar(value="0.3")
        self.viz_mode = tk.StringVar(value="mbd")
        self.inspect_full = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        pad = dict(padx=10, pady=5)

        # --- File selection ---
        file_frame = ttk.LabelFrame(self, text="Input STEP File")
        file_frame.pack(fill=tk.X, **pad)

        ttk.Entry(file_frame, textvariable=self.step_path, width=60).pack(
            side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True
        )
        ttk.Button(file_frame, text="Browse...", command=self._browse).pack(
            side=tk.LEFT, padx=5, pady=5
        )

        # --- Options ---
        opt_frame = ttk.LabelFrame(self, text="Voxelization Options")
        opt_frame.pack(fill=tk.X, **pad)

        ttk.Label(opt_frame, text="Pitch:").grid(row=0, column=0, padx=5, pady=4, sticky=tk.W)
        ttk.Entry(opt_frame, textvariable=self.pitch, width=8).grid(row=0, column=1, padx=5, pady=4, sticky=tk.W)

        ttk.Label(opt_frame, text="Linear deflection:").grid(row=0, column=2, padx=5, pady=4, sticky=tk.W)
        ttk.Entry(opt_frame, textvariable=self.linear_def, width=8).grid(row=0, column=3, padx=5, pady=4, sticky=tk.W)

        ttk.Label(opt_frame, text="Angular deflection:").grid(row=0, column=4, padx=5, pady=4, sticky=tk.W)
        ttk.Entry(opt_frame, textvariable=self.angular_def, width=8).grid(row=0, column=5, padx=5, pady=4, sticky=tk.W)

        ttk.Checkbutton(opt_frame, text="Fill solid", variable=self.fill).grid(
            row=0, column=6, padx=10, pady=4, sticky=tk.W
        )

        # --- Voxelize button ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, **pad)

        self.vox_btn = ttk.Button(btn_frame, text="Voxelize", command=self._run_voxelize)
        self.vox_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=5)

        # --- Log ---
        log_frame = ttk.LabelFrame(self, text="Output Log")
        log_frame.pack(fill=tk.BOTH, expand=True, **pad)

        self.log = tk.Text(
            log_frame, height=12, state=tk.DISABLED,
            bg="#1e1e1e", fg="#d4d4d4", font=("Courier", 11), wrap=tk.WORD,
        )
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Inspect & Visualize ---
        post_frame = ttk.LabelFrame(self, text="Inspect & Visualize")
        post_frame.pack(fill=tk.X, **pad)

        self.inspect_btn = ttk.Button(
            post_frame, text="Inspect JSON", command=self._run_inspect, state=tk.DISABLED
        )
        self.inspect_btn.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Checkbutton(post_frame, text="Full detail", variable=self.inspect_full).pack(
            side=tk.LEFT, padx=(0, 10), pady=5
        )

        ttk.Label(post_frame, text="View mode:").pack(side=tk.LEFT, padx=(15, 2), pady=5)
        ttk.Combobox(
            post_frame, textvariable=self.viz_mode, width=12,
            values=["mbd", "regions", "pmi", "explode", "occupancy"], state="readonly",
        ).pack(side=tk.LEFT, padx=5, pady=5)

        self.viz_btn = ttk.Button(
            post_frame, text="Visualize", command=self._run_visualize, state=tk.DISABLED
        )
        self.viz_btn.pack(side=tk.LEFT, padx=5, pady=5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select STEP file",
            filetypes=[("STEP files", "*.step *.stp"), ("All files", "*.*")],
        )
        if path:
            self.step_path.set(path)

    def _append_log(self, text: str):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.config(state=tk.DISABLED)

    def _npz_path(self) -> Path:
        p = Path(self.step_path.get())
        return p.with_name(p.stem + "_voxel_semantic.npz")

    def _json_path(self) -> Path:
        p = Path(self.step_path.get())
        return p.with_name(p.stem + "_voxel_semantic.json")

    def _stream(self, cmd, on_done=None):
        """Run cmd in a background thread, streaming stdout/stderr to the log."""
        def _run():
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(SCRIPT_DIR),
                )
                for line in proc.stdout:
                    self.after(0, self._append_log, line)
                proc.wait()
                if on_done:
                    self.after(0, on_done, proc.returncode)
            except Exception as exc:
                self.after(0, self._append_log, f"ERROR: {exc}\n")
                if on_done:
                    self.after(0, on_done, 1)

        threading.Thread(target=_run, daemon=True).start()

    def _open_in_terminal(self, cmd):
        """Open a command in a new macOS Terminal window (for rich output)."""
        shell_cmd = " ".join(f'"{c}"' for c in cmd)
        apple_script = f'tell application "Terminal" to do script "{shell_cmd}"'
        subprocess.Popen(["osascript", "-e", apple_script])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _run_voxelize(self):
        step = self.step_path.get().strip()
        if not step or not Path(step).exists():
            messagebox.showerror("Error", "Please select a valid STEP file.")
            return

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "step_ap242_to_voxel_semantic_layer.py"),
            step,
            "--pitch", self.pitch.get(),
            "--linear-deflection", self.linear_def.get(),
            "--angular-deflection", self.angular_def.get(),
            "--no-preview",
        ]
        if self.fill.get():
            cmd.append("--fill")

        self.vox_btn.config(state=tk.DISABLED)
        self.inspect_btn.config(state=tk.DISABLED)
        self.viz_btn.config(state=tk.DISABLED)
        self._append_log("--- Starting voxelization ---\n")

        def on_done(returncode):
            self.vox_btn.config(state=tk.NORMAL)
            if returncode == 0:
                self._append_log("--- Done ---\n")
                self.inspect_btn.config(state=tk.NORMAL)
                self.viz_btn.config(state=tk.NORMAL)
            else:
                self._append_log(f"--- Failed (exit code {returncode}) ---\n")

        self._stream(cmd, on_done)

    def _run_inspect(self):
        json_path = self._json_path()
        if not json_path.exists():
            messagebox.showerror("Error", f"JSON file not found:\n{json_path}")
            return
        cmd = [sys.executable, str(SCRIPT_DIR / "inspect_json.py"), str(json_path)]
        if self.inspect_full.get():
            cmd.append("--full")
        self._open_in_terminal(cmd)

    def _run_visualize(self):
        npz_path = self._npz_path()
        json_path = self._json_path()
        if not npz_path.exists():
            messagebox.showerror("Error", f"NPZ file not found:\n{npz_path}")
            return

        mode = self.viz_mode.get()
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "view3d.py"),
            str(npz_path),
            "--mode", mode,
        ]
        if mode in ("pmi", "explode", "mbd"):
            if not json_path.exists():
                messagebox.showerror("Error", f"JSON file not found:\n{json_path}")
                return
            cmd += ["--json", str(json_path)]

        self._append_log(f"--- Launching view3d --mode {mode} ---\n")
        self._stream(cmd)


if __name__ == "__main__":
    app = Step2VoxGUI()
    app.mainloop()
