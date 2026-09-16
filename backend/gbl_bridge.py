"""
gbl_bridge.py

Wraps the user's real `build_gbl_pipeline.py` (Silicon Labs compiler.py + Simplicity
Commander) to turn an approved .tflite into a .gbl OTA firmware image.

IMPORTANT — this cannot be a simple "copy our vendored script and run it" integration:
build_gbl_pipeline.py resolves its own PROJECT_ROOT as `Path(__file__).resolve().parent`
and expects sibling directories next to itself: `config/tflite/`, `cmake_gcc/build/`,
`autogen/`. Those live inside the user's actual Simplicity Studio project — they are NOT
part of this Python app. So this bridge does NOT run the copy vendored under
backend/vendor/; it shells out to whatever real script path is configured in
config/app_config.json (`gbl_pipeline_script_path`), which must point at the real
build_gbl_pipeline.py sitting inside the actual firmware project, on a machine that has
the Silicon Labs SDK + Simplicity Commander installed (Windows, per that script).

If that isn't available (e.g. running this app on Linux/macOS, or during development
before the firmware project is wired up), we fall back to a MOCK build: a placeholder
file is written and clearly labeled as such in the returned build info. Nothing here
pretends a mock output is a real, flashable .gbl.
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


class GblBuildResult(dict):
    """Just a typed dict: {mode, success, gbl_path, log, note}."""


def _load_app_config(app_config_path: str) -> Dict:
    p = Path(app_config_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _real_mode_available(script_path: Optional[str]) -> bool:
    if not script_path:
        return False
    sp = Path(script_path)
    if not sp.exists():
        return False
    # build_gbl_pipeline.py is written for Windows (commander.exe, Windows home paths).
    # We don't hard-require platform.system() == "Windows" here (in case someone adapts
    # the script for another OS later) but we do require the script to actually exist.
    return True


def convert_to_gbl(
    tflite_path: str,
    output_gbl_path: str,
    app_config_path: str = "config/app_config.json",
    timeout_sec: int = 1800,
) -> GblBuildResult:
    tflite_path = str(Path(tflite_path).resolve())
    output_gbl_path = str(Path(output_gbl_path).resolve())
    Path(output_gbl_path).parent.mkdir(parents=True, exist_ok=True)

    cfg = _load_app_config(app_config_path)
    script_path = cfg.get("gbl_pipeline_script_path")

    if _real_mode_available(script_path):
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path), tflite_path, output_gbl_path],
                capture_output=True, text=True, timeout=timeout_sec,
            )
            log = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode == 0 and Path(output_gbl_path).exists():
                return GblBuildResult(
                    mode="real", success=True, gbl_path=output_gbl_path,
                    log=log[-8000:],
                    note="Built via the real Silicon Labs toolchain (build_gbl_pipeline.py).",
                )
            return GblBuildResult(
                mode="real", success=False, gbl_path=None,
                log=log[-8000:],
                note=f"Real build script ran but failed (exit code {proc.returncode}). "
                     f"See log. Falling back to no firmware output for this version.",
            )
        except subprocess.TimeoutExpired:
            return GblBuildResult(
                mode="real", success=False, gbl_path=None, log="",
                note=f"Real build timed out after {timeout_sec}s.",
            )
        except Exception as e:
            return GblBuildResult(
                mode="real", success=False, gbl_path=None, log=str(e),
                note=f"Real build raised an exception: {e}",
            )

    # ---- Mock fallback ----
    placeholder = (
        b"SOMNIGUARD_MOCK_GBL\x00"
        + f"source_tflite={Path(tflite_path).name};built_at={time.time()}".encode("utf-8")
    )
    with open(output_gbl_path, "wb") as f:
        f.write(placeholder)

    reason = (
        "No gbl_pipeline_script_path configured or found."
        if not script_path else
        f"Configured script path does not exist: {script_path}"
    )
    return GblBuildResult(
        mode="mock", success=True, gbl_path=output_gbl_path, log="",
        note=(
            f"{reason} Wrote a placeholder file instead of a real .gbl. "
            f"To produce a real, flashable image: run this app on the Windows machine with "
            f"the Silicon Labs SDK + Simplicity Commander installed, and set "
            f"config/app_config.json -> gbl_pipeline_script_path to the build_gbl_pipeline.py "
            f"that lives inside your actual firmware project (next to its config/tflite, "
            f"cmake_gcc/build, autogen directories)."
        ),
    )
