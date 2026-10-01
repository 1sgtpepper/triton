"""Run one exact fork-built Triton wheel's focused WarpSpecialization test.

Set WS_WHEEL, WS_TEST_FILE, WS_TEST_NAME, and WS_RESULT_DIR before modal run.
The image is built without a GPU. The only GPU function uses one Hopper
accelerator, has a five-minute deadline, and is not deployed.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import modal

if modal.is_local() and "WS_WHEEL" in os.environ:
    wheel = Path(os.environ["WS_WHEEL"]).resolve()
    test_file = Path(os.environ["WS_TEST_FILE"]).resolve()
    test_name = os.environ["WS_TEST_NAME"]
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    test_sha256 = hashlib.sha256(test_file.read_bytes()).hexdigest()
    image = (
        modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
        .apt_install("build-essential")
        .pip_install("torch==2.8.0", "pytest==8.4.2", "numpy==2.2.6")
        .add_local_file(wheel, f"/opt/{wheel.name}", copy=True)
        .run_commands(f"python -m pip install --no-deps --force-reinstall /opt/{wheel.name}")
        .run_commands("python -c 'import triton, triton._internal_testing; print(triton.__version__)'")
        .add_local_file(test_file, "/opt/test_warp_specialization.py", copy=True)
    )
else:
    image = modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
app = modal.App("warp-partition-validation")


@app.function(image=image, gpu="H100", timeout=300, max_containers=1,
              scaledown_window=2, cpu=2, memory=8192)
def validate(test_name: str, wheel_name: str):
    device = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
         "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-s", "--tb=short",
         f"/opt/test_warp_specialization.py::{test_name}"],
        text=True, capture_output=True, timeout=270,
    )
    return {
        "device": device,
        "wheel_sha256": hashlib.sha256(Path(f"/opt/{wheel_name}").read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path("/opt/test_warp_specialization.py").read_bytes()).hexdigest(),
        "exit_status": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.local_entrypoint()
def main():
    result = validate.remote(test_name, wheel.name)
    assert result["wheel_sha256"] == wheel_sha256
    assert result["test_sha256"] == test_sha256
    output = Path(os.environ["WS_RESULT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if result["exit_status"]:
        raise SystemExit(result["exit_status"])
