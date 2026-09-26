import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EVIDENCE = Path("evidence").resolve()
EVIDENCE.mkdir(exist_ok=True)
original = (HERE / "kernel.py").read_text()
corrected = original.replace("rk2[:, None] * 64 * NBLK", "rk2[:, None] * 128 * NBLK")
carried = corrected.replace(
    "    for blk in", "    acc = tl.zeros((128, 128), tl.float32)\n    for blk in"
).replace('out_dtype=tl.float32)', 'acc=acc, out_dtype=tl.float32)')
cases = {
    "posted": original,
    "corrected_stride": corrected,
    "stage_one": corrected.replace("num_stages=2", "num_stages=1"),
    "carried": carried,
    "static_unroll": corrected.replace("tl.range(0, NBLK, num_stages=2)", "tl.static_range(0, NBLK)"),
}
compile_script = '''
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from kernel import _loop_dot_scaled
signature = {"lhs": "*u8", "lhs_sf": "*u8", "rhs": "*u8", "rhs_sf": "*u8", "out": "*fp32"}
source = triton.compiler.ASTSource(
    _loop_dot_scaled, signature, constexprs={"NBLK": 8},
    attrs={(i,): [["tt.divisibility", 16]] for i in range(5)},
)
kernel = triton.compile(source, target=GPUTarget("cuda", 100, 32), options={"num_warps": 4, "num_ctas": 1})
for extension, output in kernel.asm.items():
    path = Path("compiled." + extension)
    path.write_bytes(output) if isinstance(output, bytes) else path.write_text(str(output))
print("COMPILED_TO_CUBIN", len(kernel.asm["cubin"]))
'''
results = []
for name, source in cases.items():
    directory = EVIDENCE / name
    directory.mkdir(exist_ok=True)
    (directory / "kernel.py").write_text(source)
    (directory / "compile.py").write_text(compile_script)
    env = dict(os.environ, TRITON_ALWAYS_COMPILE="1", MLIR_ENABLE_DUMP="1",
               TRITON_REPRODUCER_PATH=str(directory / "reproducer.mlir"))
    with (directory / "stdout.txt").open("w") as out, (directory / "stderr.txt").open("w") as err:
        try:
            process = subprocess.run([sys.executable, "compile.py"], cwd=directory, env=env,
                                     stdout=out, stderr=err, timeout=180)
            status = process.returncode
        except subprocess.TimeoutExpired:
            status = "timeout"
    row = {"case": name, "returncode": status, "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
    results.append(row)
    print(json.dumps(row), flush=True)

provenance = {"source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "python": sys.version, "results": results}
(EVIDENCE / "results.json").write_text(json.dumps(provenance, indent=2))
if not any(row["returncode"] == 0 for row in results):
    raise SystemExit("No positive control compiled; inspect environment and failure stages")
