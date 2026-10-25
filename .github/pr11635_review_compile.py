"""Compile the exact PR fixtures without a GPU and audit their numerical oracles."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import textwrap

import numpy as np
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--architectures", type=int, nargs="+", default=[100, 103, 107])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = Path("python/test/gluon/test_core.py")
    source = fixture.read_text()
    tree = ast.parse(source)
    imports = """from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout, TensorMemoryScalesLayout, allocate_tensor_memory, tcgen05_mma_scaled
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, fence_async_shared
"""
    records = []
    for name in ("test_tcgen05_mma_scaled_sliced_a_scales", "test_tcgen05_mma_scaled_two_ctas_repeated_m"):
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        kernel = next(n for n in function.body if isinstance(n, ast.FunctionDef))
        cases = ast.literal_eval(next(d.args[1] for d in function.decorator_list
                                      if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                                      and d.func.attr == "parametrize"))
        expected_ast = next(n.value.args[0] for n in function.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == "expected" for t in n.targets))
        module_path = args.output / (name + ".py")
        module_path.write_text(imports + "\n@gluon.jit\n" + textwrap.dedent(ast.get_source_segment(source, kernel)) + "\n")
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for i, case in enumerate(cases):
            paired = name.endswith("repeated_m")
            if paired:
                m, k, parent_m, offset = 512, case, 512, 0
                constants = {"K": k}
            else:
                m, k, parent_m, offset = case
                constants = dict(M=m, K=k, PARENT_M=parent_m, OFFSET=offset)
            expected = np.asarray(eval(compile(ast.Expression(expected_ast), "<fixture oracle>", "eval"),
                                       dict(M=m, K=k, offset=offset)), dtype=np.float32)
            # Independently expand scale bytes onto individual dot-product terms.
            values_a = np.repeat(np.arange(1, k // 128 + 1, dtype=np.float32), 128)
            rows = np.arange(offset, offset + m)[:, None]
            scale_columns = np.arange(k // 32)[None, :]
            scale_bytes = (127 + 4 * (rows // 128) + scale_columns // 4).astype(np.uint8)
            scales = np.exp2(scale_bytes.astype(np.int32) - 127).astype(np.float32)
            dense_reference = (values_a * np.repeat(scales, 32, axis=1)).sum(axis=1, dtype=np.float32)
            np.testing.assert_array_equal(expected, dense_reference)
            oracle = {"exact": True, "min": float(expected.min()), "max": float(expected.max())}
            if k > 128:
                permuted = scales.reshape(m, k // 128, 4)[:, ::-1, :].reshape(m, k // 32)
                wrong = (values_a * np.repeat(permuted, 32, axis=1)).sum(axis=1, dtype=np.float32)
                oracle["reversed_word_mismatches"] = int(np.count_nonzero(wrong != expected))
                assert oracle["reversed_word_mismatches"] == m
            for arch in args.architectures:
                compiled = triton.compile(
                    GluonASTSource(module.kernel, signature=dict(out="*fp32", **{v: "constexpr" for v in constants}),
                                   constexprs=constants),
                    target=GPUTarget("cuda", arch, 32), options={"num_warps": 4, "num_ctas": 2 if paired else 1})
                prefix = f"{name}-{i}-sm{arch}"
                for fmt in ("ttgir", "llir", "ptx", "cubin"):
                    data = compiled.asm[fmt]
                    path = args.output / f"{prefix}.{fmt}"
                    path.write_bytes(data) if isinstance(data, bytes) else path.write_text(data)
                mma = [line.strip() for line in compiled.asm["ptx"].splitlines() if "tcgen05.mma." in line]
                assert mma and all("kind::mxf8f6f4" in line for line in mma)
                assert all(f"cta_group::{2 if paired else 1}" in line for line in mma)
                record = dict(name=name, case=case, arch=arch, kernel=compiled.name, oracle=oracle,
                              mma_count=len(mma), metadata=compiled.metadata._asdict(),
                              cubin_sha256=hashlib.sha256(compiled.asm["cubin"]).hexdigest())
                records.append(record)
                print(json.dumps({k: record[k] for k in ("name", "case", "arch", "mma_count", "oracle")}), flush=True)
    provenance = dict(source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      fixture_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      compiler_version=triton.__version__, compiler_path=triton.__file__)
    (args.output / "results.json").write_text(json.dumps(dict(provenance=provenance, cases=records), indent=2, default=str))
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
