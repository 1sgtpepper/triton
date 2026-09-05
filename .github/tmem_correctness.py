"""Small, exact-output TMEM correctness investigations for SM100.

Compile only (no GPU):
    python .github/tmem_correctness.py --compile-only --output-dir evidence

Execute on a B200:
    python .github/tmem_correctness.py --output-dir evidence

The exit status is nonzero if any case fails. A baseline failure is evidence to
investigate, not a successful correctness test. No timing measurements are made.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import traceback

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon._runtime import GluonASTSource
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier


@gluon.jit
def scale_view_kernel(out, scale_out, M: gl.constexpr, K: gl.constexpr,
                      PARENT_M: gl.constexpr, COMPACT: gl.constexpr,
                      USE_ACC: gl.constexpr):
    N: gl.constexpr = 128
    reg: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    a = gl.full([M, K], 1, gl.float8e5, reg)
    b = gl.full([K, N], 1, gl.float8e5, reg)
    smem_a = gl.allocate_shared_memory(
        gl.float8e5, [M, K], gl.NVMMASharedLayout(128, 8, transposed=False), a)
    smem_b = gl.allocate_shared_memory(
        gl.float8e5, [K, N], gl.NVMMASharedLayout(128, 8, transposed=True), b)

    parent = allocate_tensor_memory(gl.uint8, [PARENT_M, K // 32], TensorMemoryScalesLayout())
    preg: gl.constexpr = parent.get_reg_layout()
    rows = gl.arange(0, PARENT_M, gl.SliceLayout(1, preg))[:, None]
    scale_values = (127 + rows // 128 + gl.zeros([PARENT_M, K // 32], gl.int32, preg)).to(gl.uint8)
    parent.store(scale_values)
    view = parent.slice(0, M, dim=0)
    if COMPACT:
        scale_a = allocate_tensor_memory(gl.uint8, [M, K // 32], TensorMemoryScalesLayout())
        scale_a.store(view.load(scale_a.get_reg_layout()))
    else:
        scale_a = view
    scale_b = allocate_tensor_memory(gl.uint8, [N, K // 32], TensorMemoryScalesLayout())
    scale_b.store(gl.full([N, K // 32], 127, gl.uint8, scale_b.get_reg_layout()))

    acc = allocate_tensor_memory(gl.float32, [M, N], TensorMemoryLayout([128, 128], 1))
    if USE_ACC:
        acc.store(gl.full([M, N], 3, gl.float32, acc.get_reg_layout()))
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma_scaled(smem_a, smem_b, acc, scale_a, scale_b,
                      "e5m2", "e5m2", use_acc=USE_ACC, mbarriers=[bar])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    result = acc.load()
    result_reg: gl.constexpr = result.type.layout
    rm = gl.arange(0, M, gl.SliceLayout(1, result_reg))[:, None]
    rn = gl.arange(0, N, gl.SliceLayout(0, result_reg))[None, :]
    gl.store(out + rm * N + rn, result)

    # Independently observe the descriptor contents, including the logical slice.
    scales = view.load()
    sreg: gl.constexpr = scales.type.layout
    sr = gl.arange(0, M, gl.SliceLayout(1, sreg))[:, None]
    sk = gl.arange(0, K // 32, gl.SliceLayout(0, sreg))[None, :]
    gl.store(scale_out + sr * (K // 32) + sk, scales)


@gluon.jit
def alias_lifetime_kernel(selector, out, temp_out, MODE: gl.constexpr,
                          TEMP_COLS: gl.constexpr, EARLY_TEMP: gl.constexpr):
    layout: gl.constexpr = TensorMemoryLayout([128, 32], 1)
    first = allocate_tensor_memory(gl.float32, [128, 32], layout)
    second = allocate_tensor_memory(gl.float32, [128, 32], layout)
    first.store(gl.full([128, 32], 3, gl.float32, first.get_reg_layout()))
    second.store(gl.full([128, 32], 5, gl.float32, second.get_reg_layout()))
    if TEMP_COLS > 0 and EARLY_TEMP:
        temp = allocate_tensor_memory(gl.float32, [128, TEMP_COLS], layout)

    # This load is a uniform scalar. The condition is not a compile-time argument.
    choice = gl.load(selector)
    if MODE == "select":
        if choice != 0:
            selected = first
        else:
            selected = second
    elif MODE == "first":
        selected = first
    else:
        selected = second

    if TEMP_COLS > 0:
        if not EARLY_TEMP:
            temp = allocate_tensor_memory(gl.float32, [128, TEMP_COLS], layout)
        temp.store(gl.full([128, TEMP_COLS], 7, gl.float32, temp.get_reg_layout()))

    values = selected.load()
    reg: gl.constexpr = values.type.layout
    r = gl.arange(0, 128, gl.SliceLayout(1, reg))[:, None]
    c = gl.arange(0, 32, gl.SliceLayout(0, reg))[None, :]
    gl.store(out + r * 32 + c, values)
    if TEMP_COLS > 0:
        temporary = temp.load()
        treg: gl.constexpr = temporary.type.layout
        tr = gl.arange(0, 128, gl.SliceLayout(1, treg))[:, None]
        tc = gl.arange(0, TEMP_COLS, gl.SliceLayout(0, treg))[None, :]
        gl.store(temp_out + tr * TEMP_COLS + tc, temporary)


SCALE_CASES = [
    ("view_two_m_two_k", 256, 256, 512, False, False),
    ("compact_parent", 256, 256, 256, False, False),
    ("one_m_tile", 128, 256, 512, False, False),
    ("one_k_word", 256, 128, 512, False, False),
    ("four_k_words", 256, 512, 512, False, False),
    ("copy_to_compact", 256, 256, 512, True, False),
    ("view_accumulate", 256, 256, 512, False, True),
]
ALIAS_CASES = [
    ("select_wide_temp", "select", 64, False),
    ("select_narrow_temp", "select", 32, False),
    ("select_no_temp", "select", 0, False),
    ("select_early_temp", "select", 64, True),
    ("direct_first", "first", 64, False),
    ("direct_second", "second", 64, False),
]


def save_compiled(compiled, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("ttgir", "llir", "ptx", "cubin"):
        data = compiled.asm.get(name)
        if isinstance(data, str):
            (directory / name).write_text(data)
        elif isinstance(data, bytes):
            (directory / name).write_bytes(data)
    (directory / "metadata.json").write_text(json.dumps(compiled.metadata._asdict(), default=str, indent=2))


def compile_scale(case, directory):
    name, m, k, parent_m, compact, use_acc = case
    constants = dict(M=m, K=k, PARENT_M=parent_m, COMPACT=compact, USE_ACC=use_acc)
    signature = dict(out="*fp32", scale_out="*u8", **{key: "constexpr" for key in constants})
    compiled = triton.compile(
        GluonASTSource(scale_view_kernel, signature=signature, constexprs=constants),
        target=GPUTarget("cuda", 100, 32), options={"num_warps": 4})
    save_compiled(compiled, directory / name)
    return {"case": name, "compiled": True}


def compile_alias(case, directory):
    name, mode, columns, early = case
    constants = dict(MODE=mode, TEMP_COLS=columns, EARLY_TEMP=early)
    signature = dict(selector="*i32", out="*fp32", temp_out="*fp32",
                     **{key: "constexpr" for key in constants})
    compiled = triton.compile(
        GluonASTSource(alias_lifetime_kernel, signature=signature, constexprs=constants),
        target=GPUTarget("cuda", 100, 32), options={"num_warps": 4})
    save_compiled(compiled, directory / name)
    return {"case": name, "compiled": True}


def execute_scale(case, directory, torch):
    name, m, k, parent_m, compact, use_acc = case
    out = torch.full((m, 128), float("nan"), device="cuda", dtype=torch.float32)
    scales = torch.full((m, k // 32), 255, device="cuda", dtype=torch.uint8)
    compiled = scale_view_kernel[(1,)](out, scales, m, k, parent_m, compact, use_acc, num_warps=4)
    torch.cuda.synchronize()
    save_compiled(compiled, directory / name)
    actual = out.cpu()
    # Powers of two times integer K are exact; no GEMM or tolerance is involved.
    expected_rows = torch.tensor([k * (2 ** (row // 128)) + (3 if use_acc else 0)
                                  for row in range(m)], dtype=torch.float32)
    expected = expected_rows[:, None].expand(m, 128)
    expected_scales = torch.tensor([127 + row // 128 for row in range(m)],
                                  dtype=torch.uint8)[:, None].expand(m, k // 32)
    record = {
        "case": name, "correct": torch.equal(actual, expected),
        "scale_view_correct": torch.equal(scales.cpu(), expected_scales),
        "mismatches": int((actual != expected).sum()),
        "row_samples": [{"row": row, "actual": actual[row, 0].item(),
                         "expected": expected[row, 0].item()} for row in range(0, m, 128)],
    }
    torch.save({"actual": actual, "expected": expected, "scales": scales.cpu()},
               directory / name / "results.pt")
    return record


def execute_alias(case, directory, torch):
    name, mode, columns, early = case
    records = []
    for choice in (0, 1):
        selector = torch.tensor([choice], device="cuda", dtype=torch.int32)
        out = torch.full((128, 32), float("nan"), device="cuda")
        temporary = torch.full((128, max(1, columns)), float("nan"), device="cuda")
        compiled = alias_lifetime_kernel[(1,)](selector, out, temporary, mode, columns, early, num_warps=4)
        torch.cuda.synchronize()
        save_compiled(compiled, directory / name)
        expected_value = 3 if mode == "first" or (mode == "select" and choice) else 5
        actual = out.cpu()
        correct = bool(torch.all(actual == expected_value))
        temp_correct = not columns or bool(torch.all(temporary.cpu() == 7))
        records.append({"choice": choice, "correct": correct, "temporary_correct": temp_correct,
                        "expected": expected_value, "actual_values": torch.unique(actual).tolist()})
        torch.save({"actual": actual, "temporary": temporary.cpu()},
                   directory / name / f"results-{choice}.pt")
    return {"case": name, "correct": all(r["correct"] and r["temporary_correct"] for r in records),
            "outcomes": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", choices=("all", "scale", "alias"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("tmem-evidence"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch = None
    environment = {"triton": importlib.metadata.version("triton"),
                   "gpu_execution": not args.compile_only}
    if not args.compile_only:
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
            raise RuntimeError("Execution requires an SM100 GPU (for example B200).")
        environment.update(torch=torch.__version__, cuda=torch.version.cuda,
                           gpu=torch.cuda.get_device_name(),
                           capability=torch.cuda.get_device_capability())
    records = []
    families = []
    if args.case in ("all", "scale"):
        families.append((SCALE_CASES, compile_scale, execute_scale))
    if args.case in ("all", "alias"):
        families.append((ALIAS_CASES, compile_alias, execute_alias))
    for cases, compile_fn, execute_fn in families:
        for case in cases:
            try:
                record = (compile_fn(case, args.output_dir) if args.compile_only
                          else execute_fn(case, args.output_dir, torch))
            except Exception:
                record = {"case": case[0], "error": traceback.format_exc()}
            records.append(record)
            print(json.dumps(record), flush=True)
    result = {"environment": environment, "cases": records}
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    return int(any("error" in r or not r.get("correct", True) or not r.get("scale_view_correct", True)
                   for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
