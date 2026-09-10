import ast, hashlib, importlib.util, json, pathlib, sys, textwrap
import triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
variant = sys.argv[1]
source = pathlib.Path("python/test/gluon/test_core.py").read_text()
function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "test_tcgen05_mma_scaled_sliced_a_scales")
kernel = next(n for n in function.body if isinstance(n, ast.FunctionDef))
cases = ast.literal_eval(next(d.args[1] for d in function.decorator_list if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "parametrize"))
expected = next(n.value.args[0] for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "expected" for t in n.targets))
imports = """from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout, TensorMemoryScalesLayout, allocate_tensor_memory, tcgen05_mma_scaled
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
"""
text = imports + "\n@gluon.jit\n" + textwrap.dedent(ast.get_source_segment(source, kernel)) + "\n"
root = pathlib.Path("evidence");root.mkdir(exist_ok=True)
(root / "fixture.sha256").write_text(hashlib.sha256(source.encode()).hexdigest())
records=[]
for mode in ([variant,"permuted"] if variant == "fixed" else [variant]):
    body = text if mode != "permuted" else text.replace("compact_cols // 4", "(K // 128 - 1 - compact_cols // 4)").replace(" + cols // 4", " + (K // 128 - 1 - cols // 4)")
    module_path = pathlib.Path("/tmp/scale_"+mode+".py");module_path.write_text(body)
    spec=importlib.util.spec_from_file_location("scale_"+mode,module_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for i, case in enumerate(cases):
        if mode=="permuted" and i!=4:continue
        m,k,parent_m,compact,use_acc,offset=case
        constants=dict(M=m,K=k,PARENT_M=parent_m,COMPACT=compact,USE_ACC=use_acc,OFFSET=offset)
        compiled=triton.compile(GluonASTSource(module.kernel,signature=dict(out="*fp32",**{k:"constexpr" for k in constants}),constexprs=constants),target=GPUTarget("cuda",100,32),options={"num_warps":4})
        name=mode+"-"+str(i);cubin=compiled.asm["cubin"];(root/(name+".cubin")).write_bytes(cubin);(root/(name+".ptx")).write_text(compiled.asm["ptx"])
        oracle=eval(compile(ast.Expression(expected),"<test oracle>","eval"),dict(M=m,K=k,offset=offset,use_acc=use_acc))
        records.append(dict(name=name,case=case,kernel=compiled.name,metadata=compiled.metadata._asdict(),sha256=hashlib.sha256(cubin).hexdigest(),expected=oracle))
        print(name,compiled.name,compiled.metadata.shared,flush=True)
(root/(variant+".json")).write_text(json.dumps(records,indent=2,default=str))
print("compiler",triton.__version__,triton.__file__)
