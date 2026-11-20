"""End-to-end CPU mock run of the notebook. Usage: python tests/run_mock.py [notebook.ipynb]
Extracts the notebook's python cells (skips the install cell 1, uses its run_stage), writes cell 1b's module to the work dir,
and executes cells in order with MOCK=1. Paths /kaggle/working -> tests/work."""
import json, os, re, shutil, sys, types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NB = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "gr00th_stress.ipynb"
WORK = HERE / "work"
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
os.environ.update(MOCK="1", MOCK_DIR=str(HERE), MOCK_REPO=str(WORK / "fakerepo"), STRESS_WORK=str(WORK), MOCK_N_EP="6", MOCK_T_PER_EP="3", MOCK_DIAG_FRAMES="8")
sys.path.insert(0, str(HERE))
try:
    import IPython  # noqa
except ImportError:  # minimal stub so `from IPython.display import Image, display` works headless
    m = types.ModuleType("IPython"); d = types.ModuleType("IPython.display")
    d.Image = lambda filename=None, **k: ("Image", filename); d.display = lambda *a, **k: print("[display]", *a)
    sys.modules.update({"IPython": m, "IPython.display": d}); m.display = d

import mock_gr00t
mock_gr00t.build_fake_repo(WORK / "fakerepo")

nb = json.load(open(NB))
cells = ["".join(c["source"]) for c in nb["cells"]]
code = [(i, s) for i, s in enumerate(cells) if nb["cells"][i]["cell_type"] == "code"]
fix = lambda s: s.replace("/kaggle/working", str(WORK)).replace("/kaggle/tmp", str(WORK / "tmp"))
def strip_hooks(src):
    return re.sub(r"^[ \t]*# <mock>\n.*?^[ \t]*# </mock>\n", "", src, flags=re.S | re.M)

ns = {"__name__": "__main__"}
exec("import os, sys, re, json, shutil, subprocess, pathlib, logging\nlogging.basicConfig(level=logging.INFO)\nlog = logging.getLogger('t')\nWORK = pathlib.Path(%r); OUT = WORK / 'outputs'; OUT.mkdir(parents=True, exist_ok=True)\n"
     "GPY = sys.executable\n" % str(WORK), ns)
c1 = cells[code[0][0]]
exec(c1[c1.index("def run_stage"):], ns)          # real run_stage from the notebook
ok = True
for i, s in code[1:]:
    if s.startswith("%%writefile"):
        body = s.split("\n", 1)[1]
        full = (ROOT / "gr00th_stress.py").read_text()
        (WORK / "gr00th_stress.py").write_text(full)   # the mock run uses the module including test hooks
        same = strip_hooks(full).rstrip() == body.rstrip()
        print(f"[cell {i}] notebook module equals repo module minus test hooks: {same}")
        ok &= same
        continue
    print(f"\n######## cell {i} ########\n" + s.split("\n")[0][:100], flush=True)
    exec(compile(fix(s), f"<cell {i}>", "exec"), ns)
print("\nALL CELLS OK" if ok else "\nCELLS RAN, MIRROR MISMATCH")
