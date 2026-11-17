import subprocess, sys, yaml
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
cfg = yaml.safe_load(open(ROOT / "config.yaml"))
RES = ROOT / cfg["paths"]["results"]
RES.mkdir(exist_ok=True)

for name in ["task4_verify", "task5_train", "task6_compile"]:
    r = subprocess.run([sys.executable, str(SRC / f"{name}.py")],
                       capture_output=True, text=True, cwd=SRC)
    out = r.stdout + r.stderr
    (RES / f"{name}.txt").write_text(out, encoding="utf-8")
    print(f"===== {name} =====\n{out}")