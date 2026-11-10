import time, statistics, yaml
from pathlib import Path
import torch

cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
c6 = cfg["task6"]
torch.manual_seed(cfg["seed"])

model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(),
                            torch.nn.Linear(128, 10)).eval()
x = torch.randn(c6["batch"], 64)
y = model(x).sum()

def walk(fn, d=0):
    if fn is None:
        return
    print("  " * d + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, d + 1)

print("--- autograd graph")
walk(y.grad_fn)
print("x     shape:", tuple(x.shape), "dtype:", x.dtype,
      "contiguous:", x.is_contiguous(), "stride:", x.stride())
xt = x.t()
print("x.t() shape:", tuple(xt.shape), "dtype:", xt.dtype,
      "contiguous:", xt.is_contiguous(), "stride:", xt.stride())

def bench(fn, n=c6["iterations"], warm=c6["warmup"]):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6   # мкс

with torch.no_grad():
    t_eager = bench(lambda: model(x))
    traced = torch.jit.trace(model, x)
    print("--- traced graph")
    print(traced.graph)
    t_trace = bench(lambda: traced(x))
    print(f"eager: {t_eager:.1f} us | jit.trace: {t_trace:.1f} us")
    try:
        compiled = torch.compile(model)
        t_comp = bench(lambda: compiled(x))
        print(f"torch.compile: {t_comp:.1f} us")
    except Exception as e:
        print("torch.compile not supported here:", type(e).__name__)

big = torch.randn(c6["big_size"], c6["big_size"])
bt = big.t()                              # non-contiguous view
print("big contiguous:", big.is_contiguous(), "| big.t() contiguous:", bt.is_contiguous())
t_c = bench(lambda: big * 2, n=30, warm=5)
t_n = bench(lambda: bt * 2, n=30, warm=5)
t_f = bench(lambda: bt.contiguous() * 2, n=30, warm=5)
print(f"contiguous: {t_c:.0f} us | non-contiguous: {t_n:.0f} us | "
      f".contiguous()+op: {t_f:.0f} us")