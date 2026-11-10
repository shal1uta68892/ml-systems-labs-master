import torch, yaml
from pathlib import Path
from engine import Value

cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
c4 = cfg["task4"]
torch.manual_seed(cfg["seed"])
N = c4["n_elements"]
a0 = torch.randn(N, dtype=torch.float64).tolist()
b0 = torch.randn(N, dtype=torch.float64).tolist()
c0 = torch.randn(N, dtype=torch.float64).tolist()

def f_value(a, b, c):
    terms = [((x * y + z) ** 2).relu() for x, y, z in zip(a, b, c)]
    total = terms[0]
    for t in terms[1:]:
        total = total + t
    return total

a = [Value(x) for x in a0]
b = [Value(x) for x in b0]
c = [Value(x) for x in c0]
out = f_value(a, b, c)
out.backward()
mine = [v.grad for v in a + b + c]

ta = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
tb = torch.tensor(b0, dtype=torch.float64, requires_grad=True)
tc = torch.tensor(c0, dtype=torch.float64, requires_grad=True)
ref_out = ((ta * tb + tc) ** 2).relu().sum()
ref_out.backward()
ref = torch.cat([ta.grad, tb.grad, tc.grad]).tolist()

def f_num(vals):
    n = N
    return f_value([Value(x) for x in vals[:n]],
                   [Value(x) for x in vals[n:2 * n]],
                   [Value(x) for x in vals[2 * n:]]).data

base = a0 + b0 + c0
eps = c4["eps"]
num = []
for i in range(len(base)):
    p = base.copy()
    m = base.copy()
    p[i] += eps
    m[i] -= eps
    num.append((f_num(p) - f_num(m)) / (2 * eps))
d1 = max(abs(x - y) for x, y in zip(mine, ref))
d2 = max(abs(x - y) for x, y in zip(mine, num))
tol = c4["tolerance"]

print("forward mine :", out.data)
print("forward torch:", ref_out.item())
print("max |mine - torch|     =", d1)
print("max |mine - numerical| =", d2)
print("PASS" if d1 < tol and d2 < tol else "FAIL", f"(tol={tol})")