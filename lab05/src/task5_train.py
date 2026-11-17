import random, time, statistics, yaml
from pathlib import Path
import torch
from sklearn.datasets import make_moons
from engine import Value

cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))
SEED, c5 = cfg["seed"], cfg["task5"]
EPOCHS, LR = c5["epochs"], c5["lr"]

X, y = make_moons(n_samples=c5["n_samples"], noise=c5["noise"], random_state=SEED)
XS = X.tolist()
YS = [float(v) for v in y]


def train_mine(seed):
    rng = random.Random(seed)
    w1 = [[Value(rng.uniform(-1, 1)) for _ in range(2)] for _ in range(8)]
    b1 = [Value(0.0) for _ in range(8)]
    w2 = [Value(rng.uniform(-1, 1)) for _ in range(8)]
    b2 = Value(0.0)
    params = [p for r in w1 for p in r] + b1 + w2 + [b2]
    loss = None
    for _ in range(EPOCHS):
        loss = Value(0.0)
        for xi, yi in zip(XS, YS):
            h = [(w1[j][0] * xi[0] + w1[j][1] * xi[1] + b1[j]).relu() for j in range(8)]
            z = b2
            for j in range(8):
                z = z + w2[j] * h[j]
            p = z.sigmoid()
            loss = loss + (-(yi * (p + 1e-12).log() + (1 - yi) * (1 - p + 1e-12).log()))
        loss = loss * (1.0 / len(XS))
        for q in params:
            q.grad = 0.0
        loss.backward()
        for q in params:
            q.data -= LR * q.grad
    return loss.data


def train_torch(seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(),
                                torch.nn.Linear(8, 1), torch.nn.Sigmoid())
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    lf = torch.nn.BCELoss()
    loss = None
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = lf(model(Xt), yt)
        loss.backward()
        opt.step()
    return loss.item()


def bench(fn):
    times, losses = [], []
    for s in range(c5["runs"]):
        t = time.perf_counter()
        l = fn(SEED + s)
        times.append(time.perf_counter() - t)
        losses.append(l)
    return statistics.median(times), statistics.median(losses)


tm, lm = bench(train_mine)
tt, lt = bench(train_torch)
print(f"mine : median {tm:.3f}s, final loss {lm:.4f}")
print(f"torch: median {tt:.3f}s, final loss {lt:.4f}")
print(f"torch faster by {tm / tt:.1f}x")