# Lab 5 Report: Frameworks and Automatic Differentiation

## 1. My autograd engine

I made a class `Value` (file `src/engine.py`). It is one number in a computation graph. Each `Value` keeps:
- `data` - the number itself,
- `grad` - its gradient (starts at 0),
- `_parents` - the nodes it was made from,
- `_backward` - a small function with the local derivative rule.

I implemented `+`, `*`, `**` and `relu`. I also added `sigmoid` and `log`, because I need them for training with BCE loss. Every operation makes a new `Value`, so the graph is built by itself during the forward pass.

How `backward()` works:
1. Sort the graph topologically, so every node comes after its parents.
2. Set the gradient of the output to 1.
3. Go through the nodes in reverse order and call `_backward()` for each.
4. Use `+=` for gradients and never `=`. If one node is used in several places, its gradients must be added together.

## 2. Checking that gradients are correct

Function: `((a*b + c)**2).relu().sum()`. I used 6 random numbers for each of a, b, c (seed 42, float64).

| Check | Result |
|---|---|
| forward value, my engine | 9.356174211694645 |
| forward value, PyTorch | 9.356174211694645 |
| max difference, mine vs PyTorch autograd | 0.0 |
| max difference, mine vs numerical gradient | 8.98e-11 |
| required limit | < 1e-6 |

Result: **PASS**. My gradients are the same as PyTorch and very close to the numerical estimate.

## 3. Training a small model on make_moons

Model: 2 -> 8 -> 1, ReLU in the hidden layer, sigmoid at the end, BCE loss. 100 samples, noise 0.1, learning rate 0.5, 100 epochs. I ran each version 5 times and took the median.

| Version | Time | Final loss |
|---|---|---|
| My engine | 6.264 s | 0.2492 |
| PyTorch | 0.110 s | 0.2660 |

PyTorch is about **57 times faster**. The losses are close, so both train the model fine. They are not exactly equal because the starting weights are different.

Why is my engine so slow? It works with single numbers in plain Python. Every number is a separate object with its own function, so one epoch makes thousands of objects and calls. PyTorch works with whole tensors in fast C++ code and handles all data in a few big operations.

## 4. Looking at the graph and tensors

Model: `Linear(64,128) -> ReLU -> Linear(128,10)`, input size (256, 64).

Autograd graph:
```
SumBackward0
  AddmmBackward0
    AccumulateGrad
    ReluBackward0
      AddmmBackward0
        AccumulateGrad
        TBackward0 -> AccumulateGrad
    TBackward0 -> AccumulateGrad
```
Each Linear layer is `AddmmBackward0` (matrix multiply + bias). `TBackward0` is the transpose of the weights. `AccumulateGrad` is where gradients of the parameters are stored.

| Tensor | shape | dtype | contiguous | stride |
|---|---|---|---|---|
| `x` | (256, 64) | float32 | True | (64, 1) |
| `x.t()` | (64, 256) | float32 | False | (1, 64) |

`x.t()` does not copy data. It is the same memory with different strides, so it is not contiguous.

## 5. Eager vs traced vs compiled

I did 20 warm-up runs and then took the median of 200 runs (CPU, no gradients).

| Mode | Time, us |
|---|---|
| eager | 295.2 |
| `torch.jit.trace` | 310.0 |
| `torch.compile` | 348.8 |

Eager was the fastest here.

## 6. Non-contiguous tensor

Tensor 4096 x 4096, operation `* 2`, median of 30 runs.

| Operation | Time, us |
|---|---|
| contiguous `big * 2` | 5790 |
| non-contiguous `big.t() * 2` | 5890 |
| `big.t().contiguous() * 2` | 76490 |

The non-contiguous version is only a bit slower (about 2%), because `* 2` is a simple element-by-element operation. The `.contiguous()` version is much slower because it copies all 16.7 million numbers every time. So copying is worth it only if we do many operations on the same tensor after it.

## 7. When does compilation help?

In my test compilation did not help. Eager (295 us) was a bit faster than `jit.trace` (310 us) and `torch.compile` (349 us). My model is very small: just two Linear layers and one ReLU. There is almost nothing to speed up or fuse, and the extra checks of compiled modes cost more than they save.

Compilation helps when:
- the model has many small operations that can be fused into fewer ones,
- the same model runs many times, so the compile time is paid back,
- we use a GPU, where fewer kernel launches and less memory traffic matter.

For small models and short runs, eager mode is the better choice.

## 8. How to run

From the main folder, with the virtual environment on:
```
pip install -r requirements.txt
python lab05/src/run_all.py
```
Results are saved in `lab05/results/`. All settings are in `lab05/config.yaml` (seed 42).