# Lab 6: Training Pipelines and Project Proposal

## Implementation and reproducibility

This lab reuses the Lab 3 CNN (20,490 parameters; reported test accuracy 88.93%)
on Fashion-MNIST. The full run configuration is in [`config.yaml`](config.yaml);
the runner validates it as one YAML mapping and records the configuration, seed,
metrics, and artifacts in local MLflow tracking. Each completed epoch writes an
atomic checkpoint containing model, optimizer, scheduler, history, RNG states,
and DataLoader generator state. The optimizer is SGD with momentum or Adam; the
scheduler is StepLR, CosineAnnealingLR, or ReduceLROnPlateau.

From the repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python lab06\src\run_lab06.py --config lab06\config.yaml train
```

The dataset is downloaded under `data/` if it is not already available.
`device: auto` selects CUDA when available and otherwise CPU. Runs, checkpoints,
tables, and plots are written under `mlruns/`, `lab06/checkpoints/`, and
`lab06/results/`; these generated files are intentionally not committed.

## Experiments

Run each requested study separately, or run `all` (compute-intensive):

```powershell
python lab06\src\run_lab06.py --config lab06\config.yaml profile
python lab06\src\run_lab06.py --config lab06\config.yaml workers
python lab06\src\run_lab06.py --config lab06\config.yaml batch-sizes
python lab06\src\run_lab06.py --config lab06\config.yaml schedules
python lab06\src\run_lab06.py --config lab06\config.yaml accumulation
python lab06\src\run_lab06.py --config lab06\config.yaml resume-demo
```

The profiler captures one complete training epoch, including labeled forward,
backward, and optimizer-step regions, and exports a Chrome trace and operator
summary. Worker-study data-loading time is the time spent waiting for batches;
the remaining category is reported as `other`. Percentages use total epoch wall
time as the denominator, so worker prefetching can reduce measured consumer wait
without changing the work done by the data pipeline.

The batch-size table reports mean training throughput across five epochs, peak
sampled process RSS, CUDA peak allocation when applicable, and official test
accuracy after epoch five. CPU RSS is sampled every 50 ms and is approximate.
The schedule study runs all six optimizer/scheduler combinations for ten epochs.
The accumulation study compares batch 256 against batch 64 with accumulation 4,
using the same seed and sample order.

To exercise a real interruption from the command line, stop after epoch 3 and
then start a second process to resume:

```powershell
python lab06\src\run_lab06.py --config lab06\config.yaml train --stop-after-epoch 3
python lab06\src\run_lab06.py --config lab06\config.yaml train --resume lab06\checkpoints\train\last.pt
```

The resume command restores optimizer/scheduler state, RNG state, and the
DataLoader generator. The `resume-demo` command performs the same checkpoint
reload in one invocation for convenience. Resumed metrics are logged in a new
MLflow run while retaining their original epoch numbers in the checkpoint
history.

## Results (populate from generated evidence)

No experiment values are pre-filled: run the commands above on the target
machine and copy the generated results into these tables.

### One-epoch profile

| Region | Time (s) | Share of epoch (%) |
|---|---:|---:|
| Data-loading wait | — | — |
| Forward pass | — | — |
| Backward pass | — | — |
| Optimizer step | — | — |
| Other | — | — |

Trace: `results/profile/one_epoch_trace.json`  
Operator summary: `results/profile/operator_summary.txt`  
Measured region breakdown: `results/profile/epoch_breakdown.json`

### DataLoader worker study

Fill from `results/worker_study.csv`; plot: `results/worker_time_breakdown.png`.

| Workers | Data wait (%) | Forward (%) | Backward (%) | Optimizer (%) | Throughput (samples/s) | Bottleneck |
|---:|---:|---:|---:|---:|---:|---|
| 0 | — | — | — | — | — | — |
| 1 | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — |
| 8 | — | — | — | — | — | — |

### Batch-size study

Fill from `results/batch_size_study.csv`; plot:
`results/batch_size_throughput.png`.

| Batch size | Throughput (samples/s) | Peak process RSS (MB) | Peak CUDA allocated (MB) | Test accuracy |
|---:|---:|---:|---:|---:|
| 32 | — | — | — | — |
| 64 | — | — | — | — |
| 128 | — | — | — | — |
| 256 | — | — | — | — |
| 512 | — | — | — | — |

### Optimizer and scheduler comparison

Fill from `results/schedule_study.csv`.

| Optimizer | Scheduler | Epochs | Final validation accuracy | Test accuracy | Final learning rate |
|---|---|---:|---:|---:|---:|
| SGD + momentum | StepLR | 10 | — | — | — |
| SGD + momentum | CosineAnnealingLR | 10 | — | — | — |
| SGD + momentum | ReduceLROnPlateau | 10 | — | — | — |
| Adam | StepLR | 10 | — | — | — |
| Adam | CosineAnnealingLR | 10 | — | — | — |
| Adam | ReduceLROnPlateau | 10 | — | — | — |

### Gradient accumulation and resume

Use `results/accumulation_study.json` to compare effective batch size and
parameter differences. Use `results/resume_curve.csv` to verify epochs 1–3 are
preserved and training continues from epoch 4 through epoch 10.

| Check | Result |
|---|---|
| Batch 256 vs. batch 64 / accumulation 4 | Pending run |
| Test-accuracy difference | Pending run |
| Maximum parameter difference | Pending run |
| Resume curve continuous from epoch 3 | Pending run |

## Project proposal

The one-page proposal draft is [`proposal.md`](proposal.md). It is not approved
and contains student name/ID placeholders; personalize it and obtain instructor
approval before submitting.

## Verification

Run the focused unit tests with:

```powershell
python -m pytest lab06\tests\test_run_lab06.py
```
