"""Reproducible Lab 6 training pipeline and experiment runner."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.profiler import record_function
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "lab06" / "config.yaml"


@dataclass(frozen=True)
class LabConfig:
    experiment_name: str
    data_dir: str
    results_dir: str
    checkpoint_dir: str
    tracking_dir: str
    seed: int
    model: str
    train_split: int
    validation_split: int
    batch_size: int
    test_batch_size: int
    epochs: int
    num_workers: int
    learning_rate: float
    optimizer: str
    momentum: float
    weight_decay: float
    scheduler: str
    step_size: int
    scheduler_gamma: float
    plateau_patience: int
    accumulation_steps: int
    device: str
    cpu_threads: int
    download: bool


def load_config(path: Path) -> LabConfig:
    yaml = importlib.import_module("yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    expected = {field.name for field in fields(LabConfig)}
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(
            f"Invalid configuration keys; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    config = LabConfig(**raw)
    validate_config(config)
    return config


def validate_config(config: LabConfig) -> None:
    if config.model != "CNN":
        raise ValueError("Lab 6 uses the best Lab 3 model: CNN.")
    if config.train_split <= 0 or config.validation_split <= 0:
        raise ValueError("Training and validation splits must be positive.")
    if config.batch_size <= 0 or config.test_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if config.epochs <= 0 or config.num_workers < 0:
        raise ValueError("Epochs must be positive and num_workers non-negative.")
    if config.accumulation_steps <= 0 or config.cpu_threads <= 0:
        raise ValueError("Accumulation steps and CPU threads must be positive.")
    if config.optimizer not in {"Adam", "SGD"}:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    if config.scheduler not in {
        "StepLR",
        "CosineAnnealingLR",
        "ReduceLROnPlateau",
    }:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {config.device}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def get_device(config: LabConfig) -> torch.device:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


class Lab3CNN(nn.Sequential):
    """The CNN used in Lab 3: 20,490 parameters and 88.93% test accuracy."""

    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )


def make_model() -> nn.Module:
    return Lab3CNN()


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(
    config: LabConfig, num_workers: int | None = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transforms = importlib.import_module("torchvision.transforms")
    datasets = importlib.import_module("torchvision.datasets")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    data_dir = resolve_path(config.data_dir)
    full_train: Dataset = datasets.FashionMNIST(
        data_dir, train=True, download=config.download, transform=transform
    )
    test_data: Dataset = datasets.FashionMNIST(
        data_dir, train=False, download=config.download, transform=transform
    )
    if config.train_split + config.validation_split != len(full_train):
        raise ValueError(
            "Configured train/validation sizes must sum to the Fashion-MNIST "
            f"training-set size ({len(full_train)})."
        )
    train_data, validation_data = random_split(
        full_train,
        [config.train_split, config.validation_split],
        generator=torch.Generator().manual_seed(config.seed),
    )
    workers = config.num_workers if num_workers is None else num_workers
    pin_memory = get_device(config).type == "cuda"
    train_generator = torch.Generator().manual_seed(config.seed)
    worker_generator = torch.Generator().manual_seed(config.seed + 1)
    common = {
        "num_workers": workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=RandomSampler(train_data, generator=train_generator),
        generator=worker_generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.test_batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.test_batch_size,
        shuffle=False,
        **common,
    )
    train_loader.lab06_sampler_generator = train_generator
    train_loader.lab06_worker_generator = worker_generator
    return train_loader, validation_loader, test_loader


def make_optimizer(config: LabConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def make_scheduler(
    config: LabConfig, optimizer: torch.optim.Optimizer
) -> Any:
    if config.scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.scheduler_gamma
        )
    if config.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_gamma,
        patience=config.plateau_patience,
    )


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int = 1,
) -> dict[str, float]:
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive.")
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    iterator = iter(loader)
    batch_count = len(loader)
    if batch_count == 0:
        raise ValueError("Training loader is empty.")
    load_seconds = forward_seconds = backward_seconds = optimizer_seconds = 0.0
    samples = 0
    weighted_loss = 0.0
    batches_seen = 0
    epoch_started = time.perf_counter()

    while batches_seen < batch_count:
        group: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(min(accumulation_steps, batch_count - batches_seen)):
            started = time.perf_counter()
            with record_function("train.data_loading"):
                group.append(next(iterator))
            load_seconds += time.perf_counter() - started
            batches_seen += 1
        group_samples = sum(int(target.size(0)) for _, target in group)
        optimizer.zero_grad(set_to_none=True)
        for inputs, targets in group:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            sync_device(device)
            started = time.perf_counter()
            with record_function("train.forward"):
                logits = model(inputs)
                loss = loss_fn(logits, targets)
            sync_device(device)
            forward_seconds += time.perf_counter() - started

            sync_device(device)
            started = time.perf_counter()
            with record_function("train.backward"):
                (loss * (targets.size(0) / group_samples)).backward()
            sync_device(device)
            backward_seconds += time.perf_counter() - started
            weighted_loss += float(loss.detach()) * targets.size(0)
            samples += targets.size(0)

        sync_device(device)
        started = time.perf_counter()
        with record_function("train.optimizer_step"):
            optimizer.step()
        sync_device(device)
        optimizer_seconds += time.perf_counter() - started

    elapsed = time.perf_counter() - epoch_started
    categorized = load_seconds + forward_seconds + backward_seconds + optimizer_seconds
    return {
        "train_loss": weighted_loss / samples,
        "samples": float(samples),
        "epoch_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "data_loading_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "other_seconds": max(0.0, elapsed - categorized),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    samples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        total_loss += float(loss_fn(logits, targets)) * targets.size(0)
        correct += int((logits.argmax(dim=1) == targets).sum())
        samples += targets.size(0)
    return total_loss / samples, correct / samples


class PeakMemoryMonitor:
    """Sample process RSS while training; CUDA allocation is reported separately."""

    def __init__(self) -> None:
        psutil = importlib.import_module("psutil")
        self._process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._peak = self._process.memory_info().rss
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        try:
            while not self._stop.wait(0.05):
                self._peak = max(self._peak, self._process.memory_info().rss)
        except BaseException as exc:
            self._error = exc

    def __enter__(self) -> PeakMemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Failed to sample process memory.") from self._error
        self._peak = max(self._peak, self._process.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024 * 1024)


def checkpoint_path(config: LabConfig, trial_name: str) -> Path:
    return resolve_path(config.checkpoint_dir) / trial_name / "last.pt"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    history: list[dict[str, float]],
    loader: DataLoader,
    config: LabConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "config": asdict(config),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler_generator": loader.lab06_sampler_generator.get_state(),
        "worker_generator": loader.lab06_worker_generator.get_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loader: DataLoader,
    device: torch.device,
    config: LabConfig,
) -> tuple[int, list[dict[str, float]]]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state["config"] != asdict(config):
        raise ValueError(
            "Checkpoint configuration differs from the active YAML configuration."
        )
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    loader.lab06_sampler_generator.set_state(state["sampler_generator"])
    loader.lab06_worker_generator.set_state(state["worker_generator"])
    return int(state["epoch"]), list(state["history"])


def config_yaml(config: LabConfig) -> str:
    yaml = importlib.import_module("yaml")
    return yaml.safe_dump(asdict(config), sort_keys=False)


@contextmanager
def tracked_run(
    config: LabConfig, name: str, params: dict[str, Any]
) -> Generator[tuple[Any, Any], None, None]:
    mlflow = importlib.import_module("mlflow")
    mlflow.set_tracking_uri(str(resolve_path(config.tracking_dir)))
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(run_name=name) as run:
        safe_params = {
            key: str(value)[:500] for key, value in params.items()
        }
        mlflow.log_params(safe_params)
        yield mlflow, run


def log_epoch(mlflow: Any, row: dict[str, float]) -> None:
    epoch = int(row["epoch"])
    metrics = {
        key: float(value)
        for key, value in row.items()
        if key != "epoch" and math.isfinite(float(value))
    }
    mlflow.log_metrics(metrics, step=epoch)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty result table: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_training(
    config: LabConfig,
    trial_name: str,
    resume: Path | None = None,
    stop_after_epoch: int | None = None,
    test_after: bool = True,
) -> dict[str, Any]:
    validate_config(config)
    set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = get_device(config)
    train_loader, validation_loader, test_loader = build_loaders(config)
    model = make_model().to(device)
    optimizer = make_optimizer(config, model)
    scheduler = make_scheduler(config, optimizer)
    history: list[dict[str, float]] = []
    start_epoch = 0
    if resume is not None:
        start_epoch, history = restore_checkpoint(
            resume, model, optimizer, scheduler, train_loader, device, config
        )
    target_epoch = config.epochs if stop_after_epoch is None else stop_after_epoch
    if target_epoch <= start_epoch or target_epoch > config.epochs:
        raise ValueError(
            f"stop_after_epoch must be in ({start_epoch}, {config.epochs}]."
        )

    run_dir = resolve_path(config.results_dir) / trial_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file = run_dir / "config.yaml"
    config_file.write_text(config_yaml(config), encoding="utf-8")
    params = asdict(config)
    if resume is not None:
        params["resumed_from_epoch"] = start_epoch
    ckpt = checkpoint_path(config, trial_name)

    with tracked_run(config, trial_name, params) as tracking:
        mlflow = tracking[0]
        mlflow.log_artifact(str(config_file))
        for row in history:
            log_epoch(mlflow, row)
        for epoch in range(start_epoch + 1, target_epoch + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with PeakMemoryMonitor() as memory:
                metrics = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    device,
                    accumulation_steps=config.accumulation_steps,
                )
            validation_loss, validation_accuracy = evaluate(
                model, validation_loader, device
            )
            if config.scheduler == "ReduceLROnPlateau":
                scheduler.step(validation_loss)
            else:
                scheduler.step()
            row = {
                "epoch": float(epoch),
                **metrics,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "peak_process_rss_mb": memory.peak_mb,
            }
            if device.type == "cuda":
                row["peak_cuda_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                )
            history.append(row)
            save_checkpoint(
                ckpt,
                model,
                optimizer,
                scheduler,
                epoch,
                history,
                train_loader,
                config,
            )
            log_epoch(mlflow, row)
            print(
                f"{trial_name} epoch {epoch}/{config.epochs}: "
                f"val_acc={validation_accuracy:.4f}, "
                f"throughput={metrics['samples_per_second']:.1f} samples/s"
            )

        history_file = run_dir / "history.json"
        history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(history_file))
        mlflow.log_artifact(str(ckpt))
        result: dict[str, Any] = {
            "history": history,
            "checkpoint": ckpt,
            "history_file": history_file,
            "completed": target_epoch == config.epochs,
        }
        if test_after and target_epoch == config.epochs:
            _, test_accuracy = evaluate(model, test_loader, device)
            mlflow.log_metric("test_accuracy", test_accuracy, step=config.epochs)
            result["test_accuracy"] = test_accuracy
        return result


def run_profile(config: LabConfig) -> Path:
    set_seed(config.seed)
    torch.set_num_threads(config.cpu_threads)
    device = get_device(config)
    train_loader, _, _ = build_loaders(config)
    model = make_model().to(device)
    optimizer = make_optimizer(config, model)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    output = resolve_path(config.results_dir) / "profile"
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "one_epoch_trace.json"
    with tracked_run(config, "one-epoch-profiler", asdict(config)) as tracking:
        mlflow = tracking[0]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as profiler:
            metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                accumulation_steps=config.accumulation_steps,
            )
        profiler.export_chrome_trace(str(trace))
        summary = output / "operator_summary.txt"
        summary.write_text(
            profiler.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=40
            ),
            encoding="utf-8",
        )
        breakdown = output / "epoch_breakdown.json"
        total = metrics["epoch_seconds"]
        breakdown.write_text(
            json.dumps(
                {
                    key: {
                        "seconds": metrics[f"{key}_seconds"],
                        "share_percent": 100 * metrics[f"{key}_seconds"] / total,
                    }
                    for key in [
                        "data_loading",
                        "forward",
                        "backward",
                        "optimizer",
                        "other",
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        mlflow.log_metrics(
            {key: value for key, value in metrics.items() if key != "samples"}
        )
        mlflow.log_artifact(str(trace))
        mlflow.log_artifact(str(summary))
        mlflow.log_artifact(str(breakdown))
    return trace


def save_plot(
    rows: list[dict[str, Any]], output: Path, x_key: str, y_keys: list[str], title: str
) -> None:
    plt = importlib.import_module("matplotlib.pyplot")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for key in y_keys:
        plt.plot([row[x_key] for row in rows], [row[key] for row in rows], marker="o", label=key)
    plt.xlabel(x_key.replace("_", " "))
    plt.ylabel("Seconds per epoch" if y_keys[0].endswith("_seconds") else "Value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def run_worker_study(config: LabConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tracked_run(
        config, "worker-count-study", {**asdict(config), "worker_counts": [0, 1, 2, 4, 8]}
    ) as tracking:
        mlflow = tracking[0]
        for workers in [0, 1, 2, 4, 8]:
            trial = replace(config, num_workers=workers)
            set_seed(trial.seed)
            torch.set_num_threads(trial.cpu_threads)
            device = get_device(trial)
            train_loader, _, _ = build_loaders(trial, num_workers=workers)
            model = make_model().to(device)
            optimizer = make_optimizer(trial, model)
            metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                accumulation_steps=trial.accumulation_steps,
            )
            total = metrics["epoch_seconds"]
            row = {
                "num_workers": workers,
                "epoch_seconds": total,
                "samples_per_second": metrics["samples_per_second"],
                **{
                    f"{key}_share_percent": 100 * metrics[f"{key}_seconds"] / total
                    for key in [
                        "data_loading",
                        "forward",
                        "backward",
                        "optimizer",
                        "other",
                    ]
                },
                **{
                    f"{key}_seconds": metrics[f"{key}_seconds"]
                    for key in [
                        "data_loading",
                        "forward",
                        "backward",
                        "optimizer",
                        "other",
                    ]
                },
            }
            rows.append(row)
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in row.items()
                    if key != "num_workers"
                },
                step=workers,
            )
            print(
                f"workers={workers}: {total:.2f}s, "
                f"{metrics['samples_per_second']:.1f} samples/s"
            )
    output = resolve_path(config.results_dir)
    table = output / "worker_study.csv"
    write_rows(table, rows)
    save_plot(
        rows,
        output / "worker_time_breakdown.png",
        "num_workers",
        ["data_loading_seconds", "forward_seconds", "backward_seconds", "optimizer_seconds"],
        "One-epoch training time by DataLoader worker count",
    )
    return rows


def run_batch_study(config: LabConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_size in [32, 64, 128, 256, 512]:
        trial = replace(config, batch_size=batch_size, epochs=5, accumulation_steps=1)
        result = run_training(trial, f"batch-size-{batch_size}")
        history = result["history"]
        row = {
            "batch_size": batch_size,
            "throughput_samples_per_second": float(
                np.mean([item["samples_per_second"] for item in history])
            ),
            "peak_process_rss_mb": max(
                item["peak_process_rss_mb"] for item in history
            ),
            "final_test_accuracy": result["test_accuracy"],
            "epochs": len(history),
        }
        if "peak_cuda_allocated_mb" in history[-1]:
            row["peak_cuda_allocated_mb"] = max(
                item["peak_cuda_allocated_mb"] for item in history
            )
        rows.append(row)
    output = resolve_path(config.results_dir)
    write_rows(output / "batch_size_study.csv", rows)
    save_plot(
        rows,
        output / "batch_size_throughput.png",
        "batch_size",
        ["throughput_samples_per_second"],
        "Batch size vs training throughput",
    )
    return rows


def run_schedule_study(config: LabConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for optimizer in ["SGD", "Adam"]:
        for scheduler in ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]:
            trial = replace(
                config,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=10,
                accumulation_steps=1,
            )
            name = f"{optimizer.lower()}-{scheduler.lower()}"
            result = run_training(trial, name)
            rows.append(
                {
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "epochs": len(result["history"]),
                    "final_validation_accuracy": result["history"][-1][
                        "validation_accuracy"
                    ],
                    "final_test_accuracy": result["test_accuracy"],
                    "final_learning_rate": result["history"][-1]["learning_rate"],
                    "history_file": str(result["history_file"].relative_to(ROOT)),
                }
            )
    write_rows(resolve_path(config.results_dir) / "schedule_study.csv", rows)
    return rows


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return state["model_state"]


def run_accumulation_study(config: LabConfig) -> dict[str, Any]:
    full_batch = run_training(
        replace(config, batch_size=256, epochs=5, accumulation_steps=1),
        "effective-batch-256",
    )
    accumulated = run_training(
        replace(config, batch_size=64, epochs=5, accumulation_steps=4),
        "batch-64-accumulation-4",
    )
    full_weights = load_weights(full_batch["checkpoint"])
    accumulated_weights = load_weights(accumulated["checkpoint"])
    max_difference = max(
        float((full_weights[key] - accumulated_weights[key]).abs().max())
        for key in full_weights
    )
    result = {
        "effective_batch_test_accuracy": full_batch["test_accuracy"],
        "accumulated_test_accuracy": accumulated["test_accuracy"],
        "absolute_test_accuracy_difference": abs(
            full_batch["test_accuracy"] - accumulated["test_accuracy"]
        ),
        "max_parameter_absolute_difference": max_difference,
        "full_batch_history": str(full_batch["history_file"].relative_to(ROOT)),
        "accumulated_history": str(accumulated["history_file"].relative_to(ROOT)),
    }
    output = resolve_path(config.results_dir) / "accumulation_study.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_resume_demo(config: LabConfig) -> list[dict[str, float]]:
    trial_name = "resume-demo"
    first = run_training(config, trial_name, stop_after_epoch=3, test_after=False)
    resumed = run_training(config, trial_name, resume=first["checkpoint"])
    curve = resumed["history"]
    output = resolve_path(config.results_dir) / "resume_curve.csv"
    write_rows(output, curve)
    return curve


def run_all(config: LabConfig) -> None:
    run_profile(config)
    run_worker_study(config)
    run_batch_study(config)
    run_schedule_study(config)
    run_accumulation_study(config)
    run_resume_demo(config)
    print(f"All studies complete. Evidence saved in {resolve_path(config.results_dir)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train the configured model.")
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Stop after saving this epoch, for an interruption/resume exercise.",
    )
    subparsers.add_parser("profile", help="Profile one epoch including optimizer.step().")
    subparsers.add_parser("workers", help="Run the [0,1,2,4,8] worker study.")
    subparsers.add_parser("batch-sizes", help="Run the five-epoch batch-size study.")
    subparsers.add_parser("schedules", help="Compare optimizers and schedulers for 10 epochs.")
    subparsers.add_parser("accumulation", help="Compare batch 256 with 64 x accumulation 4.")
    subparsers.add_parser("resume-demo", help="Exercise save-at-epoch-3 and checkpoint resume.")
    subparsers.add_parser("all", help="Run all experiments; this is compute-intensive.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "train":
        run_training(
            config,
            "train",
            resume=args.resume,
            stop_after_epoch=args.stop_after_epoch,
        )
    elif args.command == "profile":
        print(f"Profiler trace: {run_profile(config)}")
    elif args.command == "workers":
        run_worker_study(config)
    elif args.command == "batch-sizes":
        run_batch_study(config)
    elif args.command == "schedules":
        run_schedule_study(config)
    elif args.command == "accumulation":
        print(json.dumps(run_accumulation_study(config), indent=2))
    elif args.command == "resume-demo":
        run_resume_demo(config)
    else:
        run_all(config)


if __name__ == "__main__":
    main()
