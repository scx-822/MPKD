"""Measure inference cost for the unchanged MPKD student architecture.

The script reports parameter count, MACs/FLOPs, peak inference memory, and
CUDA latency for the student checkpoint.  MPKD and CRLD use the same student
architecture; both checkpoints can be passed to verify that their deployment
cost is identical up to measurement noise.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from thop import profile

from mdistiller.models import cifar_model_dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "overhead"


def load_student(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")

    prefixes = ("module.student.", "student.", "module.")
    state = dict(state)
    if not any(key.startswith("conv1.") for key in state):
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in state):
                state = {key[len(prefix) :]: value for key, value in state.items()}
                break

    model = cifar_model_dict["resnet8x4"][0](num_classes=100)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint does not match resnet8x4: "
            f"missing={incompatible.missing_keys[:3]}, "
            f"unexpected={incompatible.unexpected_keys[:3]}"
        )
    return model.eval().to(device)


def measure_model(model, device, batch_size, warmup, repeats):
    example = torch.randn(batch_size, 3, 32, 32, device=device)
    with torch.inference_mode():
        macs, params = profile(model, inputs=(example,), verbose=False)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            for _ in range(warmup):
                model(example)
            torch.cuda.synchronize(device)
            start.record()
            for _ in range(repeats):
                model(example)
            end.record()
            end.synchronize()
        elapsed_ms = start.elapsed_time(end) / repeats
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        with torch.inference_mode():
            for _ in range(warmup):
                model(example)
            start_time = time.perf_counter()
            for _ in range(repeats):
                model(example)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / repeats
        peak_memory_mb = float("nan")

    return {
        "parameters_m": params / 1e6,
        "macs_g_per_image": macs / 1e9 / batch_size,
        "flops_g_per_image": 2.0 * macs / 1e9 / batch_size,
        "batch_size": batch_size,
        "latency_ms_per_batch": elapsed_ms,
        "latency_ms_per_image": elapsed_ms / batch_size,
        "peak_inference_memory_mb": peak_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure MPKD student deployment cost")
    parser.add_argument("--crld-checkpoint", type=Path, required=True)
    parser.add_argument("--mpkd-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.batch_size < 1 or args.warmup < 1 or args.repeats < 1:
        parser.error("batch size, warmup, and repeats must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method, checkpoint in (
        ("CRLD student", args.crld_checkpoint),
        ("MPKD student", args.mpkd_checkpoint),
    ):
        checkpoint = checkpoint.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = load_student(checkpoint, device)
        metrics = measure_model(
            model,
            device=device,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        row = {"method": method, "checkpoint": str(checkpoint), **metrics}
        rows.append(row)
        print(
            f"{method}: params={metrics['parameters_m']:.3f}M, "
            f"FLOPs={metrics['flops_g_per_image']:.4f}G/image, "
            f"latency={metrics['latency_ms_per_image']:.4f} ms/image, "
            f"peak_memory={metrics['peak_inference_memory_mb']:.1f} MB",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = args.output_dir / "student_deployment_overhead.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
