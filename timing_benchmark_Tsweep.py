#!/usr/bin/env python3
"""
Timing benchmark across T_sweep subfolders.

Measures GPU batch sampling, GPU single-event sampling, and CPU single-event
sampling (1 event only — CPU is slow) for each T_* subdirectory.

Usage:
    python timing_benchmark_Tsweep.py \
        --sweep_dir T_sweep_cosine_charge \
        --n_events 128 \
        --batch_size 32 \
        --n_warmup 3 \
        --plot_out timing_Tsweep.png
"""

import argparse
import time
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from particle_diffusion_cosine_charge import load_meta_and_model, sample_batch, sample_single


# ============================================================
# GPU MONITORING
# ============================================================
def gpu_stats():
    if not torch.cuda.is_available():
        return {}
    return {
        "vram_alloc_mb":    torch.cuda.memory_allocated() / 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved()  / 1024**2,
    }


# ============================================================
# DISCOVER SUBFOLDERS
# ============================================================
def discover_T_dirs(sweep_dir):
    entries = []
    for name in os.listdir(sweep_dir):
        path = os.path.join(sweep_dir, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("T_"):
            continue
        try:
            T_val = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if not (os.path.exists(os.path.join(path, "meta.pt")) and
                os.path.exists(os.path.join(path, "ckpt_last.pt"))):
            print(f"  Skipping {name} — missing meta.pt or ckpt_last.pt")
            continue
        if T_val > 500:
            continue
        entries.append((T_val, path))
    entries.sort(key=lambda x: x[0])
    return entries


# ============================================================
# BENCHMARK HELPERS
# ============================================================
def time_batch_gpu(meta, ddpm, device, n_events, batch_size, n_warmup):
    """GPU batch: returns (time_per_event_mins, throughput_ev_per_sec)."""
    n_batches = max(1, math.ceil(n_events / batch_size))
    for _ in range(n_warmup):
        sample_batch(meta, ddpm, device, batch_size=batch_size)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_batches):
        sample_batch(meta, ddpm, device, batch_size=batch_size)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    total = n_batches * batch_size
    elapsed = t1 - t0
    return elapsed / total / 60.0, total / elapsed


def time_single_gpu(meta, ddpm, device, n_events, n_warmup):
    """GPU single-event: returns (time_per_event_mins, throughput_ev_per_sec)."""
    for _ in range(n_warmup):
        sample_single(meta, ddpm, device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_events):
        sample_single(meta, ddpm, device)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    return elapsed / n_events / 60.0, n_events / elapsed


def time_single_cpu(meta, ddpm_cpu):
    """CPU single-event (1 event, no warmup): returns time_per_event_mins."""
    t0 = time.perf_counter()
    sample_single(meta, ddpm_cpu, "cpu")
    t1 = time.perf_counter()
    return (t1 - t0) / 60.0


# ============================================================
# PLOT
# ============================================================
def plot_results(T_vals, times_batch, times_single_g, times_single_c,
                 outpath="timing_Tsweep.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    series = [
        ("GPU Batch Sampling",  times_batch,    "tab:blue",   "-.",  "D"),
        ("GPU Single Sampling", times_single_g, "tab:orange", "-",   "o"),
        ("CPU Single Sampling", times_single_c, "tab:red",    "--",  "s"),
    ]

    for ax_idx, ax in enumerate(axes):
        for label, times, color, ls, marker in series:
            if times is None:
                continue
            kw = dict(label=label, color=color, linestyle=ls,
                      marker=marker, linewidth=2, markersize=8)
            if ax_idx == 0:
                ax.semilogy(T_vals, times, **kw)
            else:
                ax.plot(T_vals, times, **kw)

        scale = "log scale" if ax_idx == 0 else "linear scale"
        ax.set_xlabel("Diffusion steps T", fontsize=13)
        ax.set_ylabel("Generation time per event (mins)", fontsize=13)
        ax.set_title(f"Sampling time vs T  [{scale}]", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xticks(T_vals)

    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {outpath}")


# ============================================================
# SAVE TEXT RESULTS
# ============================================================
def save_results(T_vals, times_batch, times_single_g, times_single_c,
                 throughputs, outpath):
    with open(outpath, "w") as f:
        f.write(f"{'T':>8}  {'Batch GPU (ms)':>16}  {'Single GPU (ms)':>16}"
                f"  {'Single CPU (ms)':>16}  {'Throughput (ev/s)':>20}\n")
        f.write("-" * 84 + "\n")
        for i, T in enumerate(T_vals):
            tb  = times_batch[i]    * 60000 if times_batch    else float("nan")
            tsg = times_single_g[i] * 60000 if times_single_g else float("nan")
            tsc = times_single_c[i] * 60000 if times_single_c else float("nan")
            tp  = throughputs[i]            if throughputs     else float("nan")
            f.write(f"{T:>8d}  {tb:>16.3f}  {tsg:>16.3f}"
                    f"  {tsc:>16.1f}  {tp:>20.2f}\n")
    print(f"Results saved to {outpath}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Time sampling across T_sweep subdirectories"
    )
    parser.add_argument("--sweep_dir",  required=True,
                        help="Parent dir containing T_25, T_50, ... subfolders")
    parser.add_argument("--n_events",   type=int, default=128,
                        help="Events to time for GPU batch and GPU single")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for GPU batch sampling")
    parser.add_argument("--n_warmup",   type=int, default=3,
                        help="Warmup iterations before GPU timing")
    parser.add_argument("--no_cpu",     action="store_true",
                        help="Skip CPU timing entirely")
    parser.add_argument("--no_single",  action="store_true",
                        help="Skip GPU single-event timing")
    parser.add_argument("--plot_out",   type=str, default=None)
    parser.add_argument("--plot_only",  action="store_true",
                        help="Skip benchmarking, just replot from existing .txt results")
    args = parser.parse_args()

    if args.plot_out is None:
        args.plot_out = os.path.join(args.sweep_dir, "timing_Tsweep.png")

    # ── Plot-only mode ────────────────────────────────────────
    if args.plot_only:
        txt_path = args.plot_out.replace(".png", ".txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Results file not found: {txt_path}")
        T_vals=[]; times_batch=[]; times_single_g=[]; times_single_c=[]
        with open(txt_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("T") or line.startswith("-"):
                    continue
                parts = line.split()
                T_vals.append(int(parts[0]))
                times_batch.append(float(parts[1]) / 60000 / 10)
                times_single_g.append(float(parts[2]) / 60000 / 10)
                times_single_c.append(float(parts[3]) / 60000 / 10)
        plot_results(T_vals, times_batch, times_single_g, times_single_c,
                     outpath=args.plot_out)
        print("Done.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU found — GPU benchmarks will run on CPU.")

    entries = discover_T_dirs(args.sweep_dir)
    if not entries:
        raise RuntimeError(f"No valid T_* subdirectories found in {args.sweep_dir}")

    print(f"\nFound {len(entries)} models: {[f'T={T}' for T, _ in entries]}")
    print(f"Device: {device}  |  batch_size={args.batch_size}  |  n_events={args.n_events}")
    print(f"CPU timing: {'disabled' if args.no_cpu else '1 event per model (no warmup)'}\n")

    T_vals         = []
    times_batch    = []
    times_single_g = [] if not args.no_single else None
    times_single_c = [] if not args.no_cpu    else None
    throughputs    = []

    for T_val, path in tqdm(entries, desc="Models"):
        print(f"\n{'='*55}")
        print(f"  T = {T_val}  ({path})")
        print(f"{'='*55}")

        # ── GPU batch ─────────────────────────────────────────
        meta, _, ddpm_gpu = load_meta_and_model(path, device)
        t_batch, tp = time_batch_gpu(
            meta, ddpm_gpu, device,
            n_events=args.n_events,
            batch_size=args.batch_size,
            n_warmup=args.n_warmup,
        )
        print(f"  GPU batch  : {t_batch*60000:.2f} ms/event  |  {tp:.1f} ev/s")
        stats = gpu_stats()
        if stats:
            print(f"  VRAM alloc : {stats['vram_alloc_mb']:.0f} MB")

        # ── GPU single ────────────────────────────────────────
        if not args.no_single:
            t_sg, tp_sg = time_single_gpu(
                meta, ddpm_gpu, device,
                n_events=args.n_events,
                n_warmup=args.n_warmup,
            )
            print(f"  GPU single : {t_sg*60000:.2f} ms/event  |  {tp_sg:.1f} ev/s")
            times_single_g.append(t_sg)

        del ddpm_gpu
        if device == "cuda":
            torch.cuda.empty_cache()

        # ── CPU single (1 event, no warmup) ───────────────────
        if not args.no_cpu:
            print(f"  CPU single : timing 1 event …", end=" ", flush=True)
            _, _, ddpm_cpu = load_meta_and_model(path, "cpu")
            t_sc = time_single_cpu(meta, ddpm_cpu)
            del ddpm_cpu
            print(f"{t_sc*60000:.1f} ms")
            times_single_c.append(t_sc)

        T_vals.append(T_val)
        times_batch.append(t_batch)
        throughputs.append(tp)

    # ── Plot ──────────────────────────────────────────────────
    plot_results(T_vals, times_batch, times_single_g, times_single_c,
                 outpath=args.plot_out)

    # ── Save text ─────────────────────────────────────────────
    txt_out = args.plot_out.replace(".png", ".txt")
    save_results(T_vals, times_batch, times_single_g, times_single_c,
                 throughputs, txt_out)

    # ── Summary table ─────────────────────────────────────────
    print(f"\n{'T':>8}  {'Batch GPU (ms)':>16}  {'Single GPU (ms)':>16}"
          f"  {'Single CPU (ms)':>16}  {'Throughput (ev/s)':>20}")
    print("-" * 84)
    for i, T in enumerate(T_vals):
        tsg = times_single_g[i] * 60000 if times_single_g else float("nan")
        tsc = times_single_c[i] * 60000 if times_single_c else float("nan")
        print(f"{T:>8d}  {times_batch[i]*60000:>16.2f}  {tsg:>16.2f}"
              f"  {tsc:>16.1f}  {throughputs[i]:>20.1f}")


if __name__ == "__main__":
    main()