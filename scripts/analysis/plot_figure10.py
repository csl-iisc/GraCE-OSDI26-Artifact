#!/usr/bin/env python3
"""Generate Figure 10 using the exact ablation plotting style from the supplied notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.transforms import Affine2D

from common import (
    copy_to_stable,
    default_acronym_file,
    ensure_output_dirs,
    gmean,
    load_acronym_mapping,
    merge_ablation,
    model_labels,
    resolve_repo_root,
    resolve_run_id,
    select_top_n,
)


def compute_speedups(merged: pd.DataFrame) -> pd.DataFrame:
    speedup_data = []
    for _, row in merged.iterrows():
        no_cudagraph_timings = row["no_cudagraph_raw_timings"]
        cudagraph_unpatched_timings = row["cudagraph_unpatched_raw_timings"]
        cudagraph_patched_timings = row["cudagraph_patched_raw_timings"]
        opt1_patched_timings = row["opt1_patched_raw_timings"]
        opt1_opt2_patched_timings = row["opt1_opt2_patched_raw_timings"]

        median_no_cg = np.median(no_cudagraph_timings)
        std_no_cg = np.std(no_cudagraph_timings)
        median_cg_un = np.median(cudagraph_unpatched_timings)
        std_cg_un = np.std(cudagraph_unpatched_timings)
        median_cg_pat = np.median(cudagraph_patched_timings)
        std_cg_pat = np.std(cudagraph_patched_timings)
        median_opt1 = np.median(opt1_patched_timings)
        std_opt1 = np.std(opt1_patched_timings)
        median_opt1o2 = np.median(opt1_opt2_patched_timings)
        std_opt1o2 = np.std(opt1_opt2_patched_timings)

        speedup_unpatched = median_no_cg / median_cg_un
        speedup_opt1 = median_no_cg / median_opt1
        speedup_opt1_opt2 = median_no_cg / median_opt1o2
        speedup_patched = median_no_cg / median_cg_pat
        speedup_patched_unpatched = median_cg_un / median_cg_pat

        error_unpatched = speedup_unpatched * np.sqrt((std_no_cg / median_no_cg) ** 2 + (std_cg_un / median_cg_un) ** 2)
        error_opt1 = speedup_opt1 * np.sqrt((std_no_cg / median_no_cg) ** 2 + (std_opt1 / median_opt1) ** 2)
        error_opt1_opt2 = speedup_opt1_opt2 * np.sqrt((std_no_cg / median_no_cg) ** 2 + (std_opt1o2 / median_opt1o2) ** 2)
        error_patched = speedup_patched * np.sqrt((std_no_cg / median_no_cg) ** 2 + (std_cg_pat / median_cg_pat) ** 2)
        error_patched_unpatched = speedup_patched_unpatched * np.sqrt((std_cg_un / median_cg_un) ** 2 + (std_cg_pat / median_cg_pat) ** 2)

        speedup_data.append({
            "name": row["name"],
            "suite": row["suite"],
            "mode": row["mode"],
            "batch_size": row["batch_size"],
            "speedup_unpatched": speedup_unpatched,
            "error_unpatched": error_unpatched,
            "speedup_opt1": speedup_opt1,
            "error_opt1": error_opt1,
            "speedup_opt1_opt2": speedup_opt1_opt2,
            "error_opt1_opt2": error_opt1_opt2,
            "speedup_patched": speedup_patched,
            "error_patched": error_patched,
            "speedup_patched_unpatched": speedup_patched_unpatched,
            "error_patched_unpatched": error_patched_unpatched,
        })
    return pd.DataFrame(speedup_data)


def add_labels(ax, rects, vals):
    for rect, val in zip(rects, vals):
        h = rect.get_height()
        ax.annotate(
            f"{val:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            fontweight="bold",
        )


def annotate_geomean(ax, x, val):
    ax.annotate(
        f"{val:.2f}",
        xy=(x, val),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        rotation=90,
        fontsize=8,
        fontweight="bold",
    )


def plot_ablation(top_speedup_df: pd.DataFrame, acronym_mapping, out_pdf: Path):
    geomean_unpatched = gmean(top_speedup_df["speedup_unpatched"])
    geomean_opt1 = gmean(top_speedup_df["speedup_opt1"])
    geomean_opt1_opt2 = gmean(top_speedup_df["speedup_opt1_opt2"])
    geomean_patched = gmean(top_speedup_df["speedup_patched"])

    x_vals = np.arange(len(top_speedup_df))
    geomean_x_val = len(x_vals) + 1

    fig, ax = plt.subplots(figsize=(max(len(top_speedup_df) * 0.6, 0), 2))

    width = 0.25
    overlay_width = width

    rects_opt1 = ax.bar(
        x_vals - width,
        top_speedup_df["speedup_opt1"],
        width,
        label="CUDA Graph-aware Code Transformation (CGCT)",
        color="#6fd6ff",
        edgecolor="black",
    )

    rects_unpatched = ax.bar(
        x_vals - width,
        top_speedup_df["speedup_unpatched"],
        overlay_width,
        label="PyTorch2-CG",
        color="#1f77b4",
        edgecolor="black",
        zorder=4,
    )

    rects_opt1_opt2 = ax.bar(
        x_vals,
        top_speedup_df["speedup_opt1_opt2"],
        width,
        label="+Parameter Indirection (PI)",
        color="#ff7f0e",
        edgecolor="black",
    )

    rects_patched = ax.bar(
        x_vals + width,
        top_speedup_df["speedup_patched"],
        width,
        label="+Selective CUDA Graphs (SCG)",
        color="#2ca02c",
        edgecolor="black",
    )

    ax.bar(geomean_x_val - width, geomean_opt1, width, color="#6fd6ff", edgecolor="black")
    ax.bar(geomean_x_val - width, geomean_unpatched, overlay_width, color="#1f77b4", edgecolor="black", zorder=4)
    ax.bar(geomean_x_val, geomean_opt1_opt2, width, color="#ff7f0e", edgecolor="black")
    ax.bar(geomean_x_val + width, geomean_patched, width, color="#2ca02c", edgecolor="black")

    ax.set_ylabel("Speedup", fontsize=10, fontweight="bold")

    ax.set_xticks(np.append(x_vals, geomean_x_val))
    ax.set_xticklabels(
        model_labels(top_speedup_df, acronym_mapping) + ["Geomean"],
        rotation=45,
        ha="right",
        fontsize=9,
        fontweight="bold",
    )

    dx_pts = 0.3
    for lbl in ax.get_xticklabels():
        lbl.set_transform(Affine2D().translate(dx_pts, 0) + lbl.get_transform())

    add_labels(ax, rects_opt1, top_speedup_df["speedup_opt1"])
    add_labels(ax, rects_opt1_opt2, top_speedup_df["speedup_opt1_opt2"])
    add_labels(ax, rects_patched, top_speedup_df["speedup_patched"])

    annotate_geomean(ax, geomean_x_val - width, geomean_opt1)
    annotate_geomean(ax, geomean_x_val, geomean_opt1_opt2)
    annotate_geomean(ax, geomean_x_val + width, geomean_patched)

    max_speedup = max(
        top_speedup_df["speedup_opt1"].max(),
        top_speedup_df["speedup_opt1_opt2"].max(),
        top_speedup_df["speedup_patched"].max(),
        geomean_opt1,
        geomean_opt1_opt2,
        geomean_patched,
    )
    yticks = np.arange(0.5, max_speedup + 0.5, 0.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t}x" if t != 0 else "" for t in yticks], fontsize=10, fontweight="bold")
    ax.set_ylim(0.5, max_speedup + 0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.7, color="gray")

    empty_handle = Patch(facecolor="none", edgecolor="none", linewidth=0)
    handles = [
        rects_unpatched[0], empty_handle, empty_handle,
        rects_opt1[0], rects_opt1_opt2[0], rects_patched[0],
    ]
    labels = [
        "PyTorch2-CG", "", "",
        "CUDA Graph-aware Code Transformation (CGCT)",
        "+Parameter Indirection (PI)",
        "+Selective CUDA Graphs (SCG)",
    ]

    legend = ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.15),
        ncol=2,
        frameon=False,
        fontsize=9,
        columnspacing=1.0,
        handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    geomean_row = {
        "name": "Geomean",
        "suite": "",
        "mode": "",
        "batch_size": "",
        "speedup_unpatched": geomean_unpatched,
        "speedup_opt1": geomean_opt1,
        "speedup_opt1_opt2": geomean_opt1_opt2,
        "speedup_patched": geomean_patched,
    }
    return pd.concat([top_speedup_df, pd.DataFrame([geomean_row])], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--single-gpu-root", default="results/raw/single_gpu")
    parser.add_argument("--acronym-file", default=None)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    raw_root = repo_root / args.single_gpu_root
    run_id = resolve_run_id(raw_root, args.run_id)
    run_dir = raw_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Single-GPU run directory not found: {run_dir}")

    acronym_file = default_acronym_file(repo_root, args.acronym_file)
    acronym_mapping = load_acronym_mapping(acronym_file)

    figures_dir, run_figures_dir, processed_dir = ensure_output_dirs(repo_root, "figure10", run_id)

    merged = merge_ablation(run_dir)
    speedup_df = compute_speedups(merged)
    top_speedup_df = select_top_n(speedup_df, args.top_n)

    speedup_df.to_csv(processed_dir / "figure10_all_speedups.csv", index=False)
    top_speedup_df.to_csv(processed_dir / "figure10_top25_without_geomean.csv", index=False)

    out_pdf = run_figures_dir / f"Ablation_Study_Top_{args.top_n}_Performance_With_Geomeans_superimposed.pdf"
    plot_df = plot_ablation(top_speedup_df, acronym_mapping, out_pdf)
    plot_df.to_csv(processed_dir / "figure10_top25_with_geomean.csv", index=False)

    copy_to_stable(out_pdf, figures_dir / "figure10.pdf")
    print(f"[OK] Wrote {out_pdf}")
    print(f"[OK] Wrote {figures_dir / 'figure10.pdf'}")


if __name__ == "__main__":
    main()
