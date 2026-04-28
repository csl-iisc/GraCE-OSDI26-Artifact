#!/usr/bin/env python3
"""Generate Figure 9 using the exact plotting style from the supplied notebook."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Affine2D

from common import (
    copy_to_stable,
    default_acronym_file,
    ensure_output_dirs,
    gmean,
    load_acronym_mapping,
    merge_three_way,
    model_labels,
    resolve_repo_root,
    resolve_run_id,
    select_top_n,
)


def compute_speedups(merged: pd.DataFrame) -> pd.DataFrame:
    speedup_data = []
    for _, row in merged.iterrows():
        no_cudagraph_timings = row["no_cudagraph_raw_timings"]
        cudagraph_patched_timings = row["cudagraph_patched_raw_timings"]
        cudagraph_unpatched_timings = row["cudagraph_unpatched_raw_timings"]

        median_no_cudagraph = np.median(no_cudagraph_timings)
        std_no_cudagraph = np.std(no_cudagraph_timings)

        median_cudagraph_patched = np.median(cudagraph_patched_timings)
        std_cudagraph_patched = np.std(cudagraph_patched_timings)

        median_cudagraph_unpatched = np.median(cudagraph_unpatched_timings)
        std_cudagraph_unpatched = np.std(cudagraph_unpatched_timings)

        speedup_unpatched = median_no_cudagraph / median_cudagraph_unpatched
        speedup_patched = median_no_cudagraph / median_cudagraph_patched
        speedup_patched_unpatched = median_cudagraph_unpatched / median_cudagraph_patched

        error_unpatched = speedup_unpatched * np.sqrt(
            (std_no_cudagraph / median_no_cudagraph) ** 2
            + (std_cudagraph_unpatched / median_cudagraph_unpatched) ** 2
        )
        error_patched = speedup_patched * np.sqrt(
            (std_no_cudagraph / median_no_cudagraph) ** 2
            + (std_cudagraph_patched / median_cudagraph_patched) ** 2
        )
        error_patched_unpatched = speedup_patched_unpatched * np.sqrt(
            (std_cudagraph_unpatched / median_cudagraph_unpatched) ** 2
            + (std_cudagraph_patched / median_cudagraph_patched) ** 2
        )

        speedup_data.append({
            "name": row["name"],
            "suite": row["suite"],
            "mode": row["mode"],
            "batch_size": row["batch_size"],
            "speedup_unpatched": speedup_unpatched,
            "error_unpatched": error_unpatched,
            "speedup_patched": speedup_patched,
            "error_patched": error_patched,
            "speedup_patched_unpatched": speedup_patched_unpatched,
            "error_patched_unpatched": error_patched_unpatched,
        })
    return pd.DataFrame(speedup_data)


def add_labels(ax, rects, speedups, fontsize=9):
    for rect, speedup in zip(rects, speedups):
        height = rect.get_height()
        ax.annotate(
            f"{speedup:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=90,
            fontweight="bold",
        )


def plot_two_bars(top_speedup_df, acronym_mapping, out_pdf: Path, system_label: str):
    # This retains the first notebook plot with error bars, in case you want it.
    fig, ax = plt.subplots(figsize=(max(len(top_speedup_df) * 1, 0), 2.0))
    x_vals = np.arange(len(top_speedup_df))
    width = 0.3

    rects2 = ax.bar(
        x_vals - 0.15,
        top_speedup_df["speedup_unpatched"],
        width,
        yerr=top_speedup_df["error_unpatched"],
        label="torch.compile w/ CUDAgraph",
        capsize=5,
        edgecolor="black",
    )
    rects3 = ax.bar(
        x_vals + 0.15,
        top_speedup_df["speedup_patched"],
        width,
        yerr=top_speedup_df["error_patched"],
        label=system_label,
        capsize=5,
        edgecolor="black",
    )

    ax.set_ylabel("Speedup", fontsize=14, fontweight="bold")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(
        model_labels(top_speedup_df, acronym_mapping),
        rotation=45,
        ha="right",
        fontsize=10,
        fontweight="bold",
    )
    add_labels(ax, rects2, top_speedup_df["speedup_unpatched"], fontsize=9)
    add_labels(ax, rects3, top_speedup_df["speedup_patched"], fontsize=9)

    max_speedup = max(top_speedup_df["speedup_patched"].max(), top_speedup_df["speedup_unpatched"].max())
    yticks = np.arange(0.5, max_speedup + 0.1, 0.5)
    plt.yticks(ticks=yticks, labels=[f"{tick}x" if tick != 0 else "" for tick in yticks], fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, max_speedup + 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = plt.legend(loc="upper right", bbox_to_anchor=(0.5, 1.5), ncol=3, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.7, color="gray")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_with_geomean(top_speedup_df, acronym_mapping, out_pdf: Path, system_label: str):
    geomean_unpatched = gmean(top_speedup_df["speedup_unpatched"])
    geomean_patched = gmean(top_speedup_df["speedup_patched"])

    geomean_data = {
        "name": "Geomean",
        "suite": "",
        "mode": "",
        "batch_size": "",
        "speedup_unpatched": geomean_unpatched,
        "error_unpatched": 0,
        "speedup_patched": geomean_patched,
        "error_patched": 0,
    }
    plot_df = pd.concat([top_speedup_df, pd.DataFrame([geomean_data])], ignore_index=True)

    x_vals = np.arange(len(plot_df) - 1)
    geomean_x_val = len(x_vals) + 1

    fig, ax = plt.subplots(figsize=(max(len(plot_df) * 0.5, 0), 2))
    width = 0.3

    rects2 = ax.bar(
        x_vals - 0.15,
        plot_df.iloc[:-1]["speedup_unpatched"],
        width,
        label="PyTorch2-CG",
        capsize=5,
        edgecolor="black",
    )
    rects3 = ax.bar(
        x_vals + 0.15,
        plot_df.iloc[:-1]["speedup_patched"],
        width,
        label=system_label,
        capsize=5,
        edgecolor="black",
    )

    ax.bar(geomean_x_val - 0.15, geomean_unpatched, width, color=rects2[0].get_facecolor(), edgecolor="black")
    ax.bar(geomean_x_val + 0.15, geomean_patched, width, color=rects3[0].get_facecolor(), edgecolor="black")

    ax.set_ylabel("Speedup", fontsize=10, fontweight="bold")
    ax.set_xticks(np.append(x_vals, geomean_x_val))
    ax.set_xticklabels(
        model_labels(plot_df, acronym_mapping),
        rotation=45,
        ha="right",
        fontsize=9,
        fontweight="bold",
    )

    dx_pts = 0.3
    for lbl in ax.get_xticklabels():
        lbl.set_transform(Affine2D().translate(dx_pts, 0) + lbl.get_transform())

    add_labels(ax, rects2, plot_df.iloc[:-1]["speedup_unpatched"], fontsize=9)
    add_labels(ax, rects3, plot_df.iloc[:-1]["speedup_patched"], fontsize=9)

    ax.annotate(f"{geomean_unpatched:.2f}", xy=(geomean_x_val - 0.15, geomean_unpatched), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9, rotation=90, fontweight="bold")
    ax.annotate(f"{geomean_patched:.2f}", xy=(geomean_x_val + 0.15, geomean_patched), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9, rotation=90, fontweight="bold")

    max_speedup = max(plot_df["speedup_patched"].max(), plot_df["speedup_unpatched"].max(), geomean_patched, geomean_unpatched)
    yticks = np.arange(0.5, max_speedup + 2, 0.5)
    plt.yticks(ticks=yticks, labels=[f"{tick}x" for tick in yticks], fontsize=10, fontweight="bold")
    ax.set_ylim(0.5, max_speedup + 0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.1), ncol=3, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.7, color="gray")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--single-gpu-root", default="results/raw/single_gpu")
    parser.add_argument("--acronym-file", default=None)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--system-label", default="GraCE")
    args = parser.parse_args()

    repo_root = resolve_repo_root(args.repo_root)
    raw_root = repo_root / args.single_gpu_root
    run_id = resolve_run_id(raw_root, args.run_id)
    run_dir = raw_root / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Single-GPU run directory not found: {run_dir}")

    acronym_file = default_acronym_file(repo_root, args.acronym_file)
    acronym_mapping = load_acronym_mapping(acronym_file)

    figures_dir, run_figures_dir, processed_dir = ensure_output_dirs(repo_root, "figure9", run_id)

    merged = merge_three_way(run_dir)
    speedup_df = compute_speedups(merged)
    top_speedup_df = select_top_n(speedup_df, args.top_n)

    speedup_df.to_csv(processed_dir / "figure9_all_speedups.csv", index=False)
    top_speedup_df.to_csv(processed_dir / "figure9_top25_without_geomean.csv", index=False)

    two_bar_pdf = run_figures_dir / f"top_{args.top_n}_speedup_comparison_two_bars.pdf"
    geomean_pdf = run_figures_dir / f"top_{args.top_n}_speedup_comparison_with_geomean.pdf"

    plot_two_bars(top_speedup_df, acronym_mapping, two_bar_pdf, args.system_label)
    plot_df = plot_with_geomean(top_speedup_df, acronym_mapping, geomean_pdf, args.system_label)
    plot_df.to_csv(processed_dir / "figure9_top25_with_geomean.csv", index=False)

    copy_to_stable(geomean_pdf, figures_dir / "figure9.pdf")
    print(f"[OK] Wrote {geomean_pdf}")
    print(f"[OK] Wrote {figures_dir / 'figure9.pdf'}")


if __name__ == "__main__":
    main()
