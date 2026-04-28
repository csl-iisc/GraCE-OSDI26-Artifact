#!/usr/bin/env python3
"""
Build the TP experiment summary and generate the TP scaling plot for the
GraCE OSDI'26 artifact.

Expected raw input layout:
  results/raw/tp/<run_id>/
    TP-1/vanilla/no-cg/*.csv
    TP-1/vanilla/cg/*.csv
    TP-1/full/cg/*.csv
    TP-2/...
    TP-4/...

The script writes:
  results/processed/tp/<run_id>/tp_summary_wide.csv
  results/processed/tp/<run_id>/tp_summary_long.csv
  figures/<run_id>/figure11_tp_scaling.pdf
  figures/figure11.pdf
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Artifact-specific defaults
# -----------------------------------------------------------------------------

DEFAULT_TP_VALUES = [1, 2, 4]
RANK_TOL_PCT = 5.0
RANK_TOL_REL = RANK_TOL_PCT / 100.0

# The artifact TP runs currently produce these three configurations.
CONFIGS = [
    {
        "key": "vanilla_ncg",
        "label": "PyTorch_NCG",
        "variant": "vanilla",
        "graph_mode": "no-cg",
        "cg_flag": "no_cudagraphs",
        "plot_label": "PyTorch2 no-CG",
    },
    {
        "key": "vanilla_cg",
        "label": "PyTorch_CG",
        "variant": "vanilla",
        "graph_mode": "cg",
        "cg_flag": "with_cudagraphs",
        "plot_label": "PyTorch2-CG",
    },
    {
        "key": "full_grace",
        "label": "Full_GraCE",
        "variant": "full",
        "graph_mode": "cg",
        "cg_flag": "with_cudagraphs",
        "plot_label": "GraCE",
    },
]

# Kept aligned with the original TP summary/plot scripts.
MODEL_ORDER = [
    "XLNetLMHeadModel",
    "DALLE2_pytorch",
    "nvidia_deeprecommender",
    "speech_transformer",
    "vision_maskrcnn",
]

MODEL_SHORT = {
    "XLNetLMHeadModel": "XLNET",
    "DALLE2_pytorch": "DALLE2",
    "nvidia_deeprecommender": "DR",
    "speech_transformer": "ST",
    "vision_maskrcnn": "VM",
}

# Batch-size choices used in the original plotting notebook.
VALID_BS_FOR_PLOT = {
    "XLNET": "bs128",
    "DALLE2": "bs2",
    "ST": "bs1",
    "VM": "bs1",
}

PLOT_MODELS = ["XLNET", "DALLE2", "ST", "VM"]
PLOT_TPS = ["TP1", "TP2", "TP4"]


# -----------------------------------------------------------------------------
# Small logging helpers
# -----------------------------------------------------------------------------


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


# -----------------------------------------------------------------------------
# Input parsing helpers
# -----------------------------------------------------------------------------


def resolve_run_id(results_root: Path, run_id: str) -> str:
    if run_id != "latest":
        return run_id

    latest_file = results_root / "latest.txt"
    if not latest_file.exists():
        raise FileNotFoundError(f"latest.txt not found: {latest_file}")

    resolved = latest_file.read_text().strip()
    if not resolved:
        raise RuntimeError(f"latest.txt is empty: {latest_file}")

    return resolved


def parse_rank_from_filename(path: Path) -> int:
    """Extract rank from *_rank_X.csv. If absent, this is a TP=1/rank-0 file."""
    m = re.search(r"_rank_(\d+)\.csv$", path.name)
    if m:
        return int(m.group(1))
    return 0


def parse_suite_mode_dtype_from_filename(path: Path) -> Tuple[str, str, str]:
    """Best-effort parse from generated CSV filename."""
    name = path.name

    # Suite names used by run_workloads.sh.
    suite = "unknown"
    for candidate in ["huggingface", "torchbench", "timm_models"]:
        if f"_{candidate}_" in name:
            suite = candidate
            break

    dtype = "unknown"
    for candidate in ["bfloat16", "amp", "float32"]:
        if f"_{candidate}_" in name:
            dtype = candidate
            break

    mode = "unknown"
    for candidate in ["inference", "training"]:
        if f"_{candidate}_" in name:
            mode = candidate
            break

    return suite, mode, dtype


def batch_to_bs_str(value: object) -> str:
    try:
        return f"bs{int(value)}"
    except Exception:
        text = str(value)
        return text if text.startswith("bs") else f"bs{text}"


def collect_medians_for_config(
    run_dir: Path,
    tp: int,
    config: dict,
    suite_filter: Optional[Iterable[str]] = None,
) -> Dict[Tuple[str, str, str, str], float]:
    """
    Return:
      dict[(model, suite, mode, bs_str)] = aggregated_median_latency

    For TP>1, each rank writes a CSV. We use the max median latency across ranks,
    preserving the behavior of the original summary script.
    """
    case_dir = run_dir / f"TP-{tp}" / config["variant"] / config["graph_mode"]

    if not case_dir.is_dir():
        warn(
            "Missing directory: "
            f"{case_dir} (skipping {config['key']}, TP={tp})"
        )
        return {}

    pattern = f"inductor_{config['cg_flag']}_*_cuda_performance*.csv"
    csv_files = sorted(case_dir.glob(pattern))

    # The pattern above intentionally excludes .json because it ends with .json,
    # but keep this guard in case the naming changes later.
    csv_files = [p for p in csv_files if p.suffix == ".csv"]

    if not csv_files:
        warn(f"No CSV files found for pattern: {case_dir / pattern}")
        return {}

    allowed_suites = set(suite_filter or [])

    # data[(model, suite, mode, bs_str)][rank] = median_latency
    data: Dict[Tuple[str, str, str, str], Dict[int, float]] = defaultdict(dict)

    for csv_path in csv_files:
        suite, mode, _dtype = parse_suite_mode_dtype_from_filename(csv_path)
        if allowed_suites and suite not in allowed_suites:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            warn(f"Failed to read CSV {csv_path}: {e}")
            continue

        if "name" not in df.columns or "batch_size" not in df.columns:
            warn(f"Skipping CSV with missing name/batch_size columns: {csv_path}")
            continue

        latency_col = None
        for candidate in ["median_latency", "median", "latency"]:
            if candidate in df.columns:
                latency_col = candidate
                break

        if latency_col is None:
            warn(f"Skipping CSV with no median latency column: {csv_path}")
            continue

        rank = parse_rank_from_filename(csv_path)

        for _, row in df.iterrows():
            model = str(row["name"])
            bs_str = batch_to_bs_str(row["batch_size"])
            med = float(row[latency_col])
            data[(model, suite, mode, bs_str)][rank] = med

    agg: Dict[Tuple[str, str, str, str], float] = {}
    for row_key, rank_dict in data.items():
        model, suite, mode, bs_str = row_key
        ranks = sorted(rank_dict.keys())
        meds = [rank_dict[r] for r in ranks]

        if len(meds) > 1:
            min_med = min(meds)
            max_med = max(meds)
            rel_diff = math.inf if min_med <= 0 else (max_med - min_med) / min_med
            if rel_diff > RANK_TOL_REL:
                warn(
                    "Large rank-to-rank variance for "
                    f"model={model}, suite={suite}, mode={mode}, batch_size={bs_str}, "
                    f"TP={tp}, variant={config['variant']}, graph={config['graph_mode']}: "
                    f"min={min_med:.6f}, max={max_med:.6f}, "
                    f"rel_diff={rel_diff * 100.0:.3f}%"
                )

        agg[row_key] = max(meds)

    info(
        f"Collected {len(agg)} rows for TP={tp}, "
        f"config={config['key']} from {case_dir}"
    )
    return agg


# -----------------------------------------------------------------------------
# Summary construction
# -----------------------------------------------------------------------------


def build_summary_df(run_dir: Path, tp_values: List[int]) -> pd.DataFrame:
    rows: Dict[Tuple[str, str, str, str], dict] = {}

    for tp in tp_values:
        for config in CONFIGS:
            per_key_medians = collect_medians_for_config(run_dir, tp, config)
            if not per_key_medians:
                continue

            for (model, suite, mode, bs_str), med in per_key_medians.items():
                row_key = (model, suite, mode, bs_str)

                if row_key not in rows:
                    rows[row_key] = {
                        "Model": model,
                        "suite": suite,
                        "mode": mode,
                        "batch_size": bs_str,
                    }

                rows[row_key][f"TP{tp}_{config['label']}"] = med

    df = pd.DataFrame(rows.values())
    if df.empty:
        raise RuntimeError(f"No TP results could be parsed from {run_dir}")

    df["ModelShort"] = df["Model"].map(MODEL_SHORT).fillna(df["Model"])

    df["_bs_num"] = (
        df["batch_size"]
        .astype(str)
        .str.extract(r"bs(\d+)", expand=False)
        .fillna("0")
        .astype(int)
    )

    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    df["_model_rank"] = df["Model"].map(order_map).fillna(10_000).astype(int)

    df = df.sort_values(
        by=["_model_rank", "_bs_num", "suite", "mode"]
    ).reset_index(drop=True)

    return df.drop(columns=["_bs_num", "_model_rank"])


def build_long_df(df_summary: pd.DataFrame, tp_values: List[int]) -> pd.DataFrame:
    labels = [c["label"] for c in CONFIGS]
    records = []

    for _, row in df_summary.iterrows():
        base = {
            "Model": row["Model"],
            "ModelShort": row["ModelShort"],
            "suite": row["suite"],
            "mode": row["mode"],
            "batch_size": row["batch_size"],
        }

        for tp in tp_values:
            rec = dict(base)
            rec["TP"] = f"TP{tp}"
            for label in labels:
                rec[label] = row.get(f"TP{tp}_{label}", pd.NA)
            records.append(rec)

    df_long = pd.DataFrame(records)

    df_long["_bs_num"] = (
        df_long["batch_size"]
        .astype(str)
        .str.extract(r"bs(\d+)", expand=False)
        .fillna("0")
        .astype(int)
    )
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    df_long["_model_rank"] = df_long["Model"].map(order_map).fillna(10_000).astype(int)

    df_long = df_long.sort_values(
        by=["_model_rank", "_bs_num", "suite", "mode", "TP"]
    ).reset_index(drop=True)

    return df_long.drop(columns=["_bs_num", "_model_rank"])


# -----------------------------------------------------------------------------
# Plot construction: intentionally close to the uploaded plotting script
# -----------------------------------------------------------------------------


def make_tp_plot(df_long: pd.DataFrame, out_pdf: Path, out_svg: Optional[Path] = None) -> None:
    df = df_long.copy()

    df["desired_bs"] = df["ModelShort"].map(VALID_BS_FOR_PLOT)

    # Prefer the paper-selected batch size, but do not crash if the artifact
    # workload file uses a different batch size for a model. In that case, use
    # the only/first available batch size and print a warning. This keeps the
    # artifact path robust while preserving the notebook style whenever the
    # expected batch size is present.
    selected_parts = []
    for model_short in PLOT_MODELS:
        sub = df[df["ModelShort"] == model_short].copy()
        if sub.empty:
            warn(f"No TP rows found for ModelShort={model_short}; it will be omitted from the plot")
            continue
        desired = VALID_BS_FOR_PLOT.get(model_short)
        desired_sub = sub[sub["batch_size"] == desired]
        if not desired_sub.empty:
            selected_parts.append(desired_sub)
            continue
        available = sorted(sub["batch_size"].dropna().astype(str).unique().tolist())
        if not available:
            warn(f"No batch sizes found for ModelShort={model_short}; it will be omitted from the plot")
            continue
        fallback_bs = available[0]
        warn(
            f"Desired batch size {desired} for ModelShort={model_short} not found; "
            f"using available batch size {fallback_bs}. Available={available}"
        )
        selected_parts.append(sub[sub["batch_size"] == fallback_bs])

    if not selected_parts:
        raise RuntimeError("No rows left after TP plot batch-size selection")

    df_plot = pd.concat(selected_parts, ignore_index=True)
    df_plot.drop(columns=["desired_bs"], inplace=True, errors="ignore")

    required_cols = ["PyTorch_NCG", "PyTorch_CG", "Full_GraCE"]
    missing_cols = [c for c in required_cols if c not in df_plot.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns for TP plot: {missing_cols}")

    df_plot["Speedup_CG_over_PyTorch_NCG"] = df_plot["PyTorch_NCG"] / df_plot["PyTorch_CG"]
    df_plot["Speedup_GraCE_over_PyTorch_NCG"] = df_plot["PyTorch_NCG"] / df_plot["Full_GraCE"]

    # ----------------- KNOBS / STYLE -----------------
    FIG_WIDTH = 4.3
    FIG_HEIGHT = 1.6

    GROUP_SPACING = 2.2
    TP_OFFSET_SPREAD = 0.7
    BAR_WIDTH = 0.20

    FONT_SIZE_XTICK = 7
    FONT_SIZE_YTICK = 7
    FONT_SIZE_YLABEL = 7
    FONT_SIZE_LEGEND = 7
    FONT_SIZE_ANNOT = 7

    XTICK_PAD = 3
    XTICK_LINE_SPACING = 1.4

    Y_LIM_MIN = 0.5
    Y_LIM_MAX = 2.5
    Y_TICK_STEP = 0.5

    GRID_LINEWIDTH = 0.8
    GRID_ALPHA = 0.6
    GRID_DASHES = (5, 2)

    SEP_COLOR = "0.8"
    SEP_LINEWIDTH = 0.8

    BAR_EDGE_COLOR = "black"
    BAR_EDGE_WIDTH = 0.6

    ANNOT_XOFFSETS = {
        ("XLNET", "TP2", "GraCE"): -0.15,
        ("XLNET", "TP4", "GraCE"): 0.17,
    }
    ANNOT_YOFFSETS = {}

    models = PLOT_MODELS
    tps = PLOT_TPS

    group_centers = np.arange(len(models)) * GROUP_SPACING
    tp_offsets = np.linspace(-TP_OFFSET_SPREAD, TP_OFFSET_SPREAD, len(tps))

    x_tp_centers = []
    cg_vals = []
    grace_vals = []
    tick_labels = []
    bar_models = []
    bar_tps = []

    for gi, m in enumerate(models):
        gc = group_centers[gi]
        for ti, tp in enumerate(tps):
            xc = gc + tp_offsets[ti]
            x_tp_centers.append(xc)

            rows = df_plot[(df_plot["ModelShort"] == m) & (df_plot["TP"] == tp)]
            if rows.empty:
                raise RuntimeError(
                    f"Missing TP plot row for ModelShort={m}, TP={tp}. "
                    f"Available rows:\n{df_plot[['ModelShort', 'batch_size', 'TP']].to_string(index=False)}"
                )
            row = rows.iloc[0]

            cg_vals.append(row["Speedup_CG_over_PyTorch_NCG"])
            grace_vals.append(row["Speedup_GraCE_over_PyTorch_NCG"])

            bar_models.append(m)
            bar_tps.append(tp)

            if ti == 1:
                tick_labels.append(f"{tp}\n{m}")
            else:
                tick_labels.append(tp)

    x_tp_centers = np.array(x_tp_centers)
    cg_vals = np.array(cg_vals, dtype=float)
    grace_vals = np.array(grace_vals, dtype=float)

    x_cg = x_tp_centers - BAR_WIDTH / 2
    x_grace = x_tp_centers + BAR_WIDTH / 2

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    bars_cg = ax.bar(
        x_cg,
        cg_vals,
        BAR_WIDTH,
        label="PyTorch2-CG",
        color="tab:blue",
        zorder=3,
    )
    bars_grace = ax.bar(
        x_grace,
        grace_vals,
        BAR_WIDTH,
        label="GraCE",
        color="tab:pink",
        zorder=3,
    )

    for b in list(bars_cg) + list(bars_grace):
        b.set_edgecolor(BAR_EDGE_COLOR)
        b.set_linewidth(BAR_EDGE_WIDTH)

    ax.set_ylabel("Speedup", fontweight="bold", fontsize=FONT_SIZE_YLABEL)

    ax.set_xticks(x_tp_centers)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_XTICK, pad=XTICK_PAD)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_YTICK)

    for t in ax.get_xticklabels():
        t.set_fontweight("bold")
        t.set_linespacing(XTICK_LINE_SPACING)
    for t in ax.get_yticklabels():
        t.set_fontweight("bold")

    ax.set_ylim(Y_LIM_MIN, Y_LIM_MAX)
    yticks = np.arange(Y_LIM_MIN, Y_LIM_MAX + 0.001, Y_TICK_STEP)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:.1f}x" for v in yticks])

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=GRID_LINEWIDTH,
        alpha=GRID_ALPHA,
        zorder=0,
        dashes=GRID_DASHES,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmin = x_tp_centers.min() - BAR_WIDTH - 0.5
    xmax = x_tp_centers.max() + BAR_WIDTH + 0.5
    ax.set_xlim(xmin, xmax)

    sep_positions = [
        0.5 * (group_centers[gi - 1] + group_centers[gi])
        for gi in range(1, len(models))
    ]
    for sep in sep_positions:
        ax.plot(
            [sep, sep],
            [-0.25, 1.02],
            transform=ax.get_xaxis_transform(),
            color=SEP_COLOR,
            linewidth=SEP_LINEWIDTH,
            zorder=1,
            clip_on=False,
        )

    leg = ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND, frameon=False)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")

    ANNOT_LEFT_SHIFT = 0#0.035  # small left shift from the bar center

    def annotate_above_limit(x_positions, heights, bar_kind):
        for x, h, m, tp in zip(x_positions, heights, bar_models, bar_tps):
            if h > Y_LIM_MAX:
                dx = ANNOT_XOFFSETS.get((m, tp, bar_kind), 0.0)
                dy = ANNOT_YOFFSETS.get((m, tp, bar_kind), 0.0)

                # Put annotation just to the left of the bar.
                # For GraCE bars this places it just left of the pink bar.
                x_text = x - ANNOT_LEFT_SHIFT + dx

                ax.text(
                    x_text,
                    Y_LIM_MAX + 0.03 + dy,
                    f"{h:.2f}x",
                    ha="right",
                    va="bottom",
                    fontsize=FONT_SIZE_ANNOT,
                    fontweight="bold",
                    color="black",
                    rotation=90,
                    clip_on=False,
                )

    annotate_above_limit(x_cg, cg_vals, "CG")
    annotate_above_limit(x_grace, grace_vals, "GraCE")

    fig.subplots_adjust(bottom=0.32, left=0.17, right=0.99, top=0.98)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_tp_values(text: str) -> List[int]:
    vals = []
    for part in re.split(r"[,\s]+", text.strip()):
        if part:
            vals.append(int(part))
    if not vals:
        raise ValueError("No TP values specified")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--tp-values", default="1 2 4")
    parser.add_argument("--save-svg", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    results_root = repo_root / "results" / "raw" / "tp"
    run_id = resolve_run_id(results_root, args.run_id)
    run_dir = results_root / run_id
    tp_values = parse_tp_values(args.tp_values)

    if not run_dir.exists():
        raise FileNotFoundError(f"TP run directory not found: {run_dir}")

    processed_dir = repo_root / "results" / "processed" / "tp" / run_id
    figure_run_dir = repo_root / "figures" / run_id
    figures_dir = repo_root / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figure_run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    info(f"Reading TP results from: {run_dir}")
    df_summary = build_summary_df(run_dir, tp_values)
    df_long = build_long_df(df_summary, tp_values)

    wide_csv = processed_dir / "tp_summary_wide.csv"
    long_csv = processed_dir / "tp_summary_long.csv"

    df_summary.to_csv(wide_csv, index=False)
    df_long.to_csv(long_csv, index=False)
    ok(f"Wrote CSV summaries: {wide_csv}, {long_csv}")

    out_pdf = figure_run_dir / "figure11_tp_scaling.pdf"
    out_svg = figure_run_dir / "figure11_tp_scaling.svg" if args.save_svg else None
    make_tp_plot(df_long, out_pdf=out_pdf, out_svg=out_svg)

    # Stable artifact-facing names.
    shutil.copyfile(out_pdf, figures_dir / "figure11.pdf")

    ok("TP summary and plot generated")
    print(f"Run id:       {run_id}")
    print(f"Wide CSV:     {wide_csv}")
    print(f"Long CSV:     {long_csv}")
    print(f"Run figure:   {out_pdf}")
    print(f"Stable fig:   {figures_dir / 'figure11.pdf'}")


if __name__ == "__main__":
    main()
