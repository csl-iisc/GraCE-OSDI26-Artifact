#!/usr/bin/env python3
"""Shared helpers for GraCE OSDI'26 figure generation."""

from __future__ import annotations

import ast
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


SUITE_CONFIGS = [
    # suite_key is used for acronym lookup and dataframe merge metadata.
    # suite_file is used in benchmark CSV filenames.
    {"suite_key": "torchbench", "suite_file": "torchbench", "mode": "inference", "dtype": "bfloat16"},
    {"suite_key": "torchbench", "suite_file": "torchbench", "mode": "training", "dtype": "amp"},
    {"suite_key": "huggingface", "suite_file": "huggingface", "mode": "inference", "dtype": "bfloat16"},
    {"suite_key": "huggingface", "suite_file": "huggingface", "mode": "training", "dtype": "amp"},
    {"suite_key": "timm", "suite_file": "timm_models", "mode": "inference", "dtype": "bfloat16"},
]


VARIANT_GRAPH_TO_DIR = {
    "vanilla_no_cg": ("vanilla", "no-cg"),
    "vanilla_cg": ("vanilla", "cg"),
    "cgct": ("cgct", "cg"),
    "cgct_pi": ("cgct-pi", "cg"),
    "full": ("full", "cg"),
}


def resolve_repo_root(repo_root: str | None) -> Path:
    if repo_root:
        return Path(repo_root).resolve()
    return Path.cwd().resolve()


def resolve_run_id(raw_root: Path, run_id: str) -> str:
    if run_id != "latest":
        return run_id
    latest_file = raw_root / "latest.txt"
    if not latest_file.exists():
        raise FileNotFoundError(
            f"Requested run_id=latest, but latest.txt does not exist: {latest_file}"
        )
    resolved = latest_file.read_text().strip()
    if not resolved:
        raise RuntimeError(f"latest.txt is empty: {latest_file}")
    return resolved


def default_acronym_file(repo_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = repo_root / path
        return path

    candidates = [
        repo_root / "configs" / "workloads_single_gpu_25_with_acronym.txt"
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_acronym_mapping(path: Path | None) -> Dict[Tuple[str, str, str, int], str]:
    mapping: Dict[Tuple[str, str, str, int], str] = {}
    if path is None:
        return mapping
    if not path.exists():
        raise FileNotFoundError(f"Acronym mapping file not found: {path}")

    with path.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"Malformed acronym line {line_no} in {path}: {line!r}")
            acronym, model, suite, mode, batch_size = parts[:5]
            suite = normalize_suite_for_acronym(suite)
            mapping[(model, suite, mode, int(batch_size))] = acronym
    return mapping


def normalize_suite_for_acronym(suite: str) -> str:
    if suite in {"TIMM", "timm_models"}:
        return "timm"
    if suite == "Huggingface":
        return "huggingface"
    if suite == "TorchBench":
        return "torchbench"
    return suite


def parse_raw_timings(value) -> np.ndarray:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.asarray([], dtype=float)
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)

    text = str(value).strip()
    if not text:
        return np.asarray([], dtype=float)

    try:
        parsed = ast.literal_eval(text)
        return np.asarray(parsed, dtype=float)
    except Exception:
        # Match the original notebook's simple split behavior, but be tolerant
        # to whitespace variation.
        text = text.strip("[]")
        if not text:
            return np.asarray([], dtype=float)
        return np.asarray([float(x.strip()) for x in text.split(",") if x.strip()], dtype=float)


def gmean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


def csv_name(graph_label: str, suite_file: str, dtype: str, mode: str) -> str:
    return f"inductor_{graph_label}_{suite_file}_{dtype}_{mode}_cuda_performance.csv"


def load_one_csv(run_dir: Path, variant: str, graph_mode: str, graph_label: str,
                 suite_key: str, suite_file: str, mode: str, dtype: str,
                 output_column: str) -> pd.DataFrame:
    csv_path = run_dir / variant / graph_mode / csv_name(graph_label, suite_file, dtype, mode)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required CSV: {csv_path}")

    data = pd.read_csv(csv_path).drop_duplicates(subset=["name", "batch_size"], keep="first")
    data["suite"] = suite_key
    data["mode"] = mode
    data = data.rename(columns={"raw_timings": output_column})

    keep = ["name", "batch_size", "suite", "mode", output_column]
    if "dev" in data.columns:
        keep.insert(0, "dev")
    return data[keep]


def merge_three_way(run_dir: Path) -> pd.DataFrame:
    """Merge vanilla/no-cg, vanilla/cg, and full/cg, matching Figure 9."""
    merged_frames: List[pd.DataFrame] = []
    for cfg in SUITE_CONFIGS:
        suite_key = cfg["suite_key"]
        suite_file = cfg["suite_file"]
        mode = cfg["mode"]
        dtype = cfg["dtype"]

        no_cg = load_one_csv(
            run_dir, "vanilla", "no-cg", "no_cudagraphs",
            suite_key, suite_file, mode, dtype, "no_cudagraph_raw_timings",
        )
        vanilla_cg = load_one_csv(
            run_dir, "vanilla", "cg", "with_cudagraphs",
            suite_key, suite_file, mode, dtype, "cudagraph_unpatched_raw_timings",
        )
        full = load_one_csv(
            run_dir, "full", "cg", "with_cudagraphs",
            suite_key, suite_file, mode, dtype, "cudagraph_patched_raw_timings",
        )

        keys = ["name", "batch_size", "suite", "mode"]
        if "dev" in no_cg.columns and "dev" in vanilla_cg.columns and "dev" in full.columns:
            keys.insert(0, "dev")

        merged = vanilla_cg.merge(no_cg, on=keys, how="inner").merge(full, on=keys, how="inner")
        merged_frames.append(merged)

    if not merged_frames:
        raise RuntimeError("No data merged for Figure 9")
    out = pd.concat(merged_frames, ignore_index=True)
    for col in [
        "no_cudagraph_raw_timings",
        "cudagraph_unpatched_raw_timings",
        "cudagraph_patched_raw_timings",
    ]:
        out[col] = out[col].apply(parse_raw_timings)
    return out


def merge_ablation(run_dir: Path) -> pd.DataFrame:
    """Merge vanilla/no-cg, vanilla/cg, cgct, cgct-pi, and full, matching Figure 10."""
    merged_frames: List[pd.DataFrame] = []
    for cfg in SUITE_CONFIGS:
        suite_key = cfg["suite_key"]
        suite_file = cfg["suite_file"]
        mode = cfg["mode"]
        dtype = cfg["dtype"]

        no_cg = load_one_csv(
            run_dir, "vanilla", "no-cg", "no_cudagraphs",
            suite_key, suite_file, mode, dtype, "no_cudagraph_raw_timings",
        )
        vanilla_cg = load_one_csv(
            run_dir, "vanilla", "cg", "with_cudagraphs",
            suite_key, suite_file, mode, dtype, "cudagraph_unpatched_raw_timings",
        )
        cgct = load_one_csv(
            run_dir, "cgct", "cg", "with_cudagraphs",
            suite_key, suite_file, mode, dtype, "opt1_patched_raw_timings",
        )
        cgct_pi = load_one_csv(
            run_dir, "cgct-pi", "cg", "with_cudagraphs",
            suite_key, suite_file, mode, dtype, "opt1_opt2_patched_raw_timings",
        )
        full = load_one_csv(
            run_dir, "full", "cg", "with_cudagraphs",
            suite_key, suite_file, mode, dtype, "cudagraph_patched_raw_timings",
        )

        keys = ["name", "batch_size", "suite", "mode"]
        frames = [no_cg, vanilla_cg, cgct, cgct_pi, full]
        if all("dev" in frame.columns for frame in frames):
            keys.insert(0, "dev")

        merged = (
            vanilla_cg
            .merge(no_cg, on=keys, how="inner")
            .merge(full, on=keys, how="inner")
            .merge(cgct, on=keys, how="inner")
            .merge(cgct_pi, on=keys, how="inner")
        )
        merged_frames.append(merged)

    if not merged_frames:
        raise RuntimeError("No data merged for Figure 10")
    out = pd.concat(merged_frames, ignore_index=True)
    for col in [
        "no_cudagraph_raw_timings",
        "cudagraph_unpatched_raw_timings",
        "cudagraph_patched_raw_timings",
        "opt1_patched_raw_timings",
        "opt1_opt2_patched_raw_timings",
    ]:
        out[col] = out[col].apply(parse_raw_timings)
    return out


def select_top_n(speedup_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    # Exact notebook logic: sort by full GraCE vs PyTorch2-CG improvement and
    # keep first unique (name, suite, mode) tuple.
    sorted_speedup_df = speedup_df.sort_values(by="speedup_patched_unpatched", ascending=False)
    displayed_models = {}
    top_selected_data = []
    for _, row in sorted_speedup_df.iterrows():
        model_key = (row["name"], row["suite"], row["mode"])
        if model_key not in displayed_models:
            top_selected_data.append(row)
            displayed_models[model_key] = True
        if len(displayed_models) == top_n:
            break
    return pd.DataFrame(top_selected_data)


def model_labels(df: pd.DataFrame, acronym_mapping: Dict[Tuple[str, str, str, int], str]) -> List[str]:
    labels = []
    for name, suite, mode, bs in zip(df["name"], df["suite"], df["mode"], df["batch_size"]):
        if name == "Geomean":
            labels.append("Geomean")
            continue
        labels.append(acronym_mapping.get((name, suite, mode, int(bs)), f"{name}_bs_{bs}"))
    return labels


def ensure_output_dirs(repo_root: Path, figure_name: str, run_id: str) -> Tuple[Path, Path, Path]:
    figures_dir = repo_root / "figures"
    run_figures_dir = figures_dir / run_id
    processed_dir = repo_root / "results" / "processed" / figure_name / run_id
    run_figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, run_figures_dir, processed_dir


def copy_to_stable(src: Path, stable: Path) -> None:
    stable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, stable)
