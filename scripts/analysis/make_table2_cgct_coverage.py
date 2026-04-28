#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


CUDA_LAUNCH_APIS = {
    "cudaLaunchKernel",
    "cuLaunchKernel",
    "cudaLaunchKernelExC_v11060",
    "cuLaunchKernelEx",
}

CUDA_GRAPH_LAUNCH_APIS = {
    "cudaGraphLaunch_v10000",
    "cudaGraphLaunch",
}


@dataclass(frozen=True)
class Workload:
    acronym: str
    model: str
    suite: str
    mode: str
    batch_size: str


@dataclass(frozen=True)
class Counts:
    cuda_launch: int
    graph_launch: int
    total: int
    trace_file: str

    @property
    def graph_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.graph_launch / self.total


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


def normalize_suite(suite: str) -> str:
    s = suite.strip()
    if s in {"Huggingface", "huggingface"}:
        return "huggingface"
    if s in {"TorchBench", "torchbench"}:
        return "torchbench"
    if s in {"TIMM", "timm", "timm_models"}:
        return "timm_models"
    return s


def parse_workloads(path: Path) -> List[Workload]:
    workloads: List[Workload] = []

    with path.open("r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) >= 5:
                acronym, model, suite, mode, batch_size = fields[:5]
            elif len(fields) == 4:
                model, suite, mode, batch_size = fields
                acronym = model
            else:
                raise ValueError(f"Malformed workload line {lineno} in {path}: {line}")

            workloads.append(
                Workload(
                    acronym=acronym,
                    model=model,
                    suite=normalize_suite(suite),
                    mode=mode.lower(),
                    batch_size=str(batch_size),
                )
            )

    if not workloads:
        raise RuntimeError(f"No workloads parsed from: {path}")

    return workloads


def read_nsys_stats_csv(path: Path) -> pd.DataFrame:
    """
    nsys stats normally writes a plain CSV with columns including:
      API Function, Kernel Name

    This helper also handles files with a short textual preamble by scanning for
    the header row containing API Function.
    """
    header_idx = 0
    with path.open("r", errors="replace") as f:
        for i, line in enumerate(f):
            if "API Function" in line and "Kernel Name" in line:
                header_idx = i
                break

    df = pd.read_csv(path, skiprows=header_idx)
    df.columns = [str(c).strip() for c in df.columns]

    if "API Function" not in df.columns:
        raise RuntimeError(f"'API Function' column not found in {path}; columns={list(df.columns)}")

    return df


def count_launches_from_trace(path: Path, use_second_half: bool = True) -> Counts:
    df = read_nsys_stats_csv(path)

    if use_second_half and len(df) > 1:
        half = len(df) // 2
        df = df.tail(half)

    api = df["API Function"].fillna("").astype(str).str.strip()

    cuda_launch = int(api.isin(CUDA_LAUNCH_APIS).sum())
    graph_launch = int(api.isin(CUDA_GRAPH_LAUNCH_APIS).sum())
    total = cuda_launch + graph_launch

    return Counts(
        cuda_launch=cuda_launch,
        graph_launch=graph_launch,
        total=total,
        trace_file=str(path),
    )


def find_trace_file(case_dir: Path, workload: Workload) -> Optional[Path]:
    trace_dir = case_dir / "cuda_kern_exec_trace"
    if not trace_dir.exists():
        return None

    patterns = [
        f"{workload.acronym}_*_cuda_kern_exec_trace.csv",
        f"{workload.acronym}*cuda_kern_exec_trace.csv",
        f"{workload.model}_*_cuda_kern_exec_trace.csv",
        f"{workload.model}*cuda_kern_exec_trace.csv",
    ]

    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(str(trace_dir / pattern)))

    unique = sorted(set(matches))
    if not unique:
        return None

    if len(unique) > 1:
        # Prefer exact prefix match with acronym.
        exact = [p for p in unique if p.name.startswith(f"{workload.acronym}_")]
        if exact:
            return exact[0]

    return unique[0]


def get_counts(case_dir: Path, workload: Workload) -> Counts:
    trace_file = find_trace_file(case_dir, workload)
    if trace_file is None:
        return Counts(cuda_launch=0, graph_launch=0, total=0, trace_file="MISSING")

    return count_launches_from_trace(trace_file)


def fmt_pct(x: float) -> str:
    return f"{x:.2f}"


def simple_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    widths = {
        c: max(len(str(c)), *(len(str(v)) for v in df[c].tolist()))
        for c in cols
    }

    def row(vals: List[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[c]) for v, c in zip(vals, cols)) + " |"

    out = []
    out.append(row(cols))
    out.append("| " + " | ".join("-" * widths[c] for c in cols) + " |")
    for _, r in df.iterrows():
        out.append(row([str(r[c]) for c in cols]))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--workloads", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    results_root = repo_root / "results" / "raw" / "table2_cgct_coverage"
    run_id = resolve_run_id(results_root, args.run_id)
    run_dir = results_root / run_id

    if args.workloads is None:
        workloads_file = repo_root / "configs" / "workloads_table2_cgct_coverage.txt"
    else:
        workloads_file = args.workloads.resolve()

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not workloads_file.exists():
        raise FileNotFoundError(f"Workload file not found: {workloads_file}")

    workloads = parse_workloads(workloads_file)

    cases = {
        "vanilla_no_cg": run_dir / "vanilla" / "no-cg",
        "vanilla_cg": run_dir / "vanilla" / "cg",
        "cgct_cg": run_dir / "cgct" / "cg",
    }

    for name, path in cases.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing case directory for {name}: {path}")

    rows: List[Dict[str, object]] = []

    for w in workloads:
        vanilla_no_cg = get_counts(cases["vanilla_no_cg"], w)
        vanilla_cg = get_counts(cases["vanilla_cg"], w)
        cgct_cg = get_counts(cases["cgct_cg"], w)

        rows.append(
            {
                "acronym": w.acronym,
                "model": w.model,
                "suite": w.suite,
                "mode": w.mode,
                "batch_size": w.batch_size,

                "vanilla_no_cg_cuda_launch": vanilla_no_cg.cuda_launch,
                "vanilla_no_cg_graph_launch": vanilla_no_cg.graph_launch,
                "vanilla_no_cg_total": vanilla_no_cg.total,

                "vanilla_cg_cuda_launch": vanilla_cg.cuda_launch,
                "vanilla_cg_graph_launch": vanilla_cg.graph_launch,
                "vanilla_cg_total": vanilla_cg.total,
                "vanilla_cg_graph_pct": vanilla_cg.graph_pct,

                "cgct_cg_cuda_launch": cgct_cg.cuda_launch,
                "cgct_cg_graph_launch": cgct_cg.graph_launch,
                "cgct_cg_total": cgct_cg.total,
                "cgct_cg_graph_pct": cgct_cg.graph_pct,

                "cgct_minus_vanilla_pct_points": cgct_cg.graph_pct - vanilla_cg.graph_pct,

                "vanilla_no_cg_trace": vanilla_no_cg.trace_file,
                "vanilla_cg_trace": vanilla_cg.trace_file,
                "cgct_cg_trace": cgct_cg.trace_file,
            }
        )

    detailed_df = pd.DataFrame(rows)

    paper_df = detailed_df[
        [
            "acronym",
            "vanilla_cg_graph_pct",
            "cgct_cg_graph_pct",
        ]
    ].copy()

    paper_df = paper_df.rename(
        columns={
            "acronym": "App.",
            "vanilla_cg_graph_pct": "% of kernels in CUDA Graphs in PyTorch2-CG",
            "cgct_cg_graph_pct": "% of kernels in CUDA Graphs in GraCE-CGCT",
        }
    )

    for c in [
        "% of kernels in CUDA Graphs in PyTorch2-CG",
        "% of kernels in CUDA Graphs in GraCE-CGCT",
    ]:
        paper_df[c] = paper_df[c].map(lambda x: round(float(x), 2))

    processed_dir = repo_root / "results" / "processed" / "table2_cgct_coverage" / run_id
    tables_dir = repo_root / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    detailed_csv = processed_dir / "table2_cgct_coverage_detailed.csv"
    paper_csv = processed_dir / "table2_cgct_coverage.csv"
    pretty_txt = processed_dir / "table2_cgct_coverage_pretty.txt"
    markdown_file = processed_dir / "table2_cgct_coverage.md"

    stable_csv = tables_dir / "table2.csv"
    stable_txt = tables_dir / "table2.txt"
    stable_md = tables_dir / "table2.md"

    detailed_df.to_csv(detailed_csv, index=False)
    paper_df.to_csv(paper_csv, index=False)
    paper_df.to_csv(stable_csv, index=False)

    try:
        pretty = paper_df.to_markdown(index=False)
    except Exception:
        pretty = simple_markdown_table(paper_df)

    pretty_txt.write_text(paper_df.to_string(index=False) + "\n")
    markdown_file.write_text(pretty + "\n")

    stable_txt.write_text(paper_df.to_string(index=False) + "\n")
    stable_md.write_text(pretty + "\n")

    print()
    print("[OK] Table 2 generated")
    print(f"Run id:       {run_id}")
    print(f"Detailed CSV: {detailed_csv}")
    print(f"Paper CSV:    {paper_csv}")
    print(f"Pretty TXT:   {pretty_txt}")
    print(f"Markdown:     {markdown_file}")
    print(f"Stable CSV:   {stable_csv}")
    print(f"Stable TXT:   {stable_txt}")
    print()
    print(paper_df.to_string(index=False))


if __name__ == "__main__":
    main()