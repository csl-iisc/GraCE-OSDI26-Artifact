#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


BANNER_RE = re.compile(r"^>{20,}\s*$")
END_BANNER_RE = re.compile(r"^<{20,}\s*$")
ACRONYM_RE = re.compile(r"\bAcronym:\s*(\S+)")
MODEL_RE = re.compile(r"\bModel:\s*(\S+)")
SUITE_RE = re.compile(r"\bSuite:\s*(\S+)")
MODE_RE = re.compile(r"\bMode:\s*(\S+)")
BATCH_RE = re.compile(r"\bBatch\s+[Ss]ize:\s*(\d+)")
DTYPE_RE = re.compile(r"\bDType:\s*(\S+)")
GRAPH_ID_RE = re.compile(r"For Function ID=\d+,\s*GraphID=GraphID\(id=(\d+)\)")
COPY_RE = re.compile(r"copied\s+(\d+)\s+(tensors|dataptrs)\s+with\s+sizes\s+\[([^\]]*)\]\s+bytes")


@dataclass(frozen=True)
class Workload:
    acronym: str
    model: str
    suite: str
    mode: str
    batch_size: str


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
    with path.open("r", errors="replace") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
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
        raise RuntimeError(f"No workloads parsed from {path}")
    return workloads


def empty_model_record() -> Dict[str, Any]:
    return {
        "acronym": None,
        "name": None,
        "suite": None,
        "mode": None,
        "dtype": None,
        "batch_size": None,
        "graphs": defaultdict(
            lambda: {
                "tensor_bytes": 0,
                "dataptr_bytes": 0,
                "tensor_count": 0,
                "dataptr_count": 0,
                "copy_events": 0,
            }
        ),
    }


def to_plain_record(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    out["graphs"] = {str(k): dict(v) for k, v in record["graphs"].items()}
    return out


def parse_sizes(raw_sizes: str) -> List[int]:
    raw_sizes = raw_sizes.strip()
    if not raw_sizes:
        return []
    return [int(x.strip()) for x in raw_sizes.split(",") if x.strip()]


def process_trace(trace_file: Path, output_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Parse one GraCE debug run.txt file.

    Supports both old headers:
      Model: XLNetLMHeadModel
      Batch Size: 1

    and artifact headers:
      Acronym: XLNET-I
      Model: XLNetLMHeadModel
      Batch size: 1

    Supports both payload forms:
      copied N tensors with sizes [...] bytes
      copied N dataptrs with sizes [...] bytes
    """
    models: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def flush_current() -> None:
        nonlocal current
        if current is None:
            return
        # Only keep real workload blocks.
        if current.get("name") or current.get("acronym"):
            models.append(to_plain_record(current))
        current = None

    with trace_file.open("r", errors="replace") as f:
        for line in f:
            if BANNER_RE.match(line):
                flush_current()
                current = empty_model_record()
                continue

            if current is None:
                continue

            if END_BANNER_RE.match(line):
                continue

            if m := ACRONYM_RE.search(line):
                current["acronym"] = m.group(1)
            if m := MODEL_RE.search(line):
                current["name"] = m.group(1)
            if m := SUITE_RE.search(line):
                current["suite"] = normalize_suite(m.group(1))
            if m := MODE_RE.search(line):
                current["mode"] = m.group(1).lower()
            if m := DTYPE_RE.search(line):
                current["dtype"] = m.group(1)
            if m := BATCH_RE.search(line):
                current["batch_size"] = int(m.group(1))

            graph_match = GRAPH_ID_RE.search(line)
            copy_match = COPY_RE.search(line)
            if graph_match and copy_match:
                graph_id = int(graph_match.group(1))
                count = int(copy_match.group(1))
                kind = copy_match.group(2)
                sizes = parse_sizes(copy_match.group(3))

                if count != len(sizes):
                    # Keep parsing. This warning is represented in the JSON so
                    # the artifact does not silently discard partially odd logs.
                    graph = current["graphs"][graph_id]
                    graph.setdefault("warnings", []).append(
                        f"declared count {count} but parsed {len(sizes)} sizes"
                    )

                graph = current["graphs"][graph_id]
                graph["copy_events"] += 1
                if kind == "tensors":
                    graph["tensor_count"] += count
                    graph["tensor_bytes"] += sum(sizes)
                elif kind == "dataptrs":
                    graph["dataptr_count"] += count
                    graph["dataptr_bytes"] += sum(sizes)

    flush_current()

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(models, indent=2) + "\n")

    return models


def model_total_bytes(model: Dict[str, Any]) -> int:
    total = 0
    for graph in model.get("graphs", {}).values():
        total += int(graph.get("tensor_bytes", 0))
        total += int(graph.get("dataptr_bytes", 0))
    return total


def model_total_copy_count(model: Dict[str, Any]) -> int:
    total = 0
    for graph in model.get("graphs", {}).values():
        total += int(graph.get("tensor_count", 0))
        total += int(graph.get("dataptr_count", 0))
    return total


def key_from_model(model: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(model.get("name")),
        str(model.get("mode", "")).lower(),
        str(model.get("batch_size")),
    )


def index_models(models: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for model in models:
        key = key_from_model(model)
        if key in out:
            # Preserve the first occurrence. Repeated benchmark retries would
            # otherwise double count if headers repeat.
            continue
        out[key] = model
    return out


def human_bytes(num_bytes: int) -> str:
    val = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if val < 1024.0:
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} PB"


def reduction_percent(direct: int, indirect: int) -> float:
    if direct <= 0:
        return 0.0
    return 100.0 * (direct - indirect) / direct


def floor2(x: float) -> float:
    return math.floor(x * 100.0) / 100.0


def resolve_run_id(results_root: Path, run_id: str) -> str:
    if run_id != "latest":
        return run_id
    latest = results_root / "latest.txt"
    if not latest.exists():
        raise FileNotFoundError(f"latest.txt not found: {latest}")
    value = latest.read_text().strip()
    if not value:
        raise RuntimeError(f"latest.txt is empty: {latest}")
    return value


def simple_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    widths = {c: max(len(str(c)), *(len(str(v)) for v in df[c].tolist())) for c in cols}

    def row(values: Iterable[Any]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[c]) for v, c in zip(values, cols)) + " |"

    lines = [row(cols), "| " + " | ".join("-" * widths[c] for c in cols) + " |"]
    for _, r in df.iterrows():
        lines.append(row([r[c] for c in cols]))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--workloads", type=Path, default=None)
    parser.add_argument("--direct-log", type=Path, default=None)
    parser.add_argument("--indirect-log", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    results_root = repo_root / "results" / "raw" / "table3_pi_copy_debug"
    run_id = resolve_run_id(results_root, args.run_id)
    run_dir = results_root / run_id

    workloads_file = args.workloads or (repo_root / "configs" / "workloads_table3_pi_copy_debug.txt")
    workloads_file = workloads_file.resolve()
    if not workloads_file.exists():
        raise FileNotFoundError(f"Workload file not found: {workloads_file}")

    direct_log = args.direct_log or (run_dir / "cgct-copy-debug" / "cg" / "run.txt")
    indirect_log = args.indirect_log or (run_dir / "cgct-pi-copy-debug" / "cg" / "run.txt")

    if not direct_log.exists():
        raise FileNotFoundError(f"Direct/cgct-copy-debug log not found: {direct_log}")
    if not indirect_log.exists():
        raise FileNotFoundError(f"Indirect/cgct-pi-copy-debug log not found: {indirect_log}")

    processed_dir = repo_root / "results" / "processed" / "table3_pi_copy_debug" / run_id
    tables_dir = repo_root / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    direct_json = processed_dir / "direct_trace.json"
    indirect_json = processed_dir / "indirect_trace.json"

    direct_models = process_trace(direct_log, direct_json)
    indirect_models = process_trace(indirect_log, indirect_json)

    direct_idx = index_models(direct_models)
    indirect_idx = index_models(indirect_models)
    workloads = parse_workloads(workloads_file)

    rows: List[Dict[str, Any]] = []
    detailed_rows: List[Dict[str, Any]] = []

    for w in workloads:
        key = (w.model, w.mode, str(w.batch_size))
        direct = direct_idx.get(key)
        indirect = indirect_idx.get(key)

        direct_bytes = model_total_bytes(direct) if direct else 0
        indirect_bytes = model_total_bytes(indirect) if indirect else 0
        direct_count = model_total_copy_count(direct) if direct else 0
        indirect_count = model_total_copy_count(indirect) if indirect else 0
        red = reduction_percent(direct_bytes, indirect_bytes)

        before_human = human_bytes(direct_bytes)
        after_human = human_bytes(indirect_bytes)

        rows.append(
            {
                "App.": w.acronym,
                "Data copied in GraCE-CGCT": before_human,
                "Data copied in GraCE-CGCT+PI": after_human,
            }
        )

        detailed_rows.append(
            {
                "acronym": w.acronym,
                "model": w.model,
                "suite": w.suite,
                "mode": w.mode,
                "batch_size": w.batch_size,
                "cgct_copy_debug_bytes": direct_bytes,
                "cgct_pi_copy_debug_bytes": indirect_bytes,
                "cgct_copy_debug_human": before_human,
                "cgct_pi_copy_debug_human": after_human,
                "cgct_copy_debug_copy_count": direct_count,
                "cgct_pi_copy_debug_copy_count": indirect_count,
                "reduction_percent": red,
                "direct_log_present": direct is not None,
                "indirect_log_present": indirect is not None,
            }
        )

    paper_df = pd.DataFrame(rows)
    detailed_df = pd.DataFrame(detailed_rows)

    paper_csv = processed_dir / "table3_pi_copy_debug.csv"
    detailed_csv = processed_dir / "table3_pi_copy_debug_detailed.csv"
    pretty_txt = processed_dir / "table3_pi_copy_debug_pretty.txt"
    markdown_file = processed_dir / "table3_pi_copy_debug.md"

    stable_csv = tables_dir / "table3.csv"
    stable_txt = tables_dir / "table3.txt"
    stable_md = tables_dir / "table3.md"
    arrow_txt = processed_dir / "table3_pi_copy_debug_arrow_format.txt"
    stable_arrow_txt = tables_dir / "table3_arrow_format.txt"

    paper_df.to_csv(paper_csv, index=False)
    detailed_df.to_csv(detailed_csv, index=False)
    paper_df.to_csv(stable_csv, index=False)

    try:
        md = paper_df.to_markdown(index=False)
    except Exception:
        md = simple_markdown_table(paper_df)

    pretty = paper_df.to_string(index=False) + "\n"
    arrow_lines = [
        f"{r['App.']}: {r['Data copied in GraCE-CGCT']} -> {r['Data copied in GraCE-CGCT+PI']}"
        for _, r in paper_df.iterrows()
    ]
    arrow_text = "\n".join(arrow_lines) + "\n"

    pretty_txt.write_text(pretty)
    markdown_file.write_text(md + "\n")
    arrow_txt.write_text(arrow_text)

    stable_txt.write_text(pretty)
    stable_md.write_text(md + "\n")
    stable_arrow_txt.write_text(arrow_text)

    print()
    print("[OK] Table 3 generated")
    print(f"Run id:        {run_id}")
    print(f"Direct JSON:   {direct_json}")
    print(f"Indirect JSON: {indirect_json}")
    print(f"Paper CSV:     {paper_csv}")
    print(f"Detailed CSV:  {detailed_csv}")
    print(f"Stable CSV:    {stable_csv}")
    print(f"Arrow TXT:     {arrow_txt}")
    print(f"Stable Arrow:  {stable_arrow_txt}")
    print()
    print(arrow_text, end="")


if __name__ == "__main__":
    main()
