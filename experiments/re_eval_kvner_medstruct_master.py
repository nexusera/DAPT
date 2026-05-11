#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelEvalSpec:
    name: str
    slug: str
    pred_file: str
    gt_file: str


def default_dapt_root() -> Path:
    remote_root = Path("/data/ocean/DAPT")
    if remote_root.is_dir():
        return remote_root
    return Path(__file__).resolve().parents[1]


def build_specs(dapt_root: Path) -> list[ModelEvalSpec]:
    runs = dapt_root / "runs"
    return [
        ModelEvalSpec(
            name="Main Full",
            slug="main_full",
            pred_file=str(runs / "macbert_eval_aligned_preds.jsonl"),
            gt_file=str(runs / "macbert_eval_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="Ablation No NSP",
            slug="ablation_no_nsp",
            pred_file=str(runs / "no_nsp_eval_aligned_preds.jsonl"),
            gt_file=str(runs / "no_nsp_eval_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="Ablation No MLM",
            slug="ablation_no_mlm",
            pred_file=str(runs / "no_mlm_eval_aligned_preds.jsonl"),
            gt_file=str(runs / "no_mlm_eval_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="No Noise Baseline",
            slug="no_noise_baseline",
            pred_file=str(runs / "no_noise_eval_aligned_preds.jsonl"),
            gt_file=str(runs / "no_noise_eval_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="MLM Ablation: kvmlm",
            slug="mlm_kvmlm",
            pred_file=str(runs / "mlm_kvmlm_task13_aligned_preds.jsonl"),
            gt_file=str(runs / "mlm_kvmlm_task13_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="MLM Ablation: plainmlm",
            slug="mlm_plainmlm",
            pred_file=str(runs / "mlm_plainmlm_task13_aligned_preds.jsonl"),
            gt_file=str(runs / "mlm_plainmlm_task13_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="Noise Ablation: bucket",
            slug="noise_bucket",
            pred_file=str(runs / "noise_bucket_task13_aligned_preds.jsonl"),
            gt_file=str(runs / "noise_bucket_task13_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="Noise Ablation: linear",
            slug="noise_linear",
            pred_file=str(runs / "noise_linear_task13_aligned_preds.jsonl"),
            gt_file=str(runs / "noise_linear_task13_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="Noise Ablation: mlp",
            slug="noise_mlp",
            pred_file=str(runs / "noise_mlp_task13_aligned_preds.jsonl"),
            gt_file=str(runs / "noise_mlp_task13_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="NSP Ratio: 1:1",
            slug="nsp_ratio_1_1",
            pred_file=str(runs / "nsp_ratio_1_1_task13_aligned_preds.jsonl"),
            gt_file=str(runs / "nsp_ratio_1_1_task13_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="NSP Ratio: 3:1",
            slug="nsp_ratio_3_1",
            pred_file=str(runs / "nsp_ratio_3_1_task13_aligned_preds.jsonl"),
            gt_file=str(runs / "nsp_ratio_3_1_task13_aligned_gt.jsonl"),
        ),
        ModelEvalSpec(
            name="NSP Ratio: 1:3",
            slug="nsp_ratio_1_3",
            pred_file=str(runs / "nsp_ratio_1_3_task13_aligned_preds.jsonl"),
            gt_file=str(runs / "nsp_ratio_1_3_task13_aligned_gt.jsonl"),
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch re-evaluate KV-NER Task1/3 results with MedStruct-S-master scorer."
    )
    parser.add_argument("--dapt_root", type=Path, default=default_dapt_root())
    parser.add_argument(
        "--scorer_dir",
        type=Path,
        default=None,
        help="Directory containing MedStruct-S-master/scorer.py",
    )
    parser.add_argument(
        "--query_set",
        type=Path,
        default=None,
        help="Query set JSON for MedStruct-S-master scorer",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to store new MedStruct-S-master reports",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma separated slugs to run, or 'all'",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a task if its new output file already exists",
    )
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=0.0,
        help="Passed to MedStruct-S-master scorer",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Passed to MedStruct-S-master scorer",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable text normalization in scorer",
    )
    parser.add_argument(
        "--disable_tau",
        action="store_true",
        help="Disable dynamic tau in scorer",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate inputs and print planned commands",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")


def select_specs(specs: list[ModelEvalSpec], models_arg: str) -> list[ModelEvalSpec]:
    if models_arg.strip().lower() == "all":
        return specs
    wanted = {item.strip() for item in models_arg.split(",") if item.strip()}
    selected = [spec for spec in specs if spec.slug in wanted]
    missing = sorted(wanted - {spec.slug for spec in selected})
    if missing:
        raise ValueError(f"Unknown model slug(s): {', '.join(missing)}")
    return selected


def build_command(
    python_exe: str,
    scorer_path: Path,
    spec: ModelEvalSpec,
    task: str,
    query_set: Path,
    output_file: Path,
    args: argparse.Namespace,
) -> list[str]:
    cmd = [
        python_exe,
        str(scorer_path),
        "--pred_file",
        spec.pred_file,
        "--gt_file",
        spec.gt_file,
        "--query_set",
        str(query_set),
        "--task_type",
        task,
        "--output_file",
        str(output_file),
        "--model_name",
        spec.name,
        "--dataset_type",
        "MedStruct-S",
        "--overlap_threshold",
        str(args.overlap_threshold),
        "--similarity_threshold",
        str(args.similarity_threshold),
    ]
    if args.no_normalize:
        cmd.append("--no_normalize")
    if args.disable_tau:
        cmd.append("--disable_tau")
    return cmd


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_task1_metrics(report: dict) -> dict[str, float | None]:
    task = report.get("tasks", {}).get("task1", {})
    metrics = task.get("metrics", {})
    exact = metrics.get("exact", {})
    approx = metrics.get("approx", {})
    return {
        "task1_exact_f1": exact.get("f1"),
        "task1_approx_f1": approx.get("f1"),
    }


def extract_task3_metrics(report: dict) -> dict[str, float | None]:
    task = report.get("tasks", {}).get("task3", {})
    metrics = task.get("metrics", {})
    ee = metrics.get("exact_exact", {})
    ea = metrics.get("exact_approximate", {})
    aa = metrics.get("approximate_approximate", {})
    return {
        "task3_ee_f1": ee.get("f1"),
        "task3_ea_f1": ea.get("f1"),
        "task3_aa_f1": aa.get("f1"),
    }


def format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def write_summary_files(
    output_dir: Path,
    rows: list[dict],
    args: argparse.Namespace,
    selected_specs: Iterable[ModelEvalSpec],
) -> None:
    summary_json = {
        "summary": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dapt_root": str(args.dapt_root),
            "scorer_dir": str(
                args.scorer_dir
                or (args.dapt_root / "dapt_eval_package" / "MedStruct-S-master")
            ),
            "query_set": str(
                args.query_set
                or (
                    args.dapt_root
                    / "dapt_eval_package"
                    / "MedStruct-S-Benchmark-feature-configurable-metrics"
                    / "keys_merged_1027_cleaned.json"
                )
            ),
            "models": [spec.slug for spec in selected_specs],
            "overlap_threshold": args.overlap_threshold,
            "similarity_threshold": args.similarity_threshold,
            "normalize": not args.no_normalize,
            "tau_dynamic": not args.disable_tau,
        },
        "results": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# MedStruct-S-master KV-NER Re-eval Summary",
        "",
        f"- Generated at: {summary_json['summary']['generated_at']}",
        f"- DAPT root: `{summary_json['summary']['dapt_root']}`",
        f"- Scorer dir: `{summary_json['summary']['scorer_dir']}`",
        f"- Query set: `{summary_json['summary']['query_set']}`",
        f"- Models: `{', '.join(summary_json['summary']['models'])}`",
        f"- overlap_threshold: `{summary_json['summary']['overlap_threshold']}`",
        f"- similarity_threshold: `{summary_json['summary']['similarity_threshold']}`",
        f"- normalize: `{summary_json['summary']['normalize']}`",
        f"- tau_dynamic: `{summary_json['summary']['tau_dynamic']}`",
        "",
        "| Model | Task1 Exact F1 | Task1 Approx F1 | Task3 EE F1 | Task3 EA F1 | Task3 AA F1 | Task1 Report | Task3 Report |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {t1e} | {t1a} | {t3ee} | {t3ea} | {t3aa} | `{task1_report}` | `{task3_report}` |".format(
                model=row["model_name"],
                t1e=format_float(row.get("task1_exact_f1")),
                t1a=format_float(row.get("task1_approx_f1")),
                t3ee=format_float(row.get("task3_ee_f1")),
                t3ea=format_float(row.get("task3_ea_f1")),
                t3aa=format_float(row.get("task3_aa_f1")),
                task1_report=row["task1_report"],
                task3_report=row["task3_report"],
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.dapt_root = args.dapt_root.resolve()

    scorer_dir = (
        args.scorer_dir.resolve()
        if args.scorer_dir
        else (args.dapt_root / "dapt_eval_package" / "MedStruct-S-master").resolve()
    )
    scorer_path = scorer_dir / "scorer.py"
    query_set = (
        args.query_set.resolve()
        if args.query_set
        else (
            args.dapt_root
            / "dapt_eval_package"
            / "MedStruct-S-Benchmark-feature-configurable-metrics"
            / "keys_merged_1027_cleaned.json"
        ).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (args.dapt_root / "runs" / "medstruct_s_master_kvner_reports").resolve()
    )

    ensure_dir(args.dapt_root, "DAPT root")
    ensure_dir(scorer_dir, "scorer dir")
    ensure_file(scorer_path, "MedStruct-S-master scorer.py")
    ensure_file(query_set, "query_set")

    specs = select_specs(build_specs(args.dapt_root), args.models)
    for spec in specs:
        ensure_file(Path(spec.pred_file), f"pred_file for {spec.slug}")
        ensure_file(Path(spec.gt_file), f"gt_file for {spec.slug}")

    output_dir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable
    results_rows: list[dict] = []

    for spec in specs:
        print("=" * 72)
        print(f"[MODEL] {spec.name} ({spec.slug})")
        print(f"[PRED]  {spec.pred_file}")
        print(f"[GT]    {spec.gt_file}")
        model_out_dir = output_dir / spec.slug
        model_out_dir.mkdir(parents=True, exist_ok=True)

        task1_out = model_out_dir / "task1_report.json"
        task3_out = model_out_dir / "task3_report.json"

        task1_cmd = build_command(
            python_exe=python_exe,
            scorer_path=scorer_path,
            spec=spec,
            task="task1",
            query_set=query_set,
            output_file=task1_out,
            args=args,
        )
        task3_cmd = build_command(
            python_exe=python_exe,
            scorer_path=scorer_path,
            spec=spec,
            task="task3",
            query_set=query_set,
            output_file=task3_out,
            args=args,
        )

        if not (args.skip_existing and task1_out.is_file()):
            run_command(task1_cmd, cwd=scorer_dir, dry_run=args.dry_run)
        else:
            print(f"[SKIP] Existing Task1 report: {task1_out}")

        if not (args.skip_existing and task3_out.is_file()):
            run_command(task3_cmd, cwd=scorer_dir, dry_run=args.dry_run)
        else:
            print(f"[SKIP] Existing Task3 report: {task3_out}")

        row = {
            "model_name": spec.name,
            "model_slug": spec.slug,
            "pred_file": spec.pred_file,
            "gt_file": spec.gt_file,
            "task1_report": str(task1_out),
            "task3_report": str(task3_out),
        }
        if not args.dry_run:
            row.update(extract_task1_metrics(load_json(task1_out)))
            row.update(extract_task3_metrics(load_json(task3_out)))
            print(
                "[DONE] "
                f"T1 exact={format_float(row.get('task1_exact_f1'))}, "
                f"T1 approx={format_float(row.get('task1_approx_f1'))}, "
                f"T3 EE={format_float(row.get('task3_ee_f1'))}, "
                f"T3 EA={format_float(row.get('task3_ea_f1'))}, "
                f"T3 AA={format_float(row.get('task3_aa_f1'))}"
            )
        results_rows.append(row)

    if not args.dry_run:
        write_summary_files(output_dir, results_rows, args, specs)
        print("=" * 72)
        print(f"[OK] Summary JSON: {output_dir / 'summary.json'}")
        print(f"[OK] Summary MD:   {output_dir / 'summary.md'}")
    else:
        print("[DRY RUN] No reports were executed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
