#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_model_config
from src.utils.process_database import create_image_pair_list, remove_doppelgangers
from remove_doppelgangers import doppelgangers_classifier


DEFAULT_SCENES = [
    "siteSTR0010",
    "siteSTR0023",
    "siteSTR0028",
    "siteSTR0042",
    "siteSTR0109",
]

SHARED_STAGE_NAMES = [
    "feature_extraction",
    "matching_and_geometric_verification",
]
VANILLA_STAGE_NAMES = [
    "vanilla_mapper",
]
VISUAL_STAGE_NAMES = [
    "pair_list_creation",
    "visual_disambiguation",
    "filtered_database_creation",
    "visual_mapper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run shared COLMAP matching once per Visym scene, then benchmark "
            "vanilla COLMAP reconstruction and COLMAP + visual disambiguation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the XDG model config YAML.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the Doppelgangers checkpoint.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "visymscenes",
        help="Root directory containing Visym scene folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "visym_colmap_benchmark",
        help="Output root for databases, reconstructions, and timing logs.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=DEFAULT_SCENES,
        help="Scene names to run.",
    )
    parser.add_argument(
        "--colmap-exe-command",
        type=str,
        default="colmap",
        help="COLMAP executable command.",
    )
    parser.add_argument(
        "--matching-type",
        type=str,
        default="vocab_tree_matcher",
        choices=["vocab_tree_matcher", "exhaustive_matcher"],
        help="COLMAP matching stage to run once before both reconstructions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Visual disambiguation threshold.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional classifier batch size override.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Optional classifier image size override.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Optional classifier resize mode override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Classifier device, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional classifier dataloader worker override.",
    )
    return parser.parse_args()


def get_cfg_value(cfg: argparse.Namespace, *keys: str, default: Any = None) -> Any:
    value: Any = cfg
    for key in keys:
        if not hasattr(value, key):
            return default
        value = getattr(value, key)
    return value


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def command_to_string(command: list[str]) -> str:
    return shlex.join(command)


def run_command(command: list[str]) -> None:
    print(f"$ {command_to_string(command)}", flush=True)
    subprocess.run(command, check=True)


def ensure_vocab_tree() -> Path:
    weights_dir = REPO_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    vocab_tree_path = weights_dir / "vocab_tree_flickr100K_words1M.bin"
    if vocab_tree_path.exists():
        return vocab_tree_path

    run_command(
        [
            "wget",
            "https://demuc.de/colmap/vocab_tree_flickr100K_words1M.bin",
            "-P",
            str(weights_dir),
        ]
    )
    return vocab_tree_path


def normalize_scene_key(value: str) -> str:
    digits = "".join(re.findall(r"\d+", value))
    if digits:
        return digits
    return value.lower()


def resolve_scene_root(dataset_root: Path, scene: str) -> Path:
    direct = dataset_root / scene
    if direct.exists() and direct.is_dir():
        return direct

    target_key = normalize_scene_key(scene)
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if normalize_scene_key(child.name) == target_key:
            return child

    raise FileNotFoundError(
        f"Could not resolve scene folder for {scene} under {dataset_root}."
    )


def resolve_scene_image_dir(dataset_root: Path, scene: str) -> tuple[Path, Path]:
    scene_root = resolve_scene_root(dataset_root, scene)
    return scene_root, scene_root


def database_has_rows(database_path: Path, table: str) -> bool:
    if not database_path.exists():
        return False
    connection = sqlite3.connect(str(database_path))
    try:
        cursor = connection.execute(f"SELECT COUNT(*) FROM {table}")
        count = int(cursor.fetchone()[0])
    except sqlite3.Error:
        return False
    finally:
        connection.close()
    return count > 0


def reconstruction_exists(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def run_stage(
    stage_name: str,
    *,
    should_skip: bool,
    action: Callable[[], Any],
    stage_log: dict[str, dict[str, Any]],
) -> Any:
    if should_skip:
        print(f"[skip] {stage_name}", flush=True)
        stage_log[stage_name] = {
            "status": "skipped",
            "duration_s": 0.0,
        }
        return None

    print(f"[run] {stage_name}", flush=True)
    start = time.perf_counter()
    try:
        result = action()
    except Exception as exc:
        duration = time.perf_counter() - start
        stage_log[stage_name] = {
            "status": "failed",
            "duration_s": duration,
            "error": str(exc),
        }
        raise

    duration = time.perf_counter() - start
    stage_log[stage_name] = {
        "status": "completed",
        "duration_s": duration,
    }
    print(f"[done] {stage_name}: {format_seconds(duration)}", flush=True)
    return result


def run_feature_extractor(colmap_exe: str, image_dir: Path, database_path: Path) -> None:
    run_command(
        [
            colmap_exe,
            "feature_extractor",
            "--image_path",
            str(image_dir),
            "--database_path",
            str(database_path),
        ]
    )


def run_matcher(colmap_exe: str, matching_type: str, database_path: Path) -> None:
    command = [
        colmap_exe,
        matching_type,
        "--database_path",
        str(database_path),
    ]
    if matching_type == "vocab_tree_matcher":
        vocab_tree_path = ensure_vocab_tree()
        command.extend(
            ["--VocabTreeMatching.vocab_tree_path", str(vocab_tree_path)]
        )
    run_command(command)


def run_mapper(colmap_exe: str, database_path: Path, image_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            colmap_exe,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(output_dir),
        ]
    )


def build_classifier_args(
    *,
    cfg: argparse.Namespace,
    image_dir: Path,
    output_dir: Path,
    ckpt_path: Path,
    batch_size: int | None,
    img_size: int | None,
    mode: str | None,
    device: str | None,
    num_workers: int | None,
) -> argparse.Namespace:
    resolved_batch_size = batch_size
    if resolved_batch_size is None:
        resolved_batch_size = int(get_cfg_value(cfg, "inference", "batch_size", default=1) or 1)

    resolved_num_workers = num_workers
    if resolved_num_workers is None:
        resolved_num_workers = int(get_cfg_value(cfg, "inference", "num_workers", default=0) or 0)

    return argparse.Namespace(
        input_image_path=str(image_dir),
        output_path=str(output_dir),
        ckpt=str(ckpt_path),
        batch_size=resolved_batch_size,
        img_size=img_size,
        mode=mode,
        device=device,
        num_workers=resolved_num_workers,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_scene_text_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Scene: {summary['scene']}",
        f"Scene dir: {summary['scene_dir']}",
        f"Image dir: {summary['image_dir']}",
        "",
        "Stage times:",
    ]
    for stage_name, info in summary["stages"].items():
        duration = format_seconds(float(info["duration_s"]))
        lines.append(f"- {stage_name}: {duration} ({info['status']})")

    lines.extend(
        [
            "",
            "Totals:",
            f"- shared_total_s: {format_seconds(float(summary['shared_total_s']))}",
            f"- vanilla_total_s: {format_seconds(float(summary['vanilla_total_s']))}",
            f"- visual_total_s: {format_seconds(float(summary['visual_total_s']))}",
            f"- scene_wall_time_s: {format_seconds(float(summary['scene_wall_time_s']))}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_global_text_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Per-site totals:",
    ]
    for item in summary["scenes"]:
        lines.append(
            f"- {item['scene']}: shared={format_seconds(float(item['shared_total_s']))}, "
            f"vanilla={format_seconds(float(item['vanilla_total_s']))}, "
            f"visual={format_seconds(float(item['visual_total_s']))}, "
            f"wall={format_seconds(float(item['scene_wall_time_s']))}"
        )

    lines.extend(
        [
            "",
            "Overall sums across all requested sites:",
            f"- site_count: {summary['num_scenes']}",
            f"- shared_total_s: {format_seconds(float(summary['shared_total_s']))}",
            f"- vanilla_total_s: {format_seconds(float(summary['vanilla_total_s']))}",
            f"- visual_total_s: {format_seconds(float(summary['visual_total_s']))}",
            f"- benchmark_wall_time_s: {format_seconds(float(summary['benchmark_wall_time_s']))}",
            "",
            "Note: shared stages are included in both vanilla_total_s and visual_total_s.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def compute_total(stage_log: dict[str, dict[str, Any]], stage_names: list[str]) -> float:
    return float(sum(stage_log[name]["duration_s"] for name in stage_names))


def updated_database_path(database_path: Path, threshold: float) -> Path:
    return database_path.with_name(
        f"{database_path.stem}_threshold_{threshold:.3f}{database_path.suffix}"
    )


def run_scene(scene: str, args: argparse.Namespace, cfg: argparse.Namespace) -> dict[str, Any]:
    scene_start = time.perf_counter()
    scene_dir, image_dir = resolve_scene_image_dir(args.dataset_root, scene)
    print(f"Resolved site folder: {scene_dir}", flush=True)
    print(f"Resolved image directory: {image_dir}", flush=True)

    scene_output_root = args.output_root / scene
    shared_dir = scene_output_root / "shared"
    vanilla_dir = scene_output_root / "vanilla"
    visual_dir = scene_output_root / "visual_disambiguation"
    shared_dir.mkdir(parents=True, exist_ok=True)
    vanilla_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    database_path = shared_dir / "database.db"
    pair_path = shared_dir / "pairs_list.npy"
    probability_path = visual_dir / "pair_probability_list.npy"
    filtered_db_path = updated_database_path(database_path, args.threshold)
    vanilla_sparse_dir = vanilla_dir / "sparse"
    visual_sparse_dir = visual_dir / "sparse"

    stage_log: dict[str, dict[str, Any]] = {}

    run_stage(
        "feature_extraction",
        should_skip=database_has_rows(database_path, "images"),
        action=lambda: run_feature_extractor(args.colmap_exe_command, image_dir, database_path),
        stage_log=stage_log,
    )
    run_stage(
        "matching_and_geometric_verification",
        should_skip=database_has_rows(database_path, "two_view_geometries"),
        action=lambda: run_matcher(args.colmap_exe_command, args.matching_type, database_path),
        stage_log=stage_log,
    )
    run_stage(
        "vanilla_mapper",
        should_skip=reconstruction_exists(vanilla_sparse_dir),
        action=lambda: run_mapper(
            args.colmap_exe_command,
            database_path,
            image_dir,
            vanilla_sparse_dir,
        ),
        stage_log=stage_log,
    )
    run_stage(
        "pair_list_creation",
        should_skip=pair_path.exists(),
        action=lambda: create_image_pair_list(str(database_path), str(shared_dir)),
        stage_log=stage_log,
    )

    classifier_args = build_classifier_args(
        cfg=cfg,
        image_dir=image_dir,
        output_dir=visual_dir,
        ckpt_path=args.ckpt,
        batch_size=args.batch_size,
        img_size=args.img_size,
        mode=args.mode,
        device=args.device,
        num_workers=args.num_workers,
    )
    run_stage(
        "visual_disambiguation",
        should_skip=probability_path.exists(),
        action=lambda: doppelgangers_classifier(classifier_args, cfg, str(pair_path)),
        stage_log=stage_log,
    )
    run_stage(
        "filtered_database_creation",
        should_skip=filtered_db_path.exists(),
        action=lambda: remove_doppelgangers(
            str(database_path),
            str(probability_path),
            str(pair_path),
            args.threshold,
        ),
        stage_log=stage_log,
    )
    run_stage(
        "visual_mapper",
        should_skip=reconstruction_exists(visual_sparse_dir),
        action=lambda: run_mapper(
            args.colmap_exe_command,
            filtered_db_path,
            image_dir,
            visual_sparse_dir,
        ),
        stage_log=stage_log,
    )

    scene_wall_time_s = time.perf_counter() - scene_start
    summary = {
        "scene": scene,
        "scene_dir": str(scene_dir),
        "image_dir": str(image_dir),
        "database_path": str(database_path),
        "filtered_database_path": str(filtered_db_path),
        "stages": stage_log,
        "shared_total_s": compute_total(stage_log, SHARED_STAGE_NAMES),
        "vanilla_total_s": compute_total(stage_log, SHARED_STAGE_NAMES + VANILLA_STAGE_NAMES),
        "visual_total_s": compute_total(stage_log, SHARED_STAGE_NAMES + VISUAL_STAGE_NAMES),
        "scene_wall_time_s": scene_wall_time_s,
    }

    write_json(scene_output_root / "timing_summary.json", summary)
    write_text(scene_output_root / "timing_summary.txt", build_scene_text_summary(summary))
    return summary


def build_global_summary(scene_summaries: list[dict[str, Any]], benchmark_wall_time_s: float) -> dict[str, Any]:
    return {
        "num_scenes": len(scene_summaries),
        "scenes": scene_summaries,
        "shared_total_s": float(sum(item["shared_total_s"] for item in scene_summaries)),
        "vanilla_total_s": float(sum(item["vanilla_total_s"] for item in scene_summaries)),
        "visual_total_s": float(sum(item["visual_total_s"] for item in scene_summaries)),
        "benchmark_wall_time_s": benchmark_wall_time_s,
    }


def main() -> None:
    args = parse_args()
    cfg = load_model_config(args.config)
    args.dataset_root = args.dataset_root.resolve()
    args.output_root = args.output_root.resolve()
    args.ckpt = args.ckpt.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    benchmark_start = time.perf_counter()
    scene_summaries = []
    for scene in args.scenes:
        print(f"\n=== {scene} ===", flush=True)
        summary = run_scene(scene, args, cfg)
        scene_summaries.append(summary)

    benchmark_wall_time_s = time.perf_counter() - benchmark_start
    global_summary = build_global_summary(scene_summaries, benchmark_wall_time_s)
    write_json(args.output_root / "timing_summary.json", global_summary)
    text_summary = build_global_text_summary(global_summary)
    write_text(args.output_root / "timing_summary.txt", text_summary)
    print("\n" + text_summary, flush=True)


if __name__ == "__main__":
    main()
