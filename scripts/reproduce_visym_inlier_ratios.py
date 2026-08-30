#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "output" / "visym_colmap_benchmark"
VARIANT_SPARSE_ROOTS = {
    "vanilla": ("vanilla", "sparse"),
    "visual_disambiguation": ("visual_disambiguation", "sparse"),
    "dg++": ("visual_disambiguation_dg++", "sparse_dg++"),
}
VARIANT_VISUAL_DISAMBIGUATION_STAGES = {
    "visual_disambiguation": ("timing_summary.json", "visual_disambiguation"),
    "dg++": ("timing_summary_dg++.json", "visual_disambiguation_dg++"),
}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.colmap_utils import read_images_binary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the Doppelgangers++ geotag inlier-ratio evaluation for "
            "saved Visym COLMAP benchmark reconstructions."
        )
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root containing per-scene benchmark outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help=(
            "Root containing Visym scene folders. Defaults to image_dir/scene_dir "
            "from each scene timing_summary.json."
        ),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Scene names to include. Defaults to timing_summary.json scenes or all siteSTR* dirs.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=30.0,
        help="RANSAC inlier threshold in ECEF meters, matching the referenced script.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum RANSAC trials for similarity-transform model proposals.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for deterministic RANSAC.",
    )
    parser.add_argument(
        "--min-registered-images",
        type=int,
        default=50,
        help=(
            "Only include components with at least this many registered images "
            "in reported components and weighted IR."
        ),
    )
    parser.add_argument(
        "--top-components",
        type=int,
        default=3,
        help=(
            "Deprecated. Summaries now report all components sharing images with "
            "the largest vanilla COLMAP reconstruction."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["issue", "sim3"],
        default="sim3",
        help=(
            "issue uses a NumPy implementation of the linked issue's "
            "RANSACRegressor-style affine inlier selection; sim3 uses a "
            "similarity-transform RANSAC matching the paper metric."
        ),
    )
    parser.add_argument(
        "--altitude",
        choices=["zero", "metadata"],
        default="zero",
        help="Use zero altitude like the original evaluation, or metadata altitude from Visym JSON.",
    )
    parser.add_argument(
        "--write-priors",
        action="store_true",
        help="Also insert computed ECEF priors into each scene database image prior_t fields.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full results as JSON.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write one CSV row per sparse component.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_scenes(benchmark_root: Path, explicit_scenes: list[str] | None) -> list[str]:
    if explicit_scenes:
        return explicit_scenes

    summary_path = benchmark_root / "timing_summary.json"
    if summary_path.exists():
        summary = load_json(summary_path)
        scenes = [item["scene"] for item in summary.get("scenes", []) if "scene" in item]
        if scenes:
            return scenes

    return sorted(path.name for path in benchmark_root.glob("siteSTR*") if path.is_dir())


def scene_dataset_root(
    benchmark_root: Path,
    dataset_root: Path | None,
    scene: str,
) -> Path:
    if dataset_root is not None:
        return dataset_root / scene

    timing_path = benchmark_root / scene / "timing_summary.json"
    if timing_path.exists():
        timing = load_json(timing_path)
        for key in ("image_dir", "scene_dir"):
            value = timing.get(key)
            if value and Path(value).exists():
                return Path(value)

    raise FileNotFoundError(
        f"Could not resolve dataset root for {scene}. Pass --dataset-root explicitly."
    )


def lla_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f**2
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    n = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    x = (n + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (n + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (n * (1 - e2) + alt) * math.sin(lat_rad)
    return np.array([x, y, z], dtype=np.float64)


def metadata_for_image(scene_root: Path, image_name: str) -> Path:
    return scene_root / Path(image_name).with_suffix(".json")


def gps_from_metadata(path: Path, altitude_mode: str) -> tuple[float, float, float] | None:
    if not path.exists():
        return None

    metadata = load_json(path)
    pose = metadata.get("extrinsics") or metadata.get("vipy", {}).get("attributes", {}).get("pose")
    if not pose:
        return None

    lat = pose.get("lat")
    lon = pose.get("lon")
    if lat is None or lon is None:
        return None

    alt = pose.get("alt") if altitude_mode == "metadata" else 0.0
    return float(lat), float(lon), float(alt or 0.0)


def load_database_images(database_path: Path) -> dict[int, str]:
    try:
        import sqlite3

        with sqlite3.connect(str(database_path)) as connection:
            return {
                int(image_id): name
                for image_id, name in connection.execute("SELECT image_id, name FROM images")
            }
    except ImportError:
        query = "SELECT image_id || char(9) || name FROM images;"
        result = subprocess.run(
            ["sqlite3", str(database_path), query],
            check=True,
            text=True,
            capture_output=True,
        )
        image_names = {}
        for line in result.stdout.splitlines():
            image_id, name = line.split("\t", maxsplit=1)
            image_names[int(image_id)] = name
        return image_names


def load_geolocations(
    database_path: Path,
    scene_root: Path,
    altitude_mode: str,
) -> dict[int, np.ndarray]:
    image_names = load_database_images(database_path)
    geolocations: dict[int, np.ndarray] = {}
    missing = 0
    for image_id, image_name in image_names.items():
        gps = gps_from_metadata(metadata_for_image(scene_root, image_name), altitude_mode)
        if gps is None:
            missing += 1
            continue
        geolocations[image_id] = lla_to_ecef(*gps)

    if missing:
        print(f"[warn] {database_path}: missing metadata for {missing} images", file=sys.stderr)
    return geolocations


def write_ecef_priors(database_path: Path, geolocations: dict[int, np.ndarray]) -> None:
    import sqlite3

    if not database_path.exists():
        return

    with sqlite3.connect(str(database_path)) as connection:
        connection.executemany(
            """
            UPDATE images
            SET prior_tx = ?, prior_ty = ?, prior_tz = ?
            WHERE image_id = ?
            """,
            [
                (float(ecef[0]), float(ecef[1]), float(ecef[2]), image_id)
                for image_id, ecef in geolocations.items()
            ],
        )
        connection.commit()


def write_scene_ecef_priors(scene_dir: Path, geolocations: dict[int, np.ndarray]) -> None:
    shared_dir = scene_dir / "shared"
    write_ecef_priors(shared_dir / "database.db", geolocations)
    for database_path in sorted(shared_dir.glob("database_threshold_*.db")):
        write_ecef_priors(database_path, geolocations)


def camera_center(image: Any) -> np.ndarray:
    rotation = image.qvec2rotmat()
    return -rotation.T @ image.tvec


def read_colmap_positions(model_path: Path) -> dict[int, np.ndarray]:
    images = read_images_binary(str(model_path / "images.bin"))
    return {
        int(image_id): camera_center(image)
        for image_id, image in images.items()
    }


def read_component_image_ids(component_path: Path) -> set[int]:
    return set(read_colmap_positions(component_path))


def fit_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_centered = source - np.mean(source, axis=0)
    target_centered = target - np.mean(target, axis=0)
    h = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        singular_values[-1] *= -1
        rotation = vt.T @ u.T

    denominator = np.sum(source_centered**2)
    if denominator <= 0:
        raise ValueError("Degenerate source points for similarity transform.")
    scale = float(np.sum(singular_values) / denominator)
    translation = np.mean(target, axis=0) - scale * rotation @ np.mean(source, axis=0)
    return scale, rotation, translation


def apply_similarity(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return scale * points @ rotation.T + translation


def fit_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([source, np.ones(len(source))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def predict_affine(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([points, np.ones(len(points))])
    return design @ coefficients


def issue_style_inliers(
    colmap_points: np.ndarray,
    geo_points: np.ndarray,
    *,
    residual_threshold: float,
    max_trials: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    best_mask: np.ndarray | None = None
    best_mean_residual = float("inf")

    if len(colmap_points) < 3:
        raise ValueError("Need at least 3 common cameras for RANSAC.")

    for _ in range(max_trials):
        sample = rng.choice(len(colmap_points), size=3, replace=False)
        coefficients = fit_affine(colmap_points[sample], geo_points[sample])
        residuals = np.sum(
            np.abs(predict_affine(colmap_points, coefficients) - geo_points),
            axis=1,
        )
        mask = residuals <= residual_threshold
        if mask.sum() < 3:
            continue

        mean_residual = float(residuals[mask].mean())
        if (
            best_mask is None
            or mask.sum() > best_mask.sum()
            or (mask.sum() == best_mask.sum() and mean_residual < best_mean_residual)
        ):
            best_mask = mask
            best_mean_residual = mean_residual

    if best_mask is None:
        raise ValueError("RANSAC failed to find at least 3 inliers.")
    return best_mask


def sim3_inliers(
    colmap_points: np.ndarray,
    geo_points: np.ndarray,
    *,
    residual_threshold: float,
    max_trials: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    best_mask: np.ndarray | None = None
    best_mean_error = float("inf")

    if len(colmap_points) < 3:
        raise ValueError("Need at least 3 common cameras for RANSAC.")

    for _ in range(max_trials):
        sample = rng.choice(len(colmap_points), size=3, replace=False)
        try:
            scale, rotation, translation = fit_similarity(
                colmap_points[sample],
                geo_points[sample],
            )
        except ValueError:
            continue

        transformed = apply_similarity(colmap_points, scale, rotation, translation)
        errors = np.linalg.norm(transformed - geo_points, axis=1)
        mask = errors <= residual_threshold
        if mask.sum() < 3:
            continue

        mean_error = float(errors[mask].mean())
        if (
            best_mask is None
            or mask.sum() > best_mask.sum()
            or (mask.sum() == best_mask.sum() and mean_error < best_mean_error)
        ):
            best_mask = mask
            best_mean_error = mean_error

    if best_mask is None:
        raise ValueError("RANSAC failed to find at least 3 inliers.")
    return best_mask


def estimate_alignment(
    colmap_positions: dict[int, np.ndarray],
    geolocations: dict[int, np.ndarray],
    *,
    residual_threshold: float,
    max_trials: int,
    random_state: int,
    method: str,
) -> dict[str, Any]:
    common_ids = sorted(set(colmap_positions).intersection(geolocations))
    if len(common_ids) < 3:
        raise ValueError("Need at least 3 common geotagged cameras.")

    colmap_points = np.array([colmap_positions[image_id] for image_id in common_ids])
    geo_points = np.array([geolocations[image_id] for image_id in common_ids])

    if method == "issue":
        inlier_mask = issue_style_inliers(
            colmap_points,
            geo_points,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            random_state=random_state,
        )
    elif method == "sim3":
        inlier_mask = sim3_inliers(
            colmap_points,
            geo_points,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    colmap_inliers = colmap_points[inlier_mask]
    geo_inliers = geo_points[inlier_mask]
    scale, rotation, translation = fit_similarity(colmap_inliers, geo_inliers)
    transformed = apply_similarity(colmap_inliers, scale, rotation, translation)

    return {
        "common_ids": common_ids,
        "inlier_mask": inlier_mask,
        "inlier_ratio": float(inlier_mask.sum() / len(common_ids)),
        "inlier_error": float(np.mean(np.linalg.norm(transformed - geo_inliers, axis=1))),
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
    }


def list_components(sparse_root: Path) -> list[Path]:
    if not sparse_root.exists():
        return []
    components = [
        path
        for path in sparse_root.iterdir()
        if path.is_dir() and (path / "images.bin").exists()
    ]
    return sorted(components, key=lambda path: int(path.name) if path.name.isdigit() else path.name)


def largest_component(sparse_root: Path) -> tuple[Path, set[int]] | None:
    best: tuple[Path, set[int]] | None = None
    for component_path in list_components(sparse_root):
        image_ids = read_component_image_ids(component_path)
        if best is None or len(image_ids) > len(best[1]):
            best = (component_path, image_ids)
    return best


def evaluate_component(
    component_path: Path,
    geolocations: dict[int, np.ndarray],
    args: argparse.Namespace,
    reference_image_ids: set[int] | None = None,
) -> dict[str, Any]:
    colmap_positions_by_id = read_colmap_positions(component_path)
    registered_image_ids = set(colmap_positions_by_id)
    if reference_image_ids is None:
        evaluation_image_ids = registered_image_ids
        reference_overlap_images = None
        selected_for_reference = True
    else:
        evaluation_image_ids = registered_image_ids.intersection(reference_image_ids)
        reference_overlap_images = len(evaluation_image_ids)
        selected_for_reference = reference_overlap_images > 0

    evaluated_positions_by_id = {
        image_id: position
        for image_id, position in colmap_positions_by_id.items()
        if image_id in evaluation_image_ids
    }
    common_ids = sorted(set(evaluated_positions_by_id).intersection(geolocations))
    base_result = {
        "component": component_path.name,
        "model_path": str(component_path),
        "registered_images": len(colmap_positions_by_id),
        "evaluated_images": len(evaluated_positions_by_id),
        "reference_overlap_images": reference_overlap_images,
        "selected_for_reference": selected_for_reference,
        "common_geotagged_images": len(common_ids),
        "qualifies": len(colmap_positions_by_id) >= args.min_registered_images,
    }

    if not selected_for_reference:
        return {
            **base_result,
            "inlier_count": 0,
            "inlier_ratio": None,
            "inlier_error_m": None,
            "error": "component does not share images with largest vanilla COLMAP model",
        }

    if len(common_ids) < 3:
        return {
            **base_result,
            "inlier_count": 0,
            "inlier_ratio": None,
            "inlier_error_m": None,
            "error": "fewer than 3 common geotagged cameras",
        }

    alignment = estimate_alignment(
        evaluated_positions_by_id,
        geolocations,
        residual_threshold=args.residual_threshold,
        max_trials=args.max_iterations,
        random_state=args.random_state,
        method=args.method,
    )
    inlier_mask = alignment["inlier_mask"]

    return {
        **base_result,
        "inlier_count": int(inlier_mask.sum()),
        "inlier_ratio": alignment["inlier_ratio"],
        "inlier_error_m": alignment["inlier_error"],
        "scale": float(alignment["scale"]),
        "rotation": alignment["rotation"].tolist(),
        "translation": alignment["translation"].tolist(),
    }


def evaluate_variant(
    sparse_root: Path,
    geolocations: dict[int, np.ndarray],
    args: argparse.Namespace,
    reference_image_ids: set[int] | None,
) -> list[dict[str, Any]]:
    return [
        evaluate_component(component, geolocations, args, reference_image_ids)
        for component in list_components(sparse_root)
    ]


def selected_components(
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid = [
        item
        for item in components
        if (
            item.get("inlier_ratio") is not None
            and item.get("selected_for_reference")
            and item.get("qualifies")
        )
    ]
    return sorted(
        valid,
        key=lambda item: (
            item["reference_overlap_images"]
            if item["reference_overlap_images"] is not None
            else item["registered_images"],
            item["evaluated_images"],
            item["common_geotagged_images"],
            item["inlier_ratio"],
        ),
        reverse=True,
    )


def weighted_inlier_ratio(components: list[dict[str, Any]]) -> float | None:
    total_weight = sum(int(item["common_geotagged_images"]) for item in components)
    if total_weight <= 0:
        return None
    total_inliers = sum(int(item["inlier_count"]) for item in components)
    return float(total_inliers / total_weight)


def visual_disambiguation_time_hours(scene_dir: Path, variant: str) -> float | None:
    timing_spec = VARIANT_VISUAL_DISAMBIGUATION_STAGES.get(variant)
    if timing_spec is None:
        return None

    timing_file, stage_name = timing_spec
    timing_path = scene_dir / timing_file
    if not timing_path.exists():
        return None

    timing = load_json(timing_path)
    stage = timing.get("stages", {}).get(stage_name)
    if stage is None or stage.get("duration_s") is None:
        return None
    return float(stage["duration_s"]) / 3600.0


def summarize_variant(
    scene_dir: Path,
    variant: str,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    reported_components = selected_components(components)
    for rank, component in enumerate(reported_components, start=1):
        component["rank"] = rank

    return {
        "weighted_inlier_ratio": weighted_inlier_ratio(reported_components),
        "weighted_common_geotagged_images": sum(
            int(item["common_geotagged_images"]) for item in reported_components
        ),
        "weighted_inlier_count": sum(int(item["inlier_count"]) for item in reported_components),
        "visual_disambiguation_time_h": visual_disambiguation_time_hours(scene_dir, variant),
        "reported_components": reported_components,
    }


def evaluate_scene(scene: str, args: argparse.Namespace) -> dict[str, Any]:
    scene_root = scene_dataset_root(args.benchmark_root, args.dataset_root, scene)
    database_path = args.benchmark_root / scene / "shared" / "database.db"
    if not database_path.exists():
        raise FileNotFoundError(f"Missing database for {scene}: {database_path}")

    geolocations = load_geolocations(database_path, scene_root, args.altitude)
    if args.write_priors:
        write_scene_ecef_priors(args.benchmark_root / scene, geolocations)

    variants = {
        name: args.benchmark_root / scene / output_dir / sparse_dir
        for name, (output_dir, sparse_dir) in VARIANT_SPARSE_ROOTS.items()
    }
    colmap_reference = largest_component(variants["vanilla"])
    if colmap_reference is None:
        reference_component_path = None
        reference_component = None
        reference_image_ids: set[int] | None = set()
    else:
        reference_component_path, reference_image_ids = colmap_reference
        reference_component = reference_component_path.name

    variant_results = {
        name: evaluate_variant(path, geolocations, args, reference_image_ids)
        for name, path in variants.items()
    }
    if reference_component_path is not None:
        reference_model_path = str(reference_component_path)
        for component in variant_results["vanilla"]:
            is_reference_component = component["model_path"] == reference_model_path
            component["selected_for_reference"] = is_reference_component
            if not is_reference_component:
                component["error"] = "not the largest vanilla COLMAP model"

    variant_summaries = {
        name: summarize_variant(args.benchmark_root / scene, name, components)
        for name, components in variant_results.items()
    }

    return {
        "scene": scene,
        "scene_root": str(scene_root),
        "database_path": str(database_path),
        "geotagged_images": len(geolocations),
        "colmap_reference_component": {
            "component": reference_component,
            "model_path": str(reference_component_path) if reference_component_path else None,
            "registered_images": len(reference_image_ids) if reference_image_ids else 0,
        },
        "component_results": variant_results,
        "variant_summaries": variant_summaries,
    }


def build_summary(args: argparse.Namespace, scenes: list[str]) -> dict[str, Any]:
    return {
        "benchmark_root": str(args.benchmark_root),
        "dataset_root": str(args.dataset_root) if args.dataset_root else None,
        "residual_threshold_m": args.residual_threshold,
        "min_registered_images": args.min_registered_images,
        "component_selection": (
            "largest vanilla COLMAP component; for visual_disambiguation and dg++, "
            "all components sharing images with that component"
        ),
        "weighted_inlier_ratio_weight": (
            "common_geotagged_images in each selected component's overlap with the "
            "largest vanilla COLMAP component"
        ),
        "method": args.method,
        "altitude": args.altitude,
        "scenes": [evaluate_scene(scene, args) for scene in scenes],
    }


def format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:7.2f}%"


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:8.2f}"


def print_summary(summary: dict[str, Any]) -> None:
    variant_width = max(len("variant"), *(len(name) for name in VARIANT_SPARSE_ROOTS))
    header = (
        f"{'scene':<12} {'variant':<{variant_width}} {'rank':>4} {'comp':>5} "
        f"{'reg':>6} {'eval':>6} {'ovlp':>6} {'geo':>6} {'inl':>6} {'IR':>9} {'wIR':>9} "
        f"{'err(m)':>9} {'disamb(h)':>10}"
    )
    print("Inlier ratio for components matching the largest vanilla COLMAP reconstruction:")
    print(header)
    print("-" * len(header))
    for scene_result in summary["scenes"]:
        for variant, variant_summary in scene_result["variant_summaries"].items():
            components = variant_summary["reported_components"]
            if not components:
                continue
            for component in components:
                print(
                    f"{scene_result['scene']:<12} "
                    f"{variant:<{variant_width}} "
                    f"{component['rank']:>4} "
                    f"{component['component']:>5} "
                    f"{component['registered_images']:>6} "
                    f"{component['evaluated_images']:>6} "
                    f"{component['reference_overlap_images'] or 0:>6} "
                    f"{component['common_geotagged_images']:>6} "
                    f"{component['inlier_count']:>6} "
                    f"{format_ratio(component['inlier_ratio']):>9} "
                    f"{format_ratio(variant_summary['weighted_inlier_ratio']):>9} "
                    f"{format_float(component['inlier_error_m']):>9} "
                    f"{format_float(variant_summary['visual_disambiguation_time_h']):>10}"
                )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "variant",
        "rank",
        "qualifies",
        "selected_for_reference",
        "component",
        "registered_images",
        "evaluated_images",
        "reference_overlap_images",
        "common_geotagged_images",
        "inlier_count",
        "inlier_ratio",
        "weighted_inlier_ratio",
        "weighted_common_geotagged_images",
        "weighted_inlier_count",
        "inlier_error_m",
        "visual_disambiguation_time_h",
        "model_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene_result in summary["scenes"]:
            for variant, variant_summary in scene_result["variant_summaries"].items():
                for component in variant_summary["reported_components"]:
                    writer.writerow(
                        {
                            "scene": scene_result["scene"],
                            "variant": variant,
                            "rank": component["rank"],
                            "qualifies": component["qualifies"],
                            "weighted_inlier_ratio": variant_summary["weighted_inlier_ratio"],
                            "weighted_common_geotagged_images": variant_summary[
                                "weighted_common_geotagged_images"
                            ],
                            "weighted_inlier_count": variant_summary["weighted_inlier_count"],
                            "visual_disambiguation_time_h": variant_summary[
                                "visual_disambiguation_time_h"
                            ],
                            **{
                                key: component.get(key)
                                for key in fieldnames
                                if key
                                not in {
                                    "scene",
                                    "variant",
                                    "rank",
                                    "qualifies",
                                    "weighted_inlier_ratio",
                                    "weighted_common_geotagged_images",
                                    "weighted_inlier_count",
                                    "visual_disambiguation_time_h",
                                }
                            },
                        }
                    )


def main() -> None:
    args = parse_args()
    args.benchmark_root = args.benchmark_root.resolve()
    if args.dataset_root is not None:
        args.dataset_root = args.dataset_root.resolve()

    scenes = load_scenes(args.benchmark_root, args.scenes)
    if not scenes:
        raise ValueError(f"No scenes found under {args.benchmark_root}")

    summary = build_summary(args, scenes)
    print_summary(summary)

    if args.json_out:
        write_json(args.json_out, summary)
        print(f"\nWrote JSON: {args.json_out}")
    if args.csv_out:
        write_csv(args.csv_out, summary)
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
