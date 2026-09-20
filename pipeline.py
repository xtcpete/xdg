"""Example hloc reconstruction using ALIKED + LightGlue and XDG.

Uses exhaustive pairs for small collections and NetVLAD retrieval otherwise,
following https://github.com/xtcpete/rdd/blob/main/demo_sfm.ipynb.
For better feature matching, see https://github.com/xtcpete/rdd.
"""

import argparse
from pathlib import Path
import sqlite3


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", "--input_image_path", type=Path, required=True)
    parser.add_argument(
        "--outputs", "--output_path", type=Path,
        default=Path("outputs/reconstruction"),
        help="Empty output directory (use a new directory for each run).",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/model_configs/xdg.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("weights/xdg.pth"))
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--feature_conf", default="aliked-n16")
    parser.add_argument("--matcher_conf", default="aliked+lightglue")
    parser.add_argument("--retrieval_conf", default="netvlad")
    parser.add_argument("--exhaustive_if_less", type=int, default=30)
    parser.add_argument("--num_matched", type=int, default=20)
    parser.add_argument("--min_match_score", type=float, default=0.2)
    parser.add_argument(
        "--camera_mode", choices=["AUTO", "SINGLE", "PER_FOLDER", "PER_IMAGE"],
        default="PER_IMAGE",
    )
    parser.add_argument("--device", default=None, help="XDG device; defaults to CUDA if available.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def has_verified_pairs(database):
    with sqlite3.connect(str(database)) as connection:
        return connection.execute(
            "SELECT 1 FROM two_view_geometries "
            "WHERE rows > 0 AND data IS NOT NULL LIMIT 1"
        ).fetchone() is not None


def main(argv=None):
    args = get_args(argv)
    if not args.images.is_dir():
        raise NotADirectoryError(args.images)
    for path in (args.config, args.ckpt):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.num_matched < 1 or args.exhaustive_if_less < 0:
        raise ValueError("--num_matched must be positive and --exhaustive_if_less nonnegative.")
    # Both hloc and XDG cache results by filename. A fresh directory prevents
    # reuse of probabilities/features from different images or checkpoints.
    if args.outputs.exists() and any(args.outputs.iterdir()):
        raise FileExistsError(f"Use an empty --outputs directory: {args.outputs}")

    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = sorted(
        path.relative_to(args.images).as_posix()
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    if len(images) < 2:
        raise ValueError(f"At least two images are required in {args.images}.")
    if any(any(char.isspace() for char in name) for name in images):
        raise ValueError("hloc pair files require image names without whitespace.")

    import pycolmap
    from hloc import (
        extract_features, match_features, pairs_from_exhaustive,
        pairs_from_retrieval, reconstruction,
    )
    from remove_doppelgangers import main as filter_database

    feature_conf = extract_features.confs[args.feature_conf]
    matcher_conf = match_features.confs[args.matcher_conf]
    args.outputs.mkdir(parents=True, exist_ok=True)
    pairs = args.outputs / "sfm_pairs.txt"

    # Image retrieval / pair selection.
    if len(images) < args.exhaustive_if_less:
        pairs_from_exhaustive.main(pairs, image_list=images)
    else:
        retrieval = extract_features.main(
            extract_features.confs[args.retrieval_conf], args.images,
            args.outputs, image_list=images,
        )
        pairs_from_retrieval.main(
            retrieval, pairs, num_matched=min(args.num_matched, len(images) - 1),
            query_list=images, db_list=images,
        )

    # Local feature extraction and matching.
    features = extract_features.main(
        feature_conf, args.images, args.outputs, image_list=images,
    )
    matches = match_features.main(
        matcher_conf, pairs, features, matches=args.outputs / "matches.h5",
    )

    # Follow hloc.reconstruction.main up to geometric verification, then insert
    # XDG. Calling reconstruction.main afterwards would recreate the database.
    sfm_dir = args.outputs / "sfm"
    sfm_dir.mkdir()
    database = sfm_dir / "database.db"
    reconstruction.create_empty_db(database)
    reconstruction.import_images(
        args.images, database, getattr(pycolmap.CameraMode, args.camera_mode),
        image_list=images,
    )
    image_ids = reconstruction.get_image_ids(database)
    reconstruction.import_features(image_ids, database, features)
    reconstruction.import_matches(
        image_ids, database, pairs, matches,
        min_match_score=args.min_match_score,
        skip_geometric_verification=False,
    )
    reconstruction.estimation_and_geometric_verification(database, pairs, args.verbose)
    if not has_verified_pairs(database):
        raise RuntimeError("No geometrically verified pairs; cannot filter or reconstruct.")

    # Keep the original database and map only from the filtered copy.
    filtered_database = Path(filter_database(
        config=str(args.config), ckpt=str(args.ckpt),
        input_image_path=str(args.images), output_path=str(args.outputs / "filtering"),
        database_path=str(database), threshold=args.threshold,
        device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
    ))
    if not has_verified_pairs(filtered_database):
        raise RuntimeError("XDG removed all verified pairs; try a lower --threshold.")
    model = reconstruction.run_reconstruction(
        sfm_dir, filtered_database, args.images, verbose=args.verbose,
    )
    if model is None:
        raise RuntimeError("Mapping failed to reconstruct a model; inspect the filtered pairs.")
    print(model.summary())
    print(f"Reconstruction saved to {sfm_dir}")
    return model


if __name__ == "__main__":
    main()
