import os
import numpy as np
import argparse
import torch
import importlib
from tqdm import tqdm

from src.utils.config import load_model_config
from src.utils.prediction import vote_ensemble_probs, vote_symmetrical_logits
from training.precision import resolve_autocast_dtype

def get_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Structure from Motion disambiguation with Doppelgangers classification model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the model config file.')
    parser.add_argument('--database_path', required=False, type=str, default=None, help="Path to the COLMAP database.")
    parser.add_argument(
        '--pairs_txt',
        type=str,
        default=None,
        help='Optional path to pairs.txt (each line: "image1 image2"). If provided, pairs are read from this file instead of querying the DB for all pairs.',
    )
    parser.add_argument('--input_image_path', type=str, required=True, help='Path to the input image dataset.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output results.')
    parser.add_argument('--threshold', type=float, default=0.8, help='Doppelgangers threshold.')
    parser.add_argument('--ckpt', type=str, default='weights/xdg.pth', help="Path to the model checkpoint.")
    parser.add_argument('--batch_size', type=int, default=None, help='Override the model config inference batch size.')
    parser.add_argument('--img_size', type=int, default=None, help='Input image size for the classifier.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (e.g. cuda, cuda:0, cpu).')
    parser.add_argument('--num_workers', type=int, default=None, help='Override the model config dataloader worker count.')
    parser.add_argument('--mode', type=str, default=None, help="Image mode override (resize, center_crop, pad).")

    args = parser.parse_args(argv)
    return args


def _get_cfg_value(cfg, *keys, default=None):
    value = cfg
    for key in keys:
        if not hasattr(value, key):
            return default
        value = getattr(value, key)
    return value


def _set_matmul_precision(cfg):
    precision = _get_cfg_value(cfg, "inference", "matmul_precision", default="high")
    if precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(precision))


def _load_decoder_checkpoint(model, checkpoint_path, strict=True):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("dec") or ckpt.get("state_dict") or ckpt.get("model") or ckpt
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} does not contain a state dictionary."
        )

    target_keys = set(model.state_dict())
    candidates = [state_dict]
    for prefix in ("module.", "model.", "model.module."):
        stripped = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if stripped:
            candidates.append(stripped)
    state_dict = max(
        candidates,
        key=lambda candidate: len(target_keys.intersection(candidate)),
    )

    model.load_state_dict(state_dict, strict=strict)


def _collapse_model_logits(model, logits):
    if bool(getattr(model, "return_symmetrical_logits", False)):
        return vote_symmetrical_logits(logits)
    return logits


def _create_pair_list_from_txt(db_path, pairs_txt_path, output_path):
    from src.utils.database import COLMAPDatabase, image_ids_to_pair_id

    if not os.path.exists(pairs_txt_path):
        raise FileNotFoundError(f"pairs.txt not found: {pairs_txt_path}")

    name_to_id = None
    if db_path is not None:
        db = COLMAPDatabase.connect(db_path)
        try:
            image_rows = db.execute("SELECT image_id, name FROM images").fetchall()
            name_to_id = {name: image_id for image_id, name in image_rows}
        finally:
            db.close()

    pairs_list = []
    missing_names = set()
    with open(pairs_txt_path, "r") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid line {line_idx} in {pairs_txt_path}: expected 2 columns, got {len(parts)}."
                )

            name1, name2 = parts
            if name_to_id is not None:
                id1 = name_to_id.get(name1)
                id2 = name_to_id.get(name2)
                if id1 is None:
                    missing_names.add(name1)
                if id2 is None:
                    missing_names.add(name2)
                if id1 is None or id2 is None:
                    continue
                pair_id = image_ids_to_pair_id(id1, id2)
                pairs_list.append([name1, name2, 0, 0, pair_id])
            else:
                pairs_list.append([name1, name2, 0, 0])

    if missing_names:
        missing_sorted = ", ".join(sorted(missing_names))
        raise ValueError(
            f"Found {len(missing_names)} image names in {pairs_txt_path} that are not present in the DB: {missing_sorted}"
        )

    num_columns = 5 if name_to_id is not None else 4
    pairs_array = np.array(pairs_list, dtype=object).reshape(-1, num_columns)
    pair_path = os.path.join(output_path, "pairs_list.npy")
    np.save(pair_path, pairs_array)
    return pair_path


def _build_inference_loader(args, cfg, pair_path):
    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None or not hasattr(inference_cfg, "dataset_type"):
        raise ValueError("Model config must define inference.dataset_type.")
    data_lib = importlib.import_module(inference_cfg.dataset_type)
    dataset_cls = getattr(data_lib, "DoppelgangersDataset")

    img_size = args.img_size
    if img_size is None:
        img_size = _get_cfg_value(cfg, "inference", "img_size")
    if img_size is None:
        img_size = _get_cfg_value(cfg, "model", "backbone", "img_size")
    if img_size is None:
        img_size = 504
    print(f"Using image size: {img_size}")
    mode = args.mode
    if mode is None:
        mode = _get_cfg_value(cfg, "inference", "mode", default="resize")

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(_get_cfg_value(cfg, "inference", "batch_size", default=1) or 1)
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(_get_cfg_value(cfg, "inference", "num_workers", default=8) or 0)

    dataset = dataset_cls(
        [args.input_image_path],
        [pair_path],
        img_size=img_size,
        mode=mode,
        phase='Inference',
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return data_loader


def doppelgangers_classifier(args, cfg, pair_path):
    """Classify image pairs using the Doppelgangers ViT classifier."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    _set_matmul_precision(cfg)

    decoder_lib = importlib.import_module(cfg.model.type)
    model = decoder_lib.decoder(cfg.model, load_backbone_pretrained=False)
    _load_decoder_checkpoint(model, args.ckpt, strict=True)
    model.to(device)
    model.eval()

    data_loader = _build_inference_loader(args, cfg, pair_path)
    prob_list = []

    if os.path.exists(f"{args.output_path}/pair_probability_list.npy"):
        print(f"Found existing probability list at {args.output_path}/pair_probability_list.npy, skipping inference.")
        return

    autocast_dtype = resolve_autocast_dtype("auto", device=device)
    if autocast_dtype is not None:
        print(f"Using autocast dtype: {str(autocast_dtype).replace('torch.', '')}")

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Disambiguating pairs"):
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device, non_blocking=True)

            images = batch.get("images")
            images_resized = batch.get("images_resized")
            if images_resized is None:
                images_resized = batch.get("image_resized")
            masks = batch.get("masks")
            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    score = model(images=images, masks=masks)
            else:
                score = model(images=images, masks=masks)
            score = _collapse_model_logits(model, score)
            probs = torch.softmax(score, dim=1)

            if images_resized is not None:
                if autocast_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        score_resized = model(images=images_resized, masks=masks)
                else:
                    score_resized = model(images=images_resized, masks=masks)
                score_resized = _collapse_model_logits(model, score_resized)
                probs_resized = torch.softmax(score_resized, dim=1)
                probs = vote_ensemble_probs(probs, probs_resized)

            prob_list.append(probs.float().cpu().numpy())

    if prob_list:
        prob_array = np.concatenate(prob_list, axis=0)
    else:
        prob_array = np.empty((0, 2), dtype=np.float32)

    np.save(f"{args.output_path}/pair_probability_list.npy", {'prob': prob_array})


def _build_args(
    config,
    input_image_path,
    output_path,
    database_path=None,
    pairs_txt=None,
    threshold=0.8,
    ckpt="weights/xdg.pth",
    batch_size=None,
    img_size=None,
    device="cuda",
    num_workers=None,
    mode=None,
):
    missing = [
        name
        for name, value in (
            ("config", config),
            ("input_image_path", input_image_path),
            ("output_path", output_path),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing required argument(s): {', '.join(missing)}")

    return argparse.Namespace(
        config=config,
        database_path=database_path,
        pairs_txt=pairs_txt,
        input_image_path=input_image_path,
        output_path=output_path,
        threshold=threshold,
        ckpt=ckpt,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        num_workers=num_workers,
        mode=mode,
    )


def main(
    config=None,
    input_image_path=None,
    output_path=None,
    *,
    database_path=None,
    pairs_txt=None,
    threshold=0.8,
    ckpt="weights/xdg.pth",
    batch_size=None,
    img_size=None,
    device="cuda",
    num_workers=None,
    mode=None,
    args=None,
    argv=None,
):
    """Run Doppelgangers filtering from Python or from the command line.

    When imported, call this function with explicit keyword arguments. When the
    module is executed as a script, call it with no arguments and it will parse
    ``sys.argv``. Tests or wrappers can pass ``argv`` to exercise CLI parsing.
    """
    if args is not None:
        if argv is not None or any(
            value is not None
            for value in (
                config,
                input_image_path,
                output_path,
                database_path,
                pairs_txt,
            )
        ):
            raise ValueError(
                "Pass either args, argv, or explicit keyword arguments, not a mixture."
            )
    elif argv is not None:
        args = get_args(argv)
    elif config is None and input_image_path is None and output_path is None:
        args = get_args()
    else:
        args = _build_args(
            config=config,
            database_path=database_path,
            pairs_txt=pairs_txt,
            input_image_path=input_image_path,
            output_path=output_path,
            threshold=threshold,
            ckpt=ckpt,
            batch_size=batch_size,
            img_size=img_size,
            device=device,
            num_workers=num_workers,
            mode=mode,
        )

    cfg = load_model_config(args.config)
    os.makedirs(args.output_path, exist_ok=True)

    if args.pairs_txt is not None:
        pair_path = _create_pair_list_from_txt(args.database_path, args.pairs_txt, args.output_path)
    else:
        if args.database_path is None:
            raise ValueError("--database_path is required when --pairs_txt is not provided.")
        from src.utils.process_database import create_image_pair_list

        pair_path = create_image_pair_list(args.database_path, args.output_path)

    doppelgangers_classifier(args, cfg, pair_path)
    update_database_path = None
    if args.database_path is not None:
        from src.utils.process_database import remove_doppelgangers

        update_database_path = remove_doppelgangers(
            args.database_path,
            f"{args.output_path}/pair_probability_list.npy",
            pair_path,
            args.threshold,
        )
    else:
        print("Skipping database update because --database_path was not provided.")

    return update_database_path

if __name__ == '__main__':
    main()
