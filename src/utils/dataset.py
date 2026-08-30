import cv2
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose

try:
    PIL_LANCZOS = Image.Resampling.LANCZOS
    PIL_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    PIL_LANCZOS = Image.LANCZOS
    PIL_BICUBIC = Image.BICUBIC


def normalize_target_size(size):
    if size is None:
        return None
    if isinstance(size, np.ndarray):
        size = size.tolist()
    if isinstance(size, (list, tuple)):
        if len(size) != 2:
            raise ValueError(f"Expected size as int or (width, height), got {size}")
        width, height = int(size[0]), int(size[1])
    else:
        width = height = int(size)
    if width <= 0 or height <= 0:
        raise ValueError(f"Target size must be positive, got {(width, height)}")
    return (width, height)

def imread_rgb(path):
    path = str(path)
    if path.lower().endswith("gif"):
        raise ValueError(f"GIF images are not supported: {path}")

    try:
        with Image.open(path) as pil_image:
            image = np.asarray(exif_transpose(pil_image).convert("RGB"))
    except Exception:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image at {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if len(image.shape)<3:
        image = image[:, :, np.newaxis]
    if image.shape[2]<3:
        image = np.concatenate((image, image, image), axis=2)
    return image  # (h, w, 3), RGB


def read_image_size(path):
    path = str(path)
    if path.lower().endswith("gif"):
        raise ValueError(f"GIF images are not supported: {path}")

    try:
        with Image.open(path) as pil_image:
            width, height = pil_image.size
            orientation = pil_image.getexif().get(0x0112)
            if orientation in {5, 6, 7, 8}:
                width, height = height, width
            return int(width), int(height)
    except Exception:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image at {path}")
        height, width = image.shape[:2]
        return int(width), int(height)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=14):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    if w_new == 0:
        w_new = df
    if h_new == 0:
        h_new = df
    return w_new, h_new


def _pil_resize(image, size):
    target_w, target_h = int(size[0]), int(size[1])
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Resize target must be positive, got {(target_w, target_h)}")

    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    height_orig, width_orig = image.shape[:2]
    if (width_orig, height_orig) == (target_w, target_h):
        return image.astype(np.float32, copy=False)

    interpolation = (
        PIL_LANCZOS
        if target_w < width_orig or target_h < height_orig
        else PIL_BICUBIC
    )
    resized = Image.fromarray(image).resize((target_w, target_h), interpolation)
    return np.asarray(resized, dtype=np.float32)


def _resize_to_fit_target(image, resize):
    image = image.astype(np.float32, copy=False)
    resize_wh = normalize_target_size(resize)
    if resize_wh is None:
        return image

    target_w, target_h = resize_wh
    height_orig, width_orig = image.shape[:2]
    scale = min(target_w / float(width_orig), target_h / float(height_orig))
    resized_w = max(1, int(round(width_orig * scale)))
    resized_h = max(1, int(round(height_orig * scale)))
    return _pil_resize(image, (resized_w, resized_h))


def _center_pad_to_target_with_mask(image, resize):
    resize_wh = normalize_target_size(resize)
    if resize_wh is None:
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        return image, mask

    target_w, target_h = resize_wh
    height, width = image.shape[:2]
    if width > target_w or height > target_h:
        raise ValueError(
            f"Cannot pad image size {(width, height)} into smaller target {(target_w, target_h)}"
        )

    padded = np.zeros((target_h, target_w, image.shape[2]), dtype=np.float32)
    mask = np.zeros((target_h, target_w), dtype=np.uint8)

    top = (target_h - height) // 2
    left = (target_w - width) // 2
    padded[top : top + height, left : left + width] = image
    mask[top : top + height, left : left + width] = 1
    return padded, mask


def _center_crop_image(image, target_w, target_h):
    return _crop_image(image, target_w, target_h, crop_strategy="center")


def _crop_image(image, target_w, target_h, crop_strategy="center", rng=None):
    height, width = image.shape[:2]
    if target_w > width or target_h > height:
        raise ValueError(
            f"Cannot center crop {(target_w, target_h)} from image size {(width, height)}"
        )

    if crop_strategy == "center":
        left = (width - target_w) // 2
        top = (height - target_h) // 2
    elif crop_strategy == "random":
        if rng is None:
            rng = np.random.default_rng()
        left = 0 if target_w == width else int(rng.integers(0, width - target_w + 1))
        top = 0 if target_h == height else int(rng.integers(0, height - target_h + 1))
    else:
        raise ValueError(
            f"Unknown crop_strategy: {crop_strategy}. Expected 'center' or 'random'."
        )

    return image[top : top + target_h, left : left + target_w]


def _rng_integers(rng, high):
    if rng is None:
        rng = np.random.default_rng()
    if hasattr(rng, "integers"):
        return int(rng.integers(high))
    return int(rng.randint(0, high))


def _pad_with_optional_aug_crop(image, base_target_wh, aug_crop=0, rng=None):
    resize_target_wh = base_target_wh
    if aug_crop and aug_crop > 1:
        jitter = _rng_integers(rng, int(aug_crop))
        resize_target_wh = (
            int(base_target_wh[0] + jitter),
            int(base_target_wh[1] + jitter),
        )
        image = _resize_to_cover_target(image, resize_target_wh)
    image = _resize_to_fit_target(image, base_target_wh)
    image, mask = _center_pad_to_target_with_mask(image, base_target_wh)
    return image, mask


def _resolve_long_edge_size(resize):
    resize_wh = normalize_target_size(resize)
    if resize_wh is None:
        return None
    return int(max(resize_wh))


def _crop_to_patch_grid(image, patch_size, square_ok=False):
    height, width = image.shape[:2]
    crop_w = (width // patch_size) * patch_size
    crop_h = (height // patch_size) * patch_size
    if crop_w < patch_size or crop_h < patch_size:
        raise ValueError(
            f"Image is too small after resize for patch size {patch_size}: {(width, height)}"
        )
    if not square_ok and width == height:
        crop_h = max(
            patch_size,
            int(((3 * crop_w) // 4) // patch_size * patch_size),
        )
    return _crop_image(image, crop_w, crop_h, crop_strategy="center")


def _load_single_image_dust3r_like(path, resize, patch_size, square_ok=False):
    image = imread_rgb(path).astype(np.float32)
    long_edge_size = _resolve_long_edge_size(resize)
    if long_edge_size is not None:
        image = _resize(image, long_edge_size)
    image = _crop_to_patch_grid(image, patch_size=patch_size, square_ok=square_ok)
    return image


def _center_pad_pair_to_common_shape(image0, image1):
    target_h = max(image0.shape[0], image1.shape[0])
    target_w = max(image0.shape[1], image1.shape[1])
    image0, mask0 = _center_pad_to_target_with_mask(image0, (target_w, target_h))
    image1, mask1 = _center_pad_to_target_with_mask(image1, (target_w, target_h))
    return image0, image1, mask0, mask1


def _resize(image, resize):
    image = image.astype(np.float32, copy=False)
    resize_wh = normalize_target_size(resize)
    if resize_wh is None:
        return image

    height_orig, width_orig = image.shape[:2]
    size = max(resize_wh)
    if size == 224:
        resize_long_edge = round(size * max(width_orig / height_orig, height_orig / width_orig))
    else:
        resize_long_edge = size

    resized_w, resized_h = get_resized_wh(width_orig, height_orig, resize=resize_long_edge)
    return _pil_resize(image, (resized_w, resized_h))


def _resize_to_cover_target(image, resize):
    image = image.astype(np.float32, copy=False)
    resize_wh = normalize_target_size(resize)
    if resize_wh is None:
        return image

    target_w, target_h = resize_wh
    height_orig, width_orig = image.shape[:2]
    scale = max(target_w / float(width_orig), target_h / float(height_orig))
    resized_w = max(target_w, int(round(width_orig * scale)))
    resized_h = max(target_h, int(round(height_orig * scale)))
    return _pil_resize(image, (resized_w, resized_h))


def _read_image_resize(path, resize):
    image = imread_rgb(path).astype(np.float32)
    height_orig, width_orig = image.shape[:2]
    image = _resize(image, resize)
    return image, (width_orig, height_orig)


def read_images(
    path0,
    path1,
    resize=None,
    mode="resize",
    return_mask=False,
    crop_strategy="center",
    rng=None,
    aug_crop=0,
    patch_size=14,
):
    """
    Args:
        path0, path1 (str): image path.
        resize (int, optional): if provided, target size for processing.
        mode (str, optional): image processing mode.
            - 'center_crop': aspect-preserving resize, then crop to target
            - 'resize': presize, then resize to the requested output size
            - 'pad': aspect-preserving resize to fit, then symmetric pad to target
            - 'dust3r_like': resize long edge, crop to patch grid, then pad pair to a common shape
            - 'both': return both center cropped and resized images
    Returns:
        image0 (numpy.array): (C, H, W) RGB image, normalized to [0, 1]
        image1 (numpy.array): (C, H, W) RGB image, normalized to [0, 1]
    """
    resize_wh = normalize_target_size(resize)

    if mode == 'resize':
        image0, (w0, h0) = _read_image_resize(path0, resize)
        image1, (w1, h1) = _read_image_resize(path1, resize)
        w0, h0 = image0.shape[1], image0.shape[0]
        w1, h1 = image1.shape[1], image1.shape[0]
    elif mode == 'center_crop':
        image0 = imread_rgb(path0).astype(np.float32)
        image1 = imread_rgb(path1).astype(np.float32)
        w0, h0 = image0.shape[1], image0.shape[0]
        w1, h1 = image1.shape[1], image1.shape[0]
    elif mode == 'dust3r_like':
        image0 = _load_single_image_dust3r_like(
            path0,
            resize=resize,
            patch_size=patch_size,
            square_ok=False,
        )
        image1 = _load_single_image_dust3r_like(
            path1,
            resize=resize,
            patch_size=patch_size,
            square_ok=False,
        )
        w0, h0 = image0.shape[1], image0.shape[0]
        w1, h1 = image1.shape[1], image1.shape[0]
    else:
        # Read image0
        image0 = imread_rgb(path0).astype(np.float32)
        w0, h0 = image0.shape[1], image0.shape[0]

        # Read image1
        image1 = imread_rgb(path1).astype(np.float32)
        w1, h1 = image1.shape[1], image1.shape[0]

    if mode == 'center_crop':
        if resize_wh is not None:
            image0 = _resize_to_cover_target(image0, resize_wh)
            image0 = _crop_image(
                image0,
                resize_wh[0],
                resize_wh[1],
                crop_strategy=crop_strategy,
                rng=rng,
            )

        if resize_wh is not None:
            image1 = _resize_to_cover_target(image1, resize_wh)
            image1 = _crop_image(
                image1,
                resize_wh[0],
                resize_wh[1],
                crop_strategy=crop_strategy,
                rng=rng,
            )

    elif mode == 'resize':
        if resize_wh is not None:
            image0 = _pil_resize(image0, resize_wh)
            image1 = _pil_resize(image1, resize_wh)
    elif mode == 'pad':
        if resize_wh is not None:
            image0, mask0 = _pad_with_optional_aug_crop(
                image0,
                resize_wh,
                aug_crop=aug_crop,
                rng=rng,
            )

            image1, mask1 = _pad_with_optional_aug_crop(
                image1,
                resize_wh,
                aug_crop=aug_crop,
                rng=rng,
            )
        else:
            mask0 = np.ones(image0.shape[:2], dtype=np.uint8)
            mask1 = np.ones(image1.shape[:2], dtype=np.uint8)
    elif mode == 'dust3r_like':
        image0, image1, mask0, mask1 = _center_pad_pair_to_common_shape(image0, image1)
    elif mode == 'both':
        # Return both center-cropped-square and directly resized images.
        crop_size = min(w0, h0)
        left = (w0 - crop_size) // 2
        top = (h0 - crop_size) // 2
        image0_cropped = image0[top:top+crop_size, left:left+crop_size]

        if resize_wh is not None:
            image0_resized = _pil_resize(image0, resize_wh)
            image0_cropped_resized = _pil_resize(image0_cropped, resize_wh)
        else:
            image0_resized = image0
            image0_cropped_resized = image0_cropped

        crop_size = min(w1, h1)
        left = (w1 - crop_size) // 2
        top = (h1 - crop_size) // 2
        image1_cropped = image1[top:top+crop_size, left:left+crop_size]
        if resize_wh is not None:
            image1_resized = _pil_resize(image1, resize_wh)
            image1_cropped_resized = _pil_resize(image1_cropped, resize_wh)
        else:
            image1_resized = image1
            image1_cropped_resized = image1_cropped
    else:
        raise ValueError(
            f"Unknown mode: {mode}. Expected 'center_crop', 'resize', 'pad', 'dust3r_like', or 'both'"
        )
    
    if mode != 'both':
        # Convert to (C, H, W)
        image0 = np.transpose(image0, (2, 0, 1))
        image1 = np.transpose(image1, (2, 0, 1))

        # Normalize to [0, 1]
        image0 = image0 / 255.0
        image1 = image1 / 255.0
    else:
        # Convert to (C, H, W) and normalize to [0, 1]
        image0_resized = np.transpose(image0_resized, (2, 0, 1)) / 255.0
        image1_resized = np.transpose(image1_resized, (2, 0, 1)) / 255.0
        image0_cropped_resized = np.transpose(image0_cropped_resized, (2, 0, 1)) / 255.0
        image1_cropped_resized = np.transpose(image1_cropped_resized, (2, 0, 1)) / 255.0

        # Return both cropped and resized images as a tuple
        image0 = (image0_cropped_resized, image0_resized)
        image1 = (image1_cropped_resized, image1_resized)
    
    if return_mask and mode in {"pad", "dust3r_like"}:
        mask0 = mask0.astype(np.float32)[None, ...]
        mask1 = mask1.astype(np.float32)[None, ...]
        return image0, image1, mask0, mask1

    return image0, image1
