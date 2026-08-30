import os.path as osp
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from ..utils.dataset import normalize_target_size, read_image_size, read_images

class DoppelgangersDataset(Dataset):
    def __init__(self,
                 image_dir,
                 pair_path,
                 img_size,
                 mode,
                 phase,
                 augment=None,
                 flip_order_prob=None,
                 aug_crop=0,
                 seed=None,
                 patch_size=14):
        """
        Doppelgangers dataset: loading images for Doppelgangers model.
        
        Args:
            image_dir (str): root directory for images.
            pair_path (str): pair_list.npy path. This contains image pair information.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 504 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
        """
        super().__init__()
        self.phase = phase
        self.image_dir = image_dir
        self.pairs_info = np.load(pair_path[0], allow_pickle=True)

        # Track cumulative lengths for each dataset
        self.cumulative_lengths = [len(self.pairs_info)]

        # Load and concatenate all additional pair paths
        for i in range(1, len(pair_path)):
            additional_pairs = np.load(pair_path[i], allow_pickle=True)
            self.pairs_info = np.concatenate([self.pairs_info, additional_pairs], axis=0)
            self.cumulative_lengths.append(len(self.pairs_info))
        
        print('loading images, #pairs: ', len(self.pairs_info))
        self.img_size = normalize_target_size(img_size)
        self.mode = mode
        self.augment = augment if phase.lower() == "train" else None
        self.aug_crop = int(aug_crop or 0) if phase.lower() == "train" else 0
        self.seed = None if seed is None else int(seed)
        self.patch_size = int(patch_size)
        self._image_size_cache = {}
        self._resolution_choice_cache = {}
        self._invalid_pair_indices = set()
        self.flip_order_prob = 0.0
        if phase.lower() == "train":
            if flip_order_prob is not None:
                self.flip_order_prob = float(flip_order_prob)
            elif isinstance(self.augment, dict):
                self.flip_order_prob = float(self.augment.get("flip_order_prob", 0.0))
            elif self.augment is not None:
                self.flip_order_prob = float(getattr(self.augment, "flip_order_prob", 0.0))

        
    def __len__(self):
        return len(self.pairs_info)

    def _apply_affine(self, image, angle, scale, mask=None):
        image = TF.affine(
            image,
            angle=angle,
            translate=[0, 0],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        if mask is None:
            return image, None
        mask = TF.affine(
            mask,
            angle=angle,
            translate=[0, 0],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )
        mask = (mask > 0.5).float()
        return image, mask

    def _apply_random_crop(self, image, mask=None):
        if isinstance(self.augment, dict):
            crop_prob = float(self.augment.get("crop_prob", 0.0))
            crop_scale_min = float(self.augment.get("crop_scale_min", 0.8))
            crop_scale_max = float(self.augment.get("crop_scale_max", 1.0))
            crop_width_prob = float(self.augment.get("crop_width_prob", 0.5))
        else:
            crop_prob = float(getattr(self.augment, "crop_prob", 0.0))
            crop_scale_min = float(getattr(self.augment, "crop_scale_min", 0.8))
            crop_scale_max = float(getattr(self.augment, "crop_scale_max", 1.0))
            crop_width_prob = float(getattr(self.augment, "crop_width_prob", 0.5))

        if crop_prob <= 0.0 or random.random() > crop_prob:
            return image, mask

        _, height, width = image.shape
        min_scale = max(1e-3, min(crop_scale_min, crop_scale_max))
        max_scale = min(1.0, max(crop_scale_min, crop_scale_max))
        crop_scale = random.uniform(min_scale, max_scale)

        crop_h = height
        crop_w = width
        if random.random() < crop_width_prob:
            crop_w = max(1, min(width, int(round(width * crop_scale))))
        else:
            crop_h = max(1, min(height, int(round(height * crop_scale))))
        top = 0 if crop_h == height else random.randint(0, height - crop_h)
        left = 0 if crop_w == width else random.randint(0, width - crop_w)

        image = TF.resized_crop(
            image,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if mask is None:
            return image, None

        mask = TF.resized_crop(
            mask,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[height, width],
            interpolation=InterpolationMode.NEAREST,
        )
        mask = (mask > 0.5).float()
        return image, mask

    def _augment_pair(self, image0, image1, mask0=None, mask1=None):
        if self.augment is None:
            return image0, image1, mask0, mask1
        if isinstance(self.augment, dict):
            prob = float(self.augment.get("prob", 1.0))
            rotate_deg = float(self.augment.get("rotate_deg", 15.0))
            zoom_min = float(self.augment.get("zoom_min", 0.95))
            zoom_max = float(self.augment.get("zoom_max", 1.05))
        else:
            prob = float(getattr(self.augment, "prob", 1.0))
            rotate_deg = float(getattr(self.augment, "rotate_deg", 15.0))
            zoom_min = float(getattr(self.augment, "zoom_min", 0.95))
            zoom_max = float(getattr(self.augment, "zoom_max", 1.05))
        image0, mask0 = self._apply_random_crop(image0, mask=mask0)
        image1, mask1 = self._apply_random_crop(image1, mask=mask1)
        if random.random() <= prob:
            angle0 = random.uniform(-rotate_deg, rotate_deg)
            scale0 = random.uniform(zoom_min, zoom_max)
            image0, mask0 = self._apply_affine(image0, angle0, scale0, mask=mask0)
        if random.random() <= prob:
            angle1 = random.uniform(-rotate_deg, rotate_deg)
            scale1 = random.uniform(zoom_min, zoom_max)
            image1, mask1 = self._apply_affine(image1, angle1, scale1, mask=mask1)
        return image0, image1, mask0, mask1

    def _maybe_flip_pair_order(self, image0, image1, mask0=None, mask1=None):
        if self.flip_order_prob <= 0.0:
            return image0, image1, mask0, mask1
        if random.random() > self.flip_order_prob:
            return image0, image1, mask0, mask1
        return image1, image0, mask1, mask0

    def _parse_sample_spec(self, sample_spec):
        if (
            isinstance(sample_spec, (tuple, list))
            and len(sample_spec) == 2
            and isinstance(sample_spec[0], (int, np.integer))
        ):
            idx = int(sample_spec[0])
            img_size = normalize_target_size(sample_spec[1])
            return idx, img_size
        return int(sample_spec), self.img_size

    def _resolve_dataset_index(self, idx):
        for dataset_idx, cumulative_len in enumerate(self.cumulative_lengths):
            if idx < cumulative_len:
                return dataset_idx
        raise IndexError(f"Pair index out of range: {idx}")

    def _resolve_pair_paths(self, idx):
        dataset_idx = self._resolve_dataset_index(idx)
        pair_info = self.pairs_info[idx]
        name0, name1 = pair_info[:2]
        img_name0 = osp.normpath(osp.join(self.image_dir[dataset_idx], str(name0)))
        img_name1 = osp.normpath(osp.join(self.image_dir[dataset_idx], str(name1)))
        return img_name0, img_name1

    def _get_cached_image_size(self, path):
        size = self._image_size_cache.get(path)
        if size is None:
            size = read_image_size(path)
            self._image_size_cache[path] = size
        return size

    @staticmethod
    def _estimate_padding_ratio(image_size, target_size):
        target_w, target_h = normalize_target_size(target_size)
        width, height = image_size
        scale = min(target_w / float(width), target_h / float(height))
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        valid_area = float(resized_w * resized_h)
        target_area = float(target_w * target_h)
        return 1.0 - (valid_area / target_area)

    def _normalize_resolution_key(self, resolutions):
        return tuple(normalize_target_size(resolution) for resolution in resolutions)

    def choose_best_resolution_for_index(self, idx, resolutions):
        idx = int(idx)
        normalized = self._normalize_resolution_key(resolutions)
        if not normalized:
            raise ValueError("choose_best_resolution_for_index requires at least one candidate.")
        if idx in self._invalid_pair_indices:
            return normalized[0]

        cache = self._resolution_choice_cache.setdefault(normalized, {})
        cached = cache.get(idx)
        if cached is not None:
            return cached

        try:
            img_name0, img_name1 = self._resolve_pair_paths(idx)
            image_sizes = (
                self._get_cached_image_size(img_name0),
                self._get_cached_image_size(img_name1),
            )
        except Exception:
            self._invalid_pair_indices.add(idx)
            return normalized[0]

        best_resolution = normalized[0]
        best_key = None
        for resolution in normalized:
            pad_ratio_sum = 0.0
            for image_size in image_sizes:
                pad_ratio_sum += self._estimate_padding_ratio(image_size, resolution)
            mean_pad_ratio = pad_ratio_sum / len(image_sizes)
            area = resolution[0] * resolution[1]
            candidate_key = (mean_pad_ratio, -area)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_resolution = resolution

        cache[idx] = best_resolution
        return best_resolution

    def choose_best_resolution(self, indices, resolutions):
        normalized = list(self._normalize_resolution_key(resolutions))
        if not normalized:
            raise ValueError("choose_best_resolution requires at least one candidate.")

        best_resolution = normalized[0]
        best_key = None
        for resolution in normalized:
            pad_ratio_sum = 0.0
            image_count = 0
            for idx in indices:
                idx = int(idx)
                if idx in self._invalid_pair_indices:
                    continue
                try:
                    img_name0, img_name1 = self._resolve_pair_paths(idx)
                    pad_ratio_sum += self._estimate_padding_ratio(
                        self._get_cached_image_size(img_name0),
                        resolution,
                    )
                    pad_ratio_sum += self._estimate_padding_ratio(
                        self._get_cached_image_size(img_name1),
                        resolution,
                    )
                except Exception:
                    self._invalid_pair_indices.add(idx)
                    continue
                image_count += 2
            if image_count == 0:
                continue
            mean_pad_ratio = pad_ratio_sum / max(1, image_count)
            area = resolution[0] * resolution[1]
            candidate_key = (mean_pad_ratio, -area)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_resolution = resolution
        return best_resolution

    def _make_numpy_rng(self, idx):
        if self.seed is not None:
            return np.random.default_rng(self.seed + int(idx))
        return np.random.default_rng()

    def __getitem__(self, idx):
        idx, img_size = self._parse_sample_spec(idx)
        if idx in self._invalid_pair_indices:
            return self.__getitem__(((idx + 1) % len(self), img_size))
        if len(self.pairs_info[idx]) == 4:
            name0, name1, label, _ = self.pairs_info[idx]
        else:
            name0, name1, label, _, _ = self.pairs_info[idx]
        try:
            img_name0, img_name1 = self._resolve_pair_paths(idx)

            rng = self._make_numpy_rng(idx)

            
            if self.mode in {"pad", "dust3r_like"}:
                image0, image1, mask0, mask1 = read_images(
                    img_name0,
                    img_name1,
                    img_size,
                    mode=self.mode,
                    return_mask=True,
                    rng=rng,
                    aug_crop=self.aug_crop,
                    patch_size=self.patch_size,
                )
            else:
                image0, image1 = read_images(
                    img_name0,
                    img_name1,
                    img_size,
                    mode=self.mode,
                    rng=rng,
                    patch_size=self.patch_size,
                )
            if isinstance(image0, np.ndarray):
                image0 = torch.from_numpy(image0)
            else:
                image0_cropped = torch.from_numpy(image0[0])
                image0_resized = torch.from_numpy(image0[1])
            if isinstance(image1, np.ndarray):
                image1 = torch.from_numpy(image1)
            else:
                image1_cropped = torch.from_numpy(image1[0])
                image1_resized = torch.from_numpy(image1[1])

            mask0_t = None
            mask1_t = None
            if self.mode in {"pad", "dust3r_like"}:
                mask0_t = torch.from_numpy(mask0)
                mask1_t = torch.from_numpy(mask1)
            if self.mode != "both":
                image0, image1, mask0_t, mask1_t = self._augment_pair(
                    image0, image1, mask0=mask0_t, mask1=mask1_t
                )
                image0, image1, mask0_t, mask1_t = self._maybe_flip_pair_order(
                    image0, image1, mask0=mask0_t, mask1=mask1_t
                )
                images = torch.cat([image0.unsqueeze(0), image1.unsqueeze(0)], dim=0)
                target_size = torch.tensor(
                    [images.shape[-1], images.shape[-2]],
                    dtype=torch.int32,
                )
                data = {
                    'images': images,  # (2, 3, h, w)
                    'gt': int(label),
                    'pair_idx': int(idx),
                    'target_size': target_size,
                }
            else:
                image0_cropped, image1_cropped, mask0_t, mask1_t = self._augment_pair(
                    image0_cropped, image1_cropped, mask0=mask0_t, mask1=mask1_t
                )
                image0_resized, image1_resized, _, _ = self._augment_pair(
                    image0_resized, image1_resized
                )
                image0_cropped, image1_cropped, mask0_t, mask1_t = self._maybe_flip_pair_order(
                    image0_cropped, image1_cropped, mask0=mask0_t, mask1=mask1_t
                )
                image0_resized, image1_resized, _, _ = self._maybe_flip_pair_order(
                    image0_resized, image1_resized
                )
                images = torch.cat([image0_cropped.unsqueeze(0), image1_cropped.unsqueeze(0)], dim=0)
                images_resized = torch.cat([image0_resized.unsqueeze(0), image1_resized.unsqueeze(0)], dim=0)
                data = {
                    'images': images,  # (2, 3, h, w)
                    'images_resized': images_resized,
                    'gt': int(label),
                    'pair_idx': int(idx),
                    'target_size': torch.tensor([img_size[0], img_size[1]], dtype=torch.int32),
                }
            
            if self.mode in {"pad", "dust3r_like"}:
                masks = torch.stack([mask0_t, mask1_t], dim=0)
                data["masks"] = masks
        except Exception as e:
            self._invalid_pair_indices.add(idx)
            print(
                f"Error loading pair {idx} with images {img_name0} and {img_name1} "
                f"at size {img_size}: {e}"
            )
            return self.__getitem__(((idx + 1) % len(self), img_size))

        return data

def get_datasets(cfg):
    train_img_size = getattr(cfg.train, "img_size", None)
    if train_img_size is None:
        train_resolutions = getattr(cfg.train, "resolutions", None)
        if train_resolutions:
            train_img_size = train_resolutions[0]

    test_img_size = getattr(cfg.test, "img_size", None)
    if test_img_size is None:
        test_resolutions = getattr(cfg.test, "resolutions", None)
        if test_resolutions:
            test_img_size = test_resolutions[0]

    train_augment = getattr(cfg.train, "augment", None)
    tr_dataset = DoppelgangersDataset(
                cfg.train.image_dir,
                cfg.train.pair_path,
                img_size=train_img_size if train_img_size is not None else 504,
                mode=getattr(cfg.train, "mode", "pad"),
                phase='Train',
                augment=train_augment,
                flip_order_prob=getattr(cfg.train, "flip_order_prob", None),
                aug_crop=getattr(cfg.train, "aug_crop", 0),
                seed=getattr(cfg.train, "seed", None),
                patch_size=getattr(cfg.train, "patch_size", 14))
    te_dataset = DoppelgangersDataset(
                cfg.test.image_dir,
                cfg.test.pair_path,
                img_size=test_img_size if test_img_size is not None else 504,
                mode=getattr(cfg.test, "mode", "pad"),
                phase='Test',
                seed=getattr(cfg.test, "seed", None),
                patch_size=getattr(cfg.test, "patch_size", 14))

    return tr_dataset, te_dataset

if __name__ == "__main__":
    pass
