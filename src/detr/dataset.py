"""Detection dataset and batching for DETR.

Adapts a COCO-format dataset into the target contract the model and criterion expect:
each item is an image tensor and a dict ``{"labels": (n,), "boxes": (n, 4)}`` with boxes
in normalized ``cxcywh``. Object counts vary per image, so :func:`collate_fn` stacks the
images but keeps the targets as a length-``B`` list rather than a single tensor.

This is the v1 pipeline: every image is resized to a fixed square size so images stack
directly and no padding mask is needed. Variable-size images with attention padding masks
are a later pass.
"""

from pathlib import Path

from PIL.Image import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.datasets import CocoDetection


def collate_fn(batch: list[tuple[Tensor, dict]]) -> tuple[Tensor, list[dict]]:
    """Assemble per-image samples into one batch.

    Args:
        batch: list of ``(image, target)`` pairs, where every image is the same
            ``(3, S, S)`` size and ``target`` is a dict with ``labels`` and ``boxes``.

    Returns:
        ``(images, targets)`` where ``images`` is ``(B, 3, S, S)`` and ``targets`` is the
        length-``B`` list of target dicts, left variable-length because the object count
        differs per image.
    """
    images = [img for img, _ in batch]
    targets = [target for _, target in batch]

    return torch.stack(images), targets


class DetectionDataset(Dataset):
    """COCO detection dataset yielding DETR-format targets.

    Wraps :class:`torchvision.datasets.CocoDetection` and converts each sample into an
    image tensor plus a target dict. COCO category ids are sparse, so they are remapped to
    a contiguous ``0..num_classes - 1`` range for the classification head.

    Args:
        root: directory of images.
        annFile: path to the COCO annotation JSON.
        image_size: side length the images are resized to. All images become
            ``image_size x image_size`` so a batch stacks without padding.

    Attributes:
        cat_id_to_label: maps a sparse COCO category id to its contiguous label.
        label_to_name: maps a contiguous label back to its category name.
    """

    def __init__(
        self,
        root: str | Path,
        annFile: str,
        image_size: int = 512
    ):
        self.coco = CocoDetection(root, annFile)
        # SPARSE list of integers from [1, 90]
        cat_ids = sorted(self.coco.coco.getCatIds())
        # Maps coco id to contiguous integer labels from [0, 79]
        self.cat_id_to_label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        # Reverse mapping
        self.label_to_name = {i: cat["name"] for i, cat in enumerate(self.coco.coco.loadCats(cat_ids))}

        self.image_size = image_size
        self.transform = v2.Compose([
            v2.Resize((self.image_size, self.image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, i: int) -> tuple[Tensor, dict]:
        """Load one sample as ``(image, target)``.

        Crowd annotations and zero-area boxes are dropped, boxes are converted to
        normalized ``cxcywh`` relative to the original image, category ids are remapped to
        contiguous labels, and the image is resized and normalized with ImageNet
        statistics.

        Args:
            i: sample index.

        Returns:
            ``(image, target)`` where ``image`` is ``(3, image_size, image_size)`` and
            ``target`` is ``{"labels": (n,), "boxes": (n, 4)}``. An image whose
            annotations are all filtered out yields empty ``(0,)`` and ``(0, 4)`` tensors.
        """
        coco_item: tuple[Image, list[dict]] = self.coco[i]
        pil_image, annotations = coco_item

        W, H = pil_image.size

        # a["bbox"][2], [3] are w, h; drop crowd and zero/negative-size boxes
        annotations = [a for a in annotations if a['iscrowd'] == 0 and a["bbox"][2] > 0 and a["bbox"][3] > 0]

        # Convert to cxcywh
        boxes = []
        for a in annotations:
            x, y, w, h = a["bbox"]
            boxes.append([
                (x + w / 2) / W,
                (y + h / 2) / H,
                w / W,
                h / H
            ])

        # Convert to tensor with shape (m, 4), with values clamped between 0 and 1
        boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp(0, 1)
        # Convert to tensor with shape (m,)
        labels_t = torch.tensor([self.cat_id_to_label[a["category_id"]] for a in annotations], dtype=torch.long)
        # Get transformed image from PIL image
        image_t: Tensor = self.transform(pil_image)

        return image_t, {"labels": labels_t, "boxes": boxes_t}
