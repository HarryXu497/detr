"""Detection datasets and batching for DETR.

Adapts a detection dataset into the target contract the model and criterion expect: each
item is an image tensor and a dict ``{"labels": (n,), "boxes": (n, 4)}`` with boxes in
normalized ``cxcywh``. Two sources are supported behind the same contract:
:class:`COCODetectionDataset` (COCO-format JSON) and :class:`VOCDetectionDataset` (Pascal
VOC XML).

Images are resized preserving aspect ratio, so within a batch they have different sizes.
:func:`collate_fn` pads them to a common size and returns a padding mask alongside the
stacked images; the targets stay a length-``B`` list because the object count differs per
image. The mask lets the model ignore the padded regions (see :meth:`DETR.forward`).
"""
from __future__ import annotations
from pathlib import Path

from PIL.Image import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.datasets import CocoDetection, VOCDetection

from detr.box_ops import box_xyxy_to_cxcywh


def eval_transform(image_size: int):
    """Deterministic image transform for evaluation and inference.

    Aspect-preserving resize (shortest side to ``image_size``, longest side capped at
    ``2 * image_size``) followed by tensor conversion and ImageNet normalization. Shared by
    the dataset's non-augmented path and :mod:`detr.predict` so training and inference
    preprocess images identically.
    """
    return v2.Compose([
        v2.Resize(image_size, max_size=image_size*2, antialias=True),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def collate_fn(batch: list[tuple[Tensor, dict]]) -> tuple[Tensor, Tensor, list[dict]]:
    """Pad variable-size images into one batch and build the padding mask.

    Args:
        batch: list of ``(image, target)`` pairs. Images may have different ``(3, H, W)``
            sizes (aspect-preserving resize); ``target`` is a dict with ``labels`` and
            ``boxes``.

    Returns:
        ``(images, mask, targets)``. ``images`` ``(B, 3, H_max, W_max)`` places each image
        top-left in a zero-padded canvas of the batch's max height and width. ``mask``
        ``(B, H_max, W_max)`` is ``True`` over padded pixels and ``False`` over real ones,
        so attention can ignore the padding. ``targets`` is the length-``B`` list of target
        dicts, kept variable-length because the object count differs per image.
    """
    # Each (3, h_i, w_i)
    images = [img for img, _ in batch]
    targets = [target for _, target in batch]
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    # (B, 3, h_max, w_max)
    batched = images[0].new_zeros(len(images), 3, max_h, max_w)
    # (B, h_max, w_max); all ones -> mask everything
    mask = torch.ones(len(images), max_h, max_w, dtype=torch.bool)

    for i, img in enumerate(images):
        _, h, w = img.shape

        # h x w chunk has the image, the rest is zeroes
        batched[i, :, :h, :w].copy_(img)
        # Unmask the portion with the image
        mask[i, :h, :w] = False

    return batched, mask, targets


class VOCDetectionDataset(Dataset):
    """Pascal VOC detection dataset yielding DETR-format targets.

    Wraps :class:`torchvision.datasets.VOCDetection` and converts each sample into an
    image tensor plus a target dict. VOC has a fixed set of 20 classes, mapped to a
    contiguous ``0..19`` range for the classification head; the no-object class is owned
    by the criterion, so it is not part of this mapping.

    Args:
        root: directory containing the ``VOCdevkit`` tree.
        image_set: which split to load (``"train"``, ``"trainval"``, or ``"val"``).
        image_size: shortest-side length the images are resized to, preserving aspect
            ratio (longest side capped at ``2 * image_size``).
        download: download and extract the VOC archive into ``root`` if absent. Off by
            default so repeated runs and tests do not re-fetch the ~2 GB archive.
        augment: apply random training augmentation (horizontal flip, resized crop, colour
            jitter) that transforms the image and boxes together. Off by default so the
            validation/inference path and the overfit gate stay deterministic.

    Attributes:
        NAME_TO_LABEL: maps a VOC class name to its contiguous label.
        LABEL_TO_NAME: maps a contiguous label back to its class name.
    """

    CLASSES = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    NAME_TO_LABEL = {n: i for i, n in enumerate(CLASSES)}
    LABEL_TO_NAME = {i: n for i, n in enumerate(CLASSES)}

    def __init__(
        self,
        root: str | Path,
        image_set: str = "train",
        image_size: int = 512,
        download: bool = False,
        augment: bool = False
    ):
        self.voc = VOCDetection(root, year="2012", image_set=image_set, download=download)

        self.image_size = image_size
        self.augment = augment
        if self.augment:
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(0.5),
                v2.RandomShortestSize(min_size=[384, 448, 512, 576, 640], max_size=1024, antialias=True),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                v2.SanitizeBoundingBoxes(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = eval_transform(image_size)

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, i: int) -> tuple[Tensor, dict]:
        """Load one sample as ``(image, target)``.

        Difficult objects and degenerate boxes are dropped, class names are mapped to
        contiguous labels, and the image is normalized with ImageNet statistics. When
        ``augment`` is set, the boxes are carried through the augmentation pipeline as a
        :class:`~torchvision.tv_tensors.BoundingBoxes` so flips and crops transform them in
        sync with the image (a crop that removes a box drops its label too); otherwise the
        image is simply resized. Boxes are returned as normalized ``cxcywh`` either way.

        Args:
            i: sample index.

        Returns:
            ``(image, target)`` where ``image`` is ``(3, H, W)`` — resized preserving
            aspect ratio, so ``H`` and ``W`` vary per image — and ``target`` is
            ``{"labels": (n,), "boxes": (n, 4)}``. An image whose objects are all filtered
            out yields empty ``(0,)`` and ``(0, 4)`` tensors.
        """
        voc_item: tuple[Image, dict] = self.voc[i]
        pil_image, target = voc_item

        W, H = pil_image.size
        objs: dict | list[dict] = target["annotation"]["object"]

        # single object fix
        if isinstance(objs, dict):
            objs = [objs]

        labels: list[int] = []

        boxes: list[list[float]] = []
        for obj in objs:
            box = obj["bndbox"]
            x_min = float(box["xmin"])
            y_min = float(box["ymin"])
            x_max = float(box["xmax"])
            y_max = float(box["ymax"])
            w = x_max - x_min
            h = y_max - y_min

            # Drop invalid boxes
            if w <= 0 or h <= 0:
                continue

            # Drop difficult objects
            if obj["difficult"] == "1":
                continue

            labels.append(self.NAME_TO_LABEL[obj["name"]])

            if self.augment:
                # Keep as xyxy
                boxes.append([
                    x_min,
                    y_min,
                    x_max,
                    y_max
                ])
            else:
                # Convert to cxcywh
                boxes.append([
                    ((x_min + x_max) / 2) / W,
                    ((y_max + y_min) / 2) / H,
                    w / W,
                    h / H
                ])

        if self.augment:
            # Convert to tensor with shape (m, 4)
            xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

            target = {
                "boxes": tv_tensors.BoundingBoxes(xyxy, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(H, W)), # pyright: ignore[reportCallIssue]
                "labels": torch.tensor(labels, dtype=torch.long)
            }

            # Transform image and boxes/labels together
            image_t, target = self.transform(pil_image, target)
            # Normalized xyxy
            boxes_t = target["boxes"].as_subclass(torch.Tensor) / self.image_size
            # Normalized cxcywh
            boxes_t = box_xyxy_to_cxcywh(boxes_t).clamp(0, 1)
            labels_t = target["labels"]
        else:
            # Convert to tensor with shape (m, 4), with values clamped between 0 and 1
            boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp(0, 1)
            # Convert to tensor with shape (m,)
            labels_t = torch.tensor(labels, dtype=torch.long)
            # Get transformed image from PIL image
            image_t: Tensor = self.transform(pil_image)

        return image_t, {"labels": labels_t, "boxes": boxes_t}


class COCODetectionDataset(Dataset):
    """COCO detection dataset yielding DETR-format targets.

    Wraps :class:`torchvision.datasets.CocoDetection` and converts each sample into an
    image tensor plus a target dict. COCO category ids are sparse, so they are remapped to
    a contiguous ``0..num_classes - 1`` range for the classification head.

    Args:
        root: directory of images.
        annFile: path to the COCO annotation JSON.
        image_size: shortest-side length the images are resized to, preserving aspect
            ratio (longest side capped at ``2 * image_size``).

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
            v2.Resize(self.image_size, max_size=image_size * 2, antialias=True),
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
