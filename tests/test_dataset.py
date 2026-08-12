from __future__ import annotations
import json

import pytest
import torch
from PIL import Image

from detr.dataset import collate_fn, COCODetectionDataset, VOCDetectionDataset


def test_collate_stacks_fixed_size_images():
    batch = [
        (torch.randn(3, 512, 512), {"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)}),
        (torch.randn(3, 512, 512), {"labels": torch.tensor([2, 3]), "boxes": torch.rand(2, 4)}),
    ]
    images, targets = collate_fn(batch)
    assert images.shape == (2, 3, 512, 512)
    assert isinstance(targets, list) and len(targets) == 2
    assert targets[0]["labels"].tolist() == [1]
    assert targets[1]["boxes"].shape == (2, 4)


def test_targets_kept_as_variable_length_list():
    batch = [
        (torch.randn(3, 512, 512), {"labels": torch.tensor([], dtype=torch.long),
                                    "boxes": torch.zeros(0, 4)}),
        (torch.randn(3, 512, 512), {"labels": torch.tensor([5]), "boxes": torch.rand(1, 4)}),
    ]
    _, targets = collate_fn(batch)
    assert targets[0]["labels"].numel() == 0   # image with no objects is allowed
    assert targets[1]["labels"].numel() == 1


# --- COCODetectionDataset over a minimal on-disk COCO fixture ---------------------

# Sparse category ids, as in real COCO, to exercise the contiguous remap.
_CATEGORIES = [
    {"id": 1, "name": "person"},
    {"id": 5, "name": "airplane"},
    {"id": 90, "name": "toothbrush"},
]


@pytest.fixture(scope="module")
def coco_root(tmp_path_factory):
    """Build a tiny COCO dataset on disk: a 640x480 image with one valid box (plus a
    crowd box and a zero-width box that must be filtered), and a 480x640 image with no
    valid objects."""
    root = tmp_path_factory.mktemp("coco")
    Image.new("RGB", (640, 480), (120, 120, 120)).save(root / "a.jpg")
    Image.new("RGB", (480, 640), (30, 30, 30)).save(root / "b.jpg")

    ann = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "b.jpg", "width": 480, "height": 640},
        ],
        "annotations": [
            # valid person box on image 1: xywh pixels
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [100, 50, 200, 150], "area": 30000, "iscrowd": 0},
            # crowd box on image 1 -> dropped
            {"id": 2, "image_id": 1, "category_id": 5,
             "bbox": [0, 0, 100, 100], "area": 10000, "iscrowd": 1},
            # zero-width box on image 1 -> dropped
            {"id": 3, "image_id": 1, "category_id": 90,
             "bbox": [10, 10, 0, 50], "area": 0, "iscrowd": 0},
            # crowd box on image 2 -> image 2 ends up with no valid objects
            {"id": 4, "image_id": 2, "category_id": 1,
             "bbox": [5, 5, 50, 50], "area": 2500, "iscrowd": 1},
        ],
        "categories": _CATEGORIES,
    }
    (root / "ann.json").write_text(json.dumps(ann))
    return root


@pytest.fixture(scope="module")
def dataset(coco_root):
    return COCODetectionDataset(str(coco_root), str(coco_root / "ann.json"), image_size=512)


def test_category_remap_is_contiguous(dataset):
    assert dataset.cat_id_to_label == {1: 0, 5: 1, 90: 2}
    assert dataset.label_to_name == {0: "person", 1: "airplane", 2: "toothbrush"}


def test_image_is_resized_to_square(dataset):
    # Would have caught Resize(int), which preserves aspect ratio and yields non-square.
    image, _ = dataset[0]
    assert image.shape == (3, 512, 512)
    assert image.dtype == torch.float32


def test_getitem_filters_and_normalizes_boxes(dataset):
    _, target = dataset[0]
    # Only the single non-crowd, non-degenerate box survives.
    assert target["labels"].tolist() == [0]        # person -> contiguous 0
    assert target["labels"].dtype == torch.long
    assert target["boxes"].shape == (1, 4)
    # xywh [100,50,200,150] on 640x480 -> normalized cxcywh
    expected = torch.tensor([[200 / 640, 125 / 480, 200 / 640, 150 / 480]])
    assert torch.allclose(target["boxes"], expected, atol=1e-6)
    assert target["boxes"].min() >= 0.0 and target["boxes"].max() <= 1.0


def test_image_with_only_filtered_annotations_is_empty(dataset):
    _, target = dataset[1]
    assert target["labels"].shape == (0,)
    assert target["boxes"].shape == (0, 4)


def test_collate_over_real_items_stacks(dataset):
    images, targets = collate_fn([dataset[0], dataset[1]])
    assert images.shape == (2, 3, 512, 512)
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[1]["boxes"].shape == (0, 4)


# --- VOCDetectionDataset over a minimal on-disk VOC fixture -------------------

_ANN_A = """<annotation>
 <filename>a.jpg</filename>
 <size><width>640</width><height>480</height><depth>3</depth></size>
 <object><name>person</name><difficult>0</difficult>
  <bndbox><xmin>101</xmin><ymin>51</ymin><xmax>301</xmax><ymax>201</ymax></bndbox></object>
</annotation>"""

# Image b's only object is marked difficult, so it must end up with no targets.
_ANN_B = """<annotation>
 <filename>b.jpg</filename>
 <size><width>480</width><height>640</height><depth>3</depth></size>
 <object><name>dog</name><difficult>1</difficult>
  <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>55</xmax><ymax>55</ymax></bndbox></object>
</annotation>"""


@pytest.fixture(scope="module")
def voc_root(tmp_path_factory):
    """Build the VOCdevkit/VOC2012 tree torchvision expects: a 640x480 image with one
    valid person box, and a 480x640 image whose only object is difficult (dropped)."""
    root = tmp_path_factory.mktemp("voc")
    voc = root / "VOCdevkit" / "VOC2012"
    for sub in ("JPEGImages", "Annotations", "ImageSets/Main"):
        (voc / sub).mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (640, 480), (120, 120, 120)).save(voc / "JPEGImages" / "a.jpg")
    Image.new("RGB", (480, 640), (30, 30, 30)).save(voc / "JPEGImages" / "b.jpg")
    (voc / "Annotations" / "a.xml").write_text(_ANN_A)
    (voc / "Annotations" / "b.xml").write_text(_ANN_B)
    (voc / "ImageSets" / "Main" / "train.txt").write_text("a\nb\n")
    return root


@pytest.fixture(scope="module")
def voc_dataset(voc_root):
    return VOCDetectionDataset(str(voc_root), image_set="train", image_size=512)


def test_voc_class_map_is_contiguous_without_background(voc_dataset):
    # 20 VOC classes mapped to 0..19; the no-object class is the criterion's, not here.
    assert len(voc_dataset.classes) == 20
    assert "background" not in voc_dataset.name_to_label
    assert voc_dataset.name_to_label["aeroplane"] == 0
    assert voc_dataset.name_to_label["person"] == 14
    assert voc_dataset.name_to_label["tvmonitor"] == 19


def test_voc_image_is_resized_to_square(voc_dataset):
    image, _ = voc_dataset[0]
    assert image.shape == (3, 512, 512)
    assert image.dtype == torch.float32


def test_voc_getitem_parses_and_normalizes_boxes(voc_dataset):
    _, target = voc_dataset[0]
    assert target["labels"].tolist() == [14]        # person -> contiguous 14
    assert target["labels"].dtype == torch.long
    assert target["boxes"].shape == (1, 4)
    # VOC xyxy [101,51,301,201] on 640x480 -> normalized cxcywh
    expected = torch.tensor([[201 / 640, 126 / 480, 200 / 640, 150 / 480]])
    assert torch.allclose(target["boxes"], expected, atol=1e-6)
    assert target["boxes"].min() >= 0.0 and target["boxes"].max() <= 1.0


def test_voc_difficult_objects_are_dropped(voc_dataset):
    _, target = voc_dataset[1]
    assert target["labels"].shape == (0,)
    assert target["boxes"].shape == (0, 4)


def test_voc_collate_over_real_items_stacks(voc_dataset):
    images, targets = collate_fn([voc_dataset[0], voc_dataset[1]])
    assert images.shape == (2, 3, 512, 512)
    assert len(targets) == 2
    assert targets[0]["boxes"].shape == (1, 4)
    assert targets[1]["boxes"].shape == (0, 4)
