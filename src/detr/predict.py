"""Single-image inference and visualization for DETR.

Loads a trained checkpoint, runs one image through the model, decodes the output with
:func:`detr.postprocess.postprocess`, keeps detections above a score threshold, and draws
the boxes and class labels onto the original image. Run as a module::

    python -m detr.predict --checkpoint runs/voc/checkpoint.pth --image path/to.jpg \\
        --num-classes 20 --num-queries 100 --dataset voc --threshold 0.3 --output pred.png

``--num-classes`` and ``--num-queries`` must match the checkpoint, and ``--dropout`` is 0
for inference.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch import Tensor
from torchvision.transforms import v2
from PIL import Image

from detr.dataset import VOCDetectionDataset, eval_transform
from detr.postprocess import postprocess
from detr.detr import DETR, DETROutputWithAuxOutputs
from detr.train import load_checkpoint


def main() -> None:
    """Load a checkpoint, run one image, and save an annotated visualization.

    Builds the model to match the checkpoint (``num_classes``/``num_queries`` must agree),
    preprocesses the image with the same transform as training, decodes the output, keeps
    detections scoring at least ``--threshold``, and writes the drawn image to ``--output``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--image")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--dataset", choices=["coco", "voc"], default="voc")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output", default="pred.png")
    args = parser.parse_args()

    transform = eval_transform(args.image_size)

    model = DETR(num_classes=args.num_classes, num_queries=args.num_queries, dropout=args.dropout)
    load_checkpoint(args.checkpoint, model)

    model.eval()

    with Image.open(args.image) as image:
        W, H = image.size
        disp = image.convert("RGB")
        image_t: Tensor = transform(disp)

    out: DETROutputWithAuxOutputs = model(image_t[None])
    target_sizes = torch.tensor([[H, W]])
    detections = postprocess(out.output, target_sizes)[0]

    # Filter out detections under the threshold
    keep = detections.scores >= args.threshold
    scores = detections.scores[keep]
    labels = detections.labels[keep]
    boxes = detections.boxes[keep]

    fig, ax = plt.subplots()

    ax.imshow(disp)
    # (N,) to N of single items + (N, 4) to N of (4,)
    for score, label, box in zip(scores.unbind(), labels.unbind(), boxes.unbind()):
        x1 = box[0].item()
        y1 = box[1].item()
        x2 = box[2].item()
        y2 = box[3].item()

        name = VOCDetectionDataset.LABEL_TO_NAME[int(label.item())] if args.dataset == "voc" else str(label.item())

        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1,
                     y2 - y1, linewidth=1, edgecolor="red", facecolor='none'))
        ax.text(x1, y1, f"{name}: {score:.2f}", color="red", fontsize=8, bbox={
            "facecolor": "white",
            "alpha": 0.5,
            "pad": 0,
            "edgecolor": "none",
        })

    ax.axis("off")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
