# DETR

A from-scratch reimplementation of [DETR](https://arxiv.org/abs/2005.12872) (DEtection TRansformer) in PyTorch: ResNet-50 backbone, a hand-written transformer, learned object queries, Hungarian matching, and the set-prediction loss, assembled and trained end to end. It reaches **70.5 mAP@0.5** (44.4 COCO-style mAP) on Pascal VOC detection.

## Highlights

- **(Mostly) written from scratch.** Everything below the pretrained ResNet-50 is hand-written: attention, the transformer, positional encoding, the matcher, the loss, etc.
- **Hand-written multi-head attention**, stacked into a 6-layer encoder and 6-layer decoder. The decoder outputs all six layers, and the loss is applied at each one (the auxiliary losses).
- **Hungarian set matching.** Predictions and ground-truth boxes are matched one-to-one by solving a class + L1 + GIoU cost matrix with the Hungarian algorithm. The loss is then cross-entropy on the matched pairs plus L1 and GIoU on the boxes.
- **Variable-size images with padding masks.** Images keep their aspect ratio; each batch is padded to its largest height and width with a mask marking the padding. The mask feeds the positional encoding and attention, so padded pixels are ignored.
- **Image augmentation.** Random flip, multi-scale resize, and colour jitter. Boxes are carried as `tv_tensors.BoundingBoxes` so the same transforms apply to them, and any box pushed out of frame is dropped along with its label. On for training, off for eval.
- **No NMS.** Because the set loss makes each slot predict at most one object, inference drops the no-object class and takes the top class per slot. There is no non-maximum suppression step.

## Results

Pascal VOC, 20 classes, on a single AWS `g5.xlarge` (A10G, 24 GB):

- AdamW, split learning rate: backbone at `1e-5`, everything else at `1e-4`
- weight decay `1e-4`, gradient norm clipped at `0.1`, `StepLR` drop at epoch 200
- batch size 6, 300 epochs, about a day of wall-clock
- loss weights from the paper (`ce=1, bbox=5, giou=2`)
- best snapshot (epoch 294) picked via validation mAP

| Configuration | map@0.5 | mAP @[.5:.95] |
|---|---|---|
| Fixed-size, no augmentation (baseline) | 0.610 | 0.367 |
| **Aspect-preserving + padding masks + augmentation** | **0.705** | **0.444** |

Augmentation and masking added 0.095 to map@0.5, and localization tightened too (the `map / map_50` ratio rose from 0.60 to 0.63).

## Training

`train.py` is the entry point. Run it as a module so the `detr` package resolves:

```bash
# Overfit gate: memorize 10 images to check the pipeline is wired correctly
python -m detr.train --dataset voc --data /data/voc --num-classes 20 \
    --overfit 10 --dropout 0 --epochs 300

# Full VOC run (augmentation and masking on), with rolling checkpoints and S3 sync
python -m detr.train --dataset voc --data /data/voc --num-classes 20 \
    --batch-size 6 --epochs 300 --lr-drop 200 --lr 1e-4 --max-norm 0.1 \
    --num-queries 100 --dropout 0.1 --keep 10 \
    --output-dir runs/voc --s3-sync s3://bucket/voc

# Evaluate a checkpoint's mAP on the VOC val split
python -m detr.eval --data /data/voc --num-classes 20 \
    --checkpoint runs/voc/best.pth

# Run detection on a single image and draw the boxes
python -m detr.predict --checkpoint runs/voc/best.pth --num-classes 20 \
    --image path/to/image.jpg
```

Checkpoints save model, optimizer, and scheduler state. A rolling `--keep` window holds the last *N* snapshots plus a stable `checkpoint.pth` and can sync them to S3. COCO runs using `--dataset coco --ann instances.json`.

```bash
make test                    # tests
make typecheck               # types
```

## Layout

```
src/detr/
  box_ops.py           cxcywh <-> xyxy, area, IoU, generalized IoU
  attention.py         from-scratch MultiHeadAttention (oracle-tested vs nn.MultiheadAttention)
  transformer.py       encoder/decoder layers + Transformer (returns all decoder layers)
  position_encoding.py sinusoidal PositionEmbeddingSine, mask-driven
  backbone.py          ResNet-50 + FrozenBatchNorm2d + 1x1 projection to d_model
  detr.py              DETR assembly, MLP box head, output dataclasses
  matcher.py           HungarianMatcher (scipy linear_sum_assignment)
  criterion.py         SetCriterion: matched CE + L1 + GIoU, with aux losses
  dataset.py           VOC + COCO datasets, augmentation, variable-size collate + padding mask
  train.py             training loop, split-LR AdamW, StepLR, checkpoint/resume, overfit gate
  postprocess.py       no-NMS decode: drop no-object, top class, cxcywh -> abs xyxy
  eval.py              COCO-style mAP via torchmetrics / pycocotools
  predict.py           load a checkpoint, run one image, draw boxes + labels
tests/                 75 unit tests across 12 test files
```
