"""Set-prediction loss for DETR.

DETR emits a fixed-size set of predictions in no particular order, so the loss first
runs the Hungarian matcher to pair each ground-truth object with exactly one prediction.
It then applies a classification loss over every query and box regression losses
(L1 and generalized IoU) over the matched pairs only. The same losses
are applied to every intermediate decoder layer as auxiliary losses.
"""

from dataclasses import dataclass

from torch import nn, Tensor
import torch.nn.functional as F
import torch

from detr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from detr.detr import DETROutput, DETROutputWithAuxOutputs
from detr.matcher import HungarianMatcher


@dataclass(kw_only=True, frozen=True, slots=True)
class SetCriterionLoss:
    """Full loss output of :class:`SetCriterion`.

    Attributes:
        loss_ce: scalar classification loss for the final decoder layer.
        loss_box: box regression losses for the final decoder layer.
        loss_ce_aux: per-layer classification losses for the auxiliary decoder layers.
        loss_box_aux: per-layer box regression losses for the auxiliary decoder layers.
    """

    loss_ce: Tensor
    loss_box: SetCriterionBoxLoss
    loss_ce_aux: list[Tensor]
    loss_box_aux: list[SetCriterionBoxLoss]


@dataclass(kw_only=True, frozen=True, slots=True)
class SetCriterionBoxLoss:
    """Box regression losses for a single decoder layer.

    Attributes:
        loss_bbox: scalar L1 loss over the matched boxes, normalized by the number of
            ground-truth objects.
        loss_giou: scalar generalized-IoU loss over the matched boxes, normalized by the
            number of ground-truth objects.
    """

    loss_bbox: Tensor
    loss_giou: Tensor


class SetCriterion(nn.Module):
    """Computes the DETR set-prediction loss.

    Args:
        num_classes: number of object classes, excluding the no-object class. The
            no-object class occupies index ``num_classes`` in the logits.
        matcher: the :class:`HungarianMatcher` used to assign predictions to targets.
        weight_dict: maps a loss name to its scalar weight. Consumed by the training
            loop when it reduces the returned losses to a single scalar, not here.
        eos_coef: relative classification weight of the no-object class, down-weighting
            the many unmatched queries so they do not dominate the loss.

    Attributes:
        empty_weight: ``(num_classes + 1,)`` per-class weight for the classification
            cross-entropy, registered as a buffer so it follows the module across
            devices. Every entry is 1 except the no-object entry, which is ``eos_coef``.
    """

    empty_weight: Tensor

    def __init__(
            self, num_classes: int, matcher: HungarianMatcher, weight_dict: dict[str, float],
            eos_coef: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        # Down-weigh the "no object" class to prevent the model from
        # always outputing "no object"
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[num_classes] = eos_coef
        # Registers a tensor as module state but not a parameter
        # Allows the tensor to be on the same device as the module.
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: DETROutputWithAuxOutputs, targets: list[dict]) -> SetCriterionLoss:
        """Compute the loss over the final decoder layer and every auxiliary layer.

        Args:
            outputs: full model output, holding the final-layer predictions and the
                per-layer auxiliary predictions.
            targets: length-``B`` list of dicts with ``labels`` ``(n,)`` and ``boxes``
                ``(n, 4)`` in normalized ``cxcywh``.

        Returns:
            The classification and box losses for the final layer, plus a per-layer list
            of the same losses for the auxiliary layers. The ground-truth count used to
            normalize the box losses is shared across all layers and floored at 1. Every
            auxiliary layer is matched independently.
        """
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)

        # Get matching (prediction_indices, gt_indices)
        indices = self.matcher(outputs.output, targets)

        # Compute losses
        loss_ce = self.loss_labels(outputs.output, targets, indices)
        loss_box = self.loss_boxes(outputs.output, targets, indices, num_boxes)

        # Compute losses for auxilary outputs
        loss_ce_aux = []
        loss_box_aux = []

        for aux in outputs.aux_outputs:
            aux_indices = self.matcher(aux, targets)

            loss_ce_aux.append(self.loss_labels(aux, targets, aux_indices))
            loss_box_aux.append(self.loss_boxes(aux, targets, aux_indices, num_boxes))

        return SetCriterionLoss(
            loss_ce=loss_ce,
            loss_box=loss_box,
            loss_ce_aux=loss_ce_aux,
            loss_box_aux=loss_box_aux,
        )

    def _get_src_permutation_idx(self, indices: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """Flatten per-image matches into gather coordinates over a ``(B, N, ...)`` tensor.

        Args:
            indices: length-``B`` list of ``(prediction_indices, gt_indices)`` from the
                matcher.

        Returns:
            ``(src_index, batch_index)``, two 1-D long tensors of equal length running
            over every matched pair in the batch. ``batch_index[k]`` is the image and
            ``src_index[k]`` the query of the ``k``-th match, so indexing a
            ``(B, N, ...)`` tensor with ``[batch_index, src_index]`` selects the matched
            rows. The gt side of each pair is not used here.
        """
        # 1D tensor specifying the prediction indices
        src_index = torch.cat([pred_index for pred_index, _ in indices])
        # 1D tensor specifying the batch index that each prediction belongs to
        # full_like repeats the batch index with the same dimensions as pred_index
        batch_index = torch.cat([torch.full_like(pred_index, i) for i, (pred_index, _) in enumerate(indices)])

        return src_index, batch_index

    def loss_labels(self, output: DETROutput, targets: list[dict], indices: list[tuple[Tensor, Tensor]]) -> Tensor:
        """Classification loss over every query.

        Builds a ``(B, N)`` target-class tensor filled with the no-object index, scatters
        each matched ground-truth label into its assigned query, and applies weighted
        cross-entropy with :attr:`empty_weight`.

        Args:
            output: predictions for one decoder layer, with ``pred_logits`` ``(B, N,
                num_classes + 1)``.
            targets: length-``B`` list of dicts with ``labels`` ``(n,)``.
            indices: matcher output for this ``output``.

        Returns:
            Scalar cross-entropy loss over all ``B * N`` queries.
        """
        # (B, N, C + 1)
        pred_logits = output.pred_logits
        B, N = pred_logits.shape[:2]

        # Fill (B, N) tensor with "no object" class
        target_classes = torch.full((B, N), self.num_classes, dtype=torch.long, device=output.pred_logits.device)
        # Get the labels corresponding to ground truth indices
        # (m,)
        target_classes_obj = torch.cat([t["labels"][gt_index] for t, (_, gt_index) in zip(targets, indices)])

        # Overwrite the (batch, image) classes to the corresponding ground truth classes
        src_index, batch_index = self._get_src_permutation_idx(indices)
        target_classes[batch_index, src_index] = target_classes_obj

        # Transpose (B, N, C + 1) to # (B, C + 1, N) for cross entropy with (B, N)
        return F.cross_entropy(output.pred_logits.transpose(1, 2), target_classes, self.empty_weight)

    def loss_boxes(self, output: DETROutput, targets: list[dict],
                   indices: list[tuple[Tensor, Tensor]],
                   num_boxes: float) -> SetCriterionBoxLoss:
        """Box regression losses over the matched pairs.

        Gathers the matched predicted and ground-truth boxes, then computes the L1 loss
        and the generalized-IoU loss between corresponding pairs. Both are summed and
        divided by ``num_boxes``.

        Args:
            output: predictions for one decoder layer, with ``pred_boxes`` ``(B, N, 4)``
                in ``cxcywh``.
            targets: length-``B`` list of dicts with ``boxes`` ``(n, 4)`` in ``cxcywh``.
            indices: matcher output for this ``output``.
            num_boxes: normalizer, the total number of ground-truth boxes in the batch,
                floored at 1.

        Returns:
            The L1 and generalized-IoU losses for this layer.
        """
        # (B, N, 4)
        pred_boxes = output.pred_boxes

        # Get boxes corresponding to ground truth indices
        src_index, batch_index = self._get_src_permutation_idx(indices)
        # (m, 4)
        src_boxes = pred_boxes[batch_index, src_index]

        # (m, 4)
        target_boxes = torch.cat([t["boxes"][gt_index] for t, (_, gt_index) in zip(targets, indices)])

        # Compute losses
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum") / num_boxes
        # Only consider the GIoU between corresponding src and target boxes
        # Can be inefficient but the size (m) is small enough
        loss_giou = (1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )).sum() / num_boxes

        return SetCriterionBoxLoss(
            loss_bbox=loss_bbox,
            loss_giou=loss_giou
        )
