"""The full DETR model: backbone, transformer, object queries, and prediction heads.

Assembles the previously built modules into an end-to-end detector. The forward pass
extracts image features, adds positional encoding, flattens to a token sequence, runs
the transformer with learned object queries, and applies a classification head and a
box-regression head to every decoder layer.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from detr.backbone import Backbone
from detr.position_encoding import PositionEmbeddingSine
from detr.transformer import Transformer


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU between layers, used as the box head.

    Args:
        input_dim: input feature dimension.
        hidden_dim: hidden layer dimension.
        output_dim: output dimension.
        num_layers: total number of linear layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()

        hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            *hidden_layers,
            nn.Linear(hidden_dim, output_dim),
        ])

        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i != len(self.layers) - 1:
                x = self.relu(x)

        return x


@dataclass
class DETROutput:
    """Predictions from a single decoder layer.

    Attributes:
        pred_logits: ``(B, N, num_classes + 1)`` class logits; the last index is the
            no-object class.
        pred_boxes: ``(B, N, 4)`` boxes in normalized ``cxcywh`` format.
    """

    pred_logits: Tensor
    pred_boxes: Tensor


@dataclass(kw_only=True, frozen=True, slots=True)
class DETROutputWithAuxOutputs:
    """Full model output.

    Attributes:
        output: predictions from the final decoder layer.
        aux_outputs: predictions from each earlier decoder layer, for auxiliary losses.
    """

    output: DETROutput
    aux_outputs: list[DETROutput]


class DETR(nn.Module):
    """End-to-end DETR detector.

    Args:
        num_classes: number of object classes, excluding the no-object class.
        num_queries: number of object queries (detection slots).
        d_model: transformer feature dimension.
        nheads: number of attention heads.
        num_encoder_layers: number of transformer encoder layers.
        num_decoder_layers: number of transformer decoder layers.
        pretrained_backbone: load ImageNet-pretrained backbone weights.
    """

    def __init__(
        self,
        num_classes: int,
        num_queries: int = 100,
        d_model: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        pretrained_backbone: bool = True
    ) -> None:
        super().__init__()
        self.backbone = Backbone(d_model, pretrained=pretrained_backbone)
        self.pos_emb = PositionEmbeddingSine(d_model // 2)
        self.query_emb = nn.Embedding(num_queries, d_model)
        self.transformer = Transformer(d_model, nheads, num_encoder_layers, num_decoder_layers, dropout=dropout)
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.box_head = MLP(d_model, d_model, 4, 3)

    def forward(self, x: Tensor) -> DETROutputWithAuxOutputs:
        """``(B, 3, H, W)`` image batch -> predictions with auxiliary outputs."""
        features: Tensor = self.backbone(x)

        B, _, h, w = features.shape
        mask = torch.zeros(B, h, w, dtype=torch.bool, device=x.device)
        pos = self.pos_emb(mask)

        # (B, d_model, h, w) -> (B, d_model, h * w) -> (h * w, B, d_model)
        src = features.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2).permute(2, 0, 1)

        # (num_decoder_layers, N, B, D) -> (num_decoder_layers, B, N, D)
        hs: Tensor = self.transformer(src, pos, self.query_emb.weight)
        hs = hs.transpose(1, 2)

        # (num_decoder_layers, B, N, D) -> (num_decoder_layers, B, N, num_classes + 1)
        outputs_class: Tensor = self.class_head(hs)
        # (num_decoder_layers, B, N, D) -> (num_decoder_layers, B, N, 4)
        outputs_coord: Tensor = self.box_head(hs).sigmoid()

        # Return last layer bc it is derived from last decoder layer (most-refined)
        return DETROutputWithAuxOutputs(
            output=DETROutput(
                pred_logits=outputs_class[-1],  # (B, N, num_classes + 1)
                pred_boxes=outputs_coord[-1],  # (B, N, 4)
            ),
            aux_outputs=[
                DETROutput(pred_logits=c, pred_boxes=b)
                for c, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
        )
