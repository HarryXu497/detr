"""ResNet-50 backbone with a channel projection.

Extracts the ``layer4`` feature map from a torchvision ResNet-50 (stride 32) and
projects its 2048 channels down to ``d_model`` with a 1x1 convolution. BatchNorm is
frozen (:class:`FrozenBatchNorm2d`) because DETR trains with small batches, so batch
statistics would be unstable.
"""

from torchvision.ops import FrozenBatchNorm2d
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torch import nn, Tensor


class Backbone(nn.Module):
    """ResNet-50 feature extractor.

    Args:
        d_model: number of output channels after the 1x1 projection.
        pretrained: load ImageNet-pretrained weights. Pass False in tests to run offline.
    """

    def __init__(self, d_model: int = 256, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)

        # Discard the last layer of ResNet
        self.body = IntermediateLayerGetter(backbone, return_layers={"layer4": "0"})
        self.proj = nn.Conv2d(2048, d_model, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, 3, H, W)`` -> ``(B, d_model, H/32, W/32)``."""
        features = self.body(x)["0"]
        return self.proj(features)