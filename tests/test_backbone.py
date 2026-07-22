import torch
from detr.backbone import Backbone


def test_output_channels_and_stride():
    net = Backbone(d_model=256, pretrained=False).eval()
    out = net(torch.randn(2, 3, 224, 224))
    assert out.shape[0] == 2
    assert out.shape[1] == 256          # projected to d_model
    assert out.shape[2] == 224 // 32    # stride-32 feature map
    assert out.shape[3] == 224 // 32


def test_handles_non_square_input():
    net = Backbone(d_model=256, pretrained=False).eval()
    out = net(torch.randn(1, 3, 224, 320))
    assert out.shape == (1, 256, 7, 10)


def test_projection_tracks_d_model():
    net = Backbone(d_model=128, pretrained=False).eval()
    out = net(torch.randn(1, 3, 256, 256))
    assert out.shape == (1, 128, 8, 8)  # 256/32 = 8, d_model = 128


def test_batchnorm_is_frozen():
    """FrozenBatchNorm2d has no trainable parameters, so no BatchNorm2d modules should
    remain and BN buffers should not require grad."""
    net = Backbone(d_model=256, pretrained=False)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules())
