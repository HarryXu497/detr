import torch
from detr.detr import DETR, MLP


def test_mlp_shape_and_layer_count():
    mlp = MLP(input_dim=256, hidden_dim=256, output_dim=4, num_layers=3)
    assert len(mlp.layers) == 3
    assert mlp(torch.randn(2, 5, 256)).shape == (2, 5, 4)


def test_output_shapes():
    model = DETR(num_classes=20, num_queries=100, num_encoder_layers=2,
                 num_decoder_layers=6, pretrained_backbone=False).eval()
    out = model(torch.randn(2, 3, 224, 224))
    assert out.output.pred_logits.shape == (2, 100, 21)   # num_classes + 1
    assert out.output.pred_boxes.shape == (2, 100, 4)


def test_boxes_are_normalized():
    model = DETR(num_classes=20, num_queries=100, num_encoder_layers=2,
                 num_decoder_layers=6, pretrained_backbone=False).eval()
    boxes = model(torch.randn(1, 3, 224, 224)).output.pred_boxes
    assert boxes.min() >= 0.0 and boxes.max() <= 1.0


def test_aux_outputs_one_per_intermediate_decoder_layer():
    ndec = 6
    model = DETR(num_classes=20, num_queries=100, num_encoder_layers=2,
                 num_decoder_layers=ndec, pretrained_backbone=False).eval()
    out = model(torch.randn(1, 3, 224, 224))
    assert len(out.aux_outputs) == ndec - 1  # final layer is the main output
    for aux in out.aux_outputs:
        assert aux.pred_logits.shape == (1, 100, 21)
        assert aux.pred_boxes.shape == (1, 100, 4)


def test_gradients_flow_end_to_end():
    model = DETR(num_classes=3, num_queries=20, num_encoder_layers=1,
                 num_decoder_layers=1, pretrained_backbone=False).train()
    out = model(torch.randn(1, 3, 128, 128))
    (out.output.pred_logits.sum() + out.output.pred_boxes.sum()).backward()
    assert model.backbone.proj.weight.grad is not None
    assert model.query_emb.weight.grad is not None
    assert model.class_head.weight.grad is not None
