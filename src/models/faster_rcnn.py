from __future__ import annotations

from typing import Literal, Optional

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def build_faster_rcnn(
    num_classes: int,
    *,
    backbone: Literal["resnet50", "resnet101", "mobilenet_v3_large"] = "resnet50",
    pretrained: bool = True,
    trainable_backbone_layers: Optional[int] = 3,
) -> FasterRCNN:
    if backbone == "resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT" if pretrained else None,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    elif backbone == "resnet101":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights="DEFAULT" if pretrained else None,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    elif backbone == "mobilenet_v3_large":
        backbone_model = torchvision.models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
        backbone_model = backbone_model.features
        backbone_model.out_channels = 960
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
        model = FasterRCNN(
            backbone_model,
            num_classes=91,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


