import torch
from torch import nn as nn
from vc_models.models.vit import model_utils


class VC1Encoder(nn.Module):
    def __init__(self, vc_model_name: str) -> None:
        super().__init__()
        assert vc_model_name in {
            model_utils.VC1_LARGE_NAME,
            model_utils.VC1_BASE_NAME,
        }

        (
            self.backbone,
            self.output_size,
            self.visual_transform,
            model_info,
        ) = model_utils.load_model(vc_model_name)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore
        # TODO rpartsey: check if this mormalization is needed
        x = x.float() / 255

        x = self.visual_transform(x)
        x = self.backbone(x)

        return x
