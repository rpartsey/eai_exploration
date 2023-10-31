import os

import hydra
import omegaconf
import torch
import vc_models
from omegaconf import open_dict
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
        ) = self._load_model(vc_model_name)

    def _load_model(self, model_name):
        """
        Loads a model from the vc_models package.
        Args:
            model_name (str): name of the model to load
        Returns:
            model (torch.nn.Module): the model
            embedding_dim (int): the dimension of the embedding
            transform (torchvision.transforms): the transform to apply to the image
            metadata (dict): the metadata of the model
        """
        models_filepath = os.path.dirname(os.path.abspath(vc_models.__file__))

        cfg_path = os.path.join(
            models_filepath, "conf", "model", f"{model_name}.yaml"
        )

        model_cfg = omegaconf.OmegaConf.load(cfg_path)
        self._override_model_cfg(model_cfg)

        # returns tuple of model, embedding_dim, transform, metadata
        return hydra.utils.call(model_cfg)

    def _override_model_cfg(self, model_cfg):
        with open_dict(model_cfg):
            if "model" in model_cfg.model:
                model = model_cfg.model.model
            else:
                model = model_cfg.model
            model.global_pool = False
            model.use_cls = False

    def forward(self, rgb: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore
        rgb_transformed = self.visual_transform(rgb)
        features = self.backbone(rgb_transformed)

        return features
