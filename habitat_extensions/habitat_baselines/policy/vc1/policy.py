#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
from gym import spaces
from omegaconf import DictConfig
from torch import nn as nn
from vc_models.models.compression_layer import create_compression_layer

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy

from .vc1_encoder import VC1Encoder


class VC1Net(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        vc_model_name,
        hidden_size: int = 512,
        rnn_type: str = "GRU",
        num_recurrent_layers: int = 1,
        freeze_backbone: bool = False,
        freeze_batchnorm: bool = False,
        **kwargs
    ):
        super().__init__()
        self._observation_space = observation_space
        self._action_space = action_space
        self._input_channels: int = 3

        rnn_input_size = 0

        self.visual_encoder = VC1Encoder(vc_model_name=vc_model_name)

        (
            self.compression,
            _,
            compression_output_size,
        ) = create_compression_layer(
            self.visual_encoder.output_size,
            self.visual_encoder.backbone.final_spatial,
        )

        self.visual_fc = nn.Sequential(
            nn.Linear(compression_output_size, hidden_size),
            nn.ReLU(True),
        )

        rnn_input_size += hidden_size

        freeze_backbone = True
        freeze_batchnorm = True
        # freeze backbone
        if freeze_backbone:
            # Freeze all backbone
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False
            if freeze_batchnorm:
                self.visual_encoder.backbone = convert_frozen_batchnorm(
                    self.visual_encoder.backbone
                )

        # previous action embedding
        self.prev_action_embedding = nn.Embedding(self._action_space.n + 1, 32)
        rnn_input_size += 32

        # state encoder
        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        # save configuration
        self._hidden_size = hidden_size

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []

        # visual encoder
        # TODO rpartsey: check if this mormalization is needed
        rgb = observations["rgb"].permute(0, 3, 1, 2).float() / 255
        rgb = self.visual_encoder(rgb)
        rgb = self.compression(rgb)
        rgb = self.visual_fc(rgb)
        x.append(rgb)

        # TODO rpartsey: add GPS and compass sesnsor

        # previous action embedding
        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )
        x.append(prev_actions)

        # state encoder
        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )

        aux_loss_state: Dict[str, torch.Tensor] = {}

        return out, rnn_hidden_states, aux_loss_state


@baseline_registry.register_policy
class VC1NetPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        vc_model_name,
        hidden_size: int = 512,
        rnn_type: str = "GRU",
        num_recurrent_layers: int = 1,
        freeze_backbone: bool = False,
        freeze_batchnorm: bool = False,
        policy_config: DictConfig = None,
        aux_loss_config: Optional[DictConfig] = None,
        **kwargs
    ):
        super().__init__(
            VC1Net(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                vc_model_name=vc_model_name,
                hidden_size=hidden_size,
                rnn_type=rnn_type,
                num_recurrent_layers=num_recurrent_layers,
                freeze_backbone=freeze_backbone,
                freeze_batchnorm=freeze_batchnorm,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        observation_space: spaces.Dict,
        action_space,
        orig_action_space=None,
        agent_name="main_agent",
        **kwargs
    ):
        return cls(
            # Spaces
            observation_space=observation_space,
            action_space=action_space,
            # RNN
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            # Backbone
            vc_model_name="vc1_vitl",
            freeze_backbone=True,
            freeze_batchnorm=True,
            # Policy
            policy_config=config.habitat_baselines.rl.policy[agent_name],
        )


def convert_frozen_batchnorm(module):
    r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
    :class:`torch.nn.FrozenBatchNorm` layers.

    Args:
        module (nn.Module): module containing one or more :attr:`BatchNorm*D` layers
        process_group (optional): process group to scope synchronization,
            default is the whole world

    Returns:
        The original :attr:`module` with the converted :class:`torch.nn.FrozenBatchNorm`
        layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
        a new :class:`torch.nn.FrozenBatchNorm` layer object will be returned
        instead.

    Example::

        >>> # Network with nn.BatchNorm layer
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> module = torch.nn.Sequential(
        >>>            torch.nn.Linear(20, 100),
        >>>            torch.nn.BatchNorm1d(100),
        >>>          ).cuda()
        >>> # creating process group (optional)
        >>> # ranks is a list of int identifying rank ids.
        >>> ranks = list(range(8))
        >>> r1, r2 = ranks[:4], ranks[4:]
        >>> # Note: every rank calls into new_group for every
        >>> # process group created, even if that rank is not
        >>> # part of the group.
        >>> # xdoctest: +SKIP("distributed")
        >>> frozen_bn_module = convert_frozen_batchnorm(module)
    """
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = _FrozenBatchNorm(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_frozen_batchnorm(child))
    del module
    return module_output


class _FrozenBatchNorm(torch.nn.modules.batchnorm._NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            **factory_kwargs
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked
                    )
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        # if self.training:
        #     bn_training = True
        # else:
        #     bn_training = (self.running_mean is None) and (self.running_var is None)
        bn_training = False

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return torch.nn.functional.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var
            if not self.training or self.track_running_stats
            else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def _check_input_dim(self, input):
        return
