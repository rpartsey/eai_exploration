#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import random
from typing import Dict, Optional

import gym.spaces as spaces
import numba
import numpy as np
from omegaconf import DictConfig
import torch

import habitat
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.core.spaces import ActionSpace, EmptySpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

from habitat_baselines.utils.common import (
    batch_obs,
    get_num_actions,
    get_action_space_info
)

random_generator = np.random.RandomState()


@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    _seed_numba(seed)
    torch.random.manual_seed(seed)


def sample_random_seed():
    set_random_seed(random_generator.randint(2 ** 32))


class PPOAgent(Agent):
    def __init__(self, config: DictConfig) -> None:
        sensor_config = config.habitat.simulator.agents.main_agent.sim_sensors
        obs_space = dict([(sensor[:-7], spaces.Box(
                low=0,
                high=1,
                shape=(sensor_config[sensor]['height'], 
                       sensor_config[sensor]['width'], 
                       3 if sensor == 'rgb_sensor' else 1),
        )) for sensor in sensor_config])
        obs_space = spaces.Dict(obs_space)

        self.obs_transforms = get_active_obs_transforms(config)
        obs_space = apply_obs_transforms_obs_space(obs_space, self.obs_transforms)
        self.action_space = ActionSpace({
            "stop": EmptySpace(),
            "move_forward": EmptySpace(),
            "turn_left": EmptySpace(),
            "turn_right": EmptySpace(),
        })

        self.device = (
            torch.device("cuda:{}".format(config.pth_gpu_id))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.habitat_baselines.rl.ppo.hidden_size
        random_generator.seed(config.random_seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        policy = baseline_registry.get_policy(config.habitat_baselines.rl.policy.name)

        self.env_action_space = spaces.Discrete(
            get_num_actions(self.action_space)
        )

        self.actor_critic = policy.from_config(
            config,
            obs_space,
            self.env_action_space,
            orig_action_space=self.action_space,
        )

        self.actor_critic.to(self.device)

        if config.model_path:
            ckpt = torch.load(config.model_path, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(ckpt["state_dict"])

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating a random model."
            )

        policy_action_space = self.actor_critic.get_policy_action_space(
            self.env_action_space
        )
        self.policy_action_space, _ = get_action_space_info(
            policy_action_space
        )

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)

        self.prev_actions = torch.zeros(
            1,
            *self.policy_action_space,
            dtype=torch.long,
            device=self.device,
        )

    def act(self, observations: Observations) -> Dict[str, int]:
        sample_random_seed()
        batch = batch_obs([observations], device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        with torch.no_grad():
            action_data = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )

            self.test_recurrent_hidden_states = action_data.rnn_hidden_states

            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(action_data.actions)  # type: ignore

        return {'action': action_data.env_actions[0].item()}
