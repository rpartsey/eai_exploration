from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.rollout_storage import (  # noqa: F401.
    RolloutStorage,
)
from habitat_baselines.common.storage import Storage
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetNet,
    PointNavResNetPolicy,
)
from habitat_baselines.rl.hrl.hierarchical_policy import (  # noqa: F401.
    HierarchicalPolicy,
)
from habitat_baselines.rl.ppo.agent_access_mgr import AgentAccessMgr
from habitat_baselines.rl.ppo.policy import NetPolicy, Policy
from habitat_baselines.rl.ppo.ppo import PPO
from habitat_baselines.rl.ppo.updater import Updater

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat_baselines.rl.ppo.single_agent_access_mgr import SingleAgentAccessMgr as SingleAgentAccessMgrBase


@baseline_registry.register_agent_access_mgr
class SingleAgentAccessMgr(SingleAgentAccessMgrBase):
    def __init__(
        self,
        config: "DictConfig",
        env_spec: EnvironmentSpec,
        is_distrib: bool,
        device,
        resume_state: Optional[Dict[str, Any]],
        num_envs: int,
        percent_done_fn: Callable[[], float],
        lr_schedule_fn: Optional[Callable[[float], float]] = None,
    ):
        super().__init__(config, env_spec, is_distrib, device, resume_state, num_envs, percent_done_fn, lr_schedule_fn)

    def _init_policy_and_updater(self, lr_schedule_fn, resume_state):
        self._actor_critic = self._create_policy()
        self._updater = self._create_updater(self._actor_critic)

        if self._updater.optimizer is None:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = LambdaLR(
                optimizer=self._updater.optimizer,
                lr_lambda=lambda _: lr_schedule_fn(self._percent_done_fn()),
            )
        if resume_state is not None:
            self._actor_critic.load_state_dict(resume_state["state_dict"])
            self._updater.optimizer.load_state_dict(
                resume_state["optim_state"]
            )
        self._policy_action_space = self._actor_critic.get_policy_action_space(
            self._env_spec.action_space
        )

    def load_state_dict(self, state: Dict) -> None:
        self._actor_critic.load_state_dict(state["state_dict"])
        if self._updater is not None:
            if "optim_state" in state:
                self._updater.optimizer.load_state_dict(state["optim_state"])
            if "lr_sched_state" in state:
                self._lr_scheduler.load_state_dict(state["lr_sched_state"][0])
