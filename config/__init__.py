#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os.path as osp
from typing import Optional

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig

from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default import get_config as _habitat_baselines_get_config


_MY_CFG_DIR = osp.dirname(inspect.getabsfile(inspect.currentframe()))
# Habitat baselines config directory inside the installed package.
# Used to access default predefined configs.
# This is equivalent to doing osp.dirname(osp.abspath(__file__))
DEFAULT_CONFIG_DIR = "config/"
CONFIG_FILE_SEPARATOR = ","


class MyConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(
            provider="eai_rearrangement",
            path="pkg://config/",
        )


def get_config(
    config_path: str,
    overrides: Optional[list] = None,
    configs_dir: str = _MY_CFG_DIR,
) -> DictConfig:
    """
    Returns config object composed of configs from yaml file (config_path) and overrides.

    :param config_path: path to the yaml config file.
    :param overrides: list of config overrides. For example, :py:`overrides=["habitat_baselines.trainer_name=ddppo"]`.
    :param configs_dir: path to the config files root directory (defaults to :ref:`_BASELINES_CFG_DIR`).
    :return: composed config object.
    """
    register_hydra_plugin(MyConfigPlugin)
    cfg = _habitat_baselines_get_config(config_path, overrides, configs_dir)

    return cfg
