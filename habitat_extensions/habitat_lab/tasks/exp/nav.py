#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# TODO, lots of typing errors in here

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import attr
import numpy as np
import quaternion
from gym import spaces

from habitat.tasks.nav.nav import PointGoalSensor

from habitat.core.registry import registry

@registry.register_sensor(name="PointGoalWithGPSCompassSensor")
class IntegratedPointGoalGPSAndCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            `goal_format` which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a `dimensionality` field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal_with_gps_compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.start_position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )
    

@registry.register_sensor(name="StepCountSensor")
class StepCountSensor(PointGoalSensor):
    r"""Sensor that returns the current step count."""
    cls_uuid: str = "step_count"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        step_count = kwargs['task'].measurements.measures['top_down_map']._step_count
        if step_count:
            return np.array([step_count, step_count], dtype=np.float32)
        return np.array([0, 0], dtype=np.float32)