from habitat.config.default_structured_configs import register_hydra_plugin, HabitatConfigPlugin
from habitat_baselines.config.default_structured_configs import HabitatBaselinesConfigPlugin
from habitat_baselines.run import main

from config import MyConfigPlugin

# registration:
import config.default_structured_configs  # noqa
import habitat_extensions.habitat_lab.tasks.exp.task # noqa
import habitat_extensions.habitat_lab.tasks.exp.measures # noqa
import habitat_extensions.habitat_lab.tasks.nav.measures # noqa
import habitat_extensions.habitat_lab.datasets.exploration_dataset # noqa
import habitat_extensions.habitat_lab.tasks.exp.nav # noqa 
import habitat_extensions.habitat_baselines.rl.ddppo.policy.resnet_policy # noqa 
import habitat_extensions.habitat_lab.config.default_structured_configs # noqa


if __name__ == "__main__":
    register_hydra_plugin(HabitatConfigPlugin)
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    register_hydra_plugin(MyConfigPlugin)
    main()
