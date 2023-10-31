# registration:
import config.default_structured_configs  # noqa
import habitat_extensions.habitat_baselines.policy.vc1.policy
import habitat_extensions.habitat_lab.datasets.exploration_dataset  # noqa
import habitat_extensions.habitat_lab.tasks.exp.measures  # noqa
import habitat_extensions.habitat_lab.tasks.exp.nav  # noqa
import habitat_extensions.habitat_lab.tasks.exp.task  # noqa
import habitat_extensions.habitat_lab.tasks.nav.measures  # noqa
from config import MyConfigPlugin
from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from habitat_baselines.run import main

if __name__ == "__main__":
    register_hydra_plugin(HabitatConfigPlugin)
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    register_hydra_plugin(MyConfigPlugin)
    main()
