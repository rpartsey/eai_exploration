from habitat.config.default_structured_configs import register_hydra_plugin, HabitatConfigPlugin
from habitat_baselines.config.default_structured_configs import HabitatBaselinesConfigPlugin
from habitat_baselines.run import main

from config import get_config, MyConfigPlugin
from habitat_extensions.habitat_lab.tasks.exp.exp import ExplorationTask  # noqa (register ExplorationTask)

# registrations
import config.default_structured_configs  # noqa structured configs
import habitat_extensions.habitat_lab.tasks.exp.exp # noqa ExplorationVisitedLocationsReward
import habitat_extensions.habitat_lab.tasks.nav.nav # noqa TopDownMap
import habitat_extensions.habitat_lab.datasets.exploration_dataset # noqa register Exploration datasets


if __name__ == "__main__":
    register_hydra_plugin(HabitatConfigPlugin)
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    register_hydra_plugin(MyConfigPlugin)
    main()
