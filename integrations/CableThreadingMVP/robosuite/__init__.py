from robosuite.environments.base import make

# Manipulation environments
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.stack import Stack
from robosuite.environments.manipulation.nut_assembly import NutAssembly
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.environments.manipulation.door import Door
from robosuite.environments.manipulation.wipe import Wipe
from robosuite.environments.manipulation.tool_hang import ToolHang
from robosuite.environments.manipulation.cable_base import CableBaseEnv
from robosuite.environments.manipulation.cable_straighten import CableStraighten
from robosuite.environments.manipulation.cable_move_to_target import CableMoveToTarget
from robosuite.environments.manipulation.rmb_chain_hang_on_hook import RMBChainHangOnHook
from robosuite.environments.manipulation.deformable_ravens_cable_tasks import (
    CableRing,
    CableShape,
)
from robosuite.environments.manipulation.softgym_rope_tasks import RopeConfiguration, RopeFlatten
from robosuite.environments.manipulation.cable_pick_lift_place import CablePickLiftPlace
from robosuite.environments.manipulation.cable_atomic_test import CableAtomicTest
from robosuite.environments.manipulation.cable_routing import CableRouting
from robosuite.environments.manipulation.cable_threading import CableThreading
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.environments.manipulation.two_arm_peg_in_hole import TwoArmPegInHole
from robosuite.environments.manipulation.two_arm_handover import TwoArmHandover
from robosuite.environments.manipulation.two_arm_transport import TwoArmTransport

from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import (
    ALL_PART_CONTROLLERS,
    load_part_controller_config,
    ALL_COMPOSITE_CONTROLLERS,
    load_composite_controller_config,
)
from robosuite.robots import ALL_ROBOTS
from robosuite.models.grippers import ALL_GRIPPERS
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

try:
    import robosuite_models
except:
    ROBOSUITE_DEFAULT_LOGGER.warning(
        "Could not import robosuite_models. Some robots may not be available. "
        "If you want to use these robots, please install robosuite_models from "
        "source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install."
    )

try:
    from robosuite.examples.third_party_controller.mink_controller import WholeBodyMinkIK

except:
    ROBOSUITE_DEFAULT_LOGGER.warning(
        "Could not load the mink-based whole-body IK. Make sure you install related import properly (e.g. pip install mink==0.0.5), otherwise you will not be able to use the default IK controller setting for GR1 robot."
    )

__version__ = "1.5.2"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
