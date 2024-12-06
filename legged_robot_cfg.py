# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the legged robot."""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import sys
import os
##
# Configuration
##
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
"""Configuration for the legged robot robot."""
# PAN
LEGGED_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    # prim_path="/Root/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.join(CURRENT_DIR, "WalkBox3.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["Joint0_.*", "Joint1_.*"], 
            stiffness={
                "Joint0_0": 15,
                "Joint0_1": 15,
                "Joint0_2": 15,
                "Joint1_0": 15,
                "Joint1_1": 15,
                "Joint1_2": 15,
            },
            damping={
                "Joint0_0": 5.0,
                "Joint0_1": 5.0,
                "Joint0_2": 5.0,
                "Joint1_0": 5.0,
                "Joint1_1": 5.0,
                "Joint1_2": 5.0,
            },
            velocity_limit={
                "Joint0_0": 2,
                "Joint0_1": 2,
                "Joint0_2": 2,
                "Joint1_0": 2,
                "Joint1_1": 2,
                "Joint1_2": 2,
            }
        ),
    },
)