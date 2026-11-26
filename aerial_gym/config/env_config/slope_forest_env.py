# File: aerial_gym/config/env_config/forest_env_sloped.py

from aerial_gym.config.asset_config.env_object_config import (
    tree_asset_params,
    object_asset_params,
    terrain_asset_params,
    bottom_wall,
    top_wall,
    left_wall,
    right_wall,
    front_wall,
    back_wall,
)

import numpy as np


class ForestEnvSlopedCfg:
    class env:
        num_envs = 64
        num_env_actions = 4
        env_spacing = 5.0
        
        num_physics_steps_per_env_step_mean = 10
        num_physics_steps_per_env_step_std = 0
        
        render_viewer_every_n_steps = 1
        reset_on_collision = True
        collision_force_threshold = 0.005
        create_ground_plane = False  # Enable ground plane
        sample_timestep_for_latency = True
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False
        
        use_warp = True
        lower_bound_min = [-10.0, -10.0, 0.0]
        lower_bound_max = [-10.0, -10.0, 0.0]
        upper_bound_min = [40.0, 10.0, 10.0]
        upper_bound_max = [40.0, 10.0, 10.0]

    class env_config:
        include_asset_type = {
            "trees": True,
            "objects": False,
            "bottom_wall": True,
            "top_wall": False,
            "left_wall": False,
            "right_wall": False,
            "front_wall": False,
            "back_wall": False,
            "terrain": False,
        }

        asset_type_to_dict_map = {
            "trees": tree_asset_params,
            "objects": object_asset_params,
            "bottom_wall": bottom_wall,
            "top_wall": top_wall,
            "left_wall": left_wall,
            "right_wall": right_wall,
            "front_wall": front_wall,
            "back_wall": back_wall,
            "terrain": terrain_asset_params,
        }