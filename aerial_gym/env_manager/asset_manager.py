from typing import Any
import torch.nn.functional as F
from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("asset_manager")
logger.setLevel("DEBUG")


class AssetManager:
    def __init__(self, global_tensor_dict, num_keep_in_env):
        self.init_tensors(global_tensor_dict, num_keep_in_env)

    def init_tensors(self, global_tensor_dict, num_keep_in_env):
        self.env_asset_state_tensor = global_tensor_dict["env_asset_state_tensor"]
        self.asset_min_state_ratio = global_tensor_dict["asset_min_state_ratio"]
        self.asset_max_state_ratio = global_tensor_dict["asset_max_state_ratio"]
        self.env_bounds_min = (
            global_tensor_dict["env_bounds_min"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.env_bounds_max = (
            global_tensor_dict["env_bounds_max"]
            .unsqueeze(1)
            .expand(-1, self.env_asset_state_tensor.shape[1], -1)
        )
        self.num_keep_in_env = num_keep_in_env
        self.terrain_height_grid = global_tensor_dict.get("terrain_height_grid", None)
        self.terrain_bounds = global_tensor_dict.get("terrain_bounds", None)

    def prepare_for_sim(self):
        self.reset(self.num_keep_in_env)
        logger.warning(f"Number of obstacles to be kept in the environment: {self.num_keep_in_env}")

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        pass

    def step(self, actions):
        pass
        # Implement this function if needed.
        # this functionality can do speciic things with the environment assets on stepping.
        # nothing really needs to be done for static environments.
        # if force needs to be applied, it should be done in the other classes and it's
        # better to leave this class to manipulate the state tensors.

    def reset(self, num_obstacles_per_env):
        self.reset_idx(torch.arange(self.env_asset_state_tensor.shape[0]), num_obstacles_per_env)

    def reset_idx(self, env_ids, num_obstacles_per_env=0):
        if num_obstacles_per_env < self.num_keep_in_env:
            logger.info(
                "Number of obstacles required in the environment by the \
                  code is lesser than the minimum number of obstacles that the environment configuration specifies."
            )
            num_obstacles_per_env = self.num_keep_in_env

        sampled_asset_state_ratio = torch_rand_float_tensor(
            self.asset_min_state_ratio, self.asset_max_state_ratio
        )
        self.env_asset_state_tensor[env_ids, :, 0:3] = torch_interpolate_ratio(
            min=self.env_bounds_min,
            max=self.env_bounds_max,
            ratio=sampled_asset_state_ratio[..., 0:3],
        )[env_ids, :, 0:3]

        x_pos = self.env_asset_state_tensor[env_ids, :, 0]
        if self.terrain_height_grid is not None and self.terrain_bounds is not None:
            y_pos = self.env_asset_state_tensor[env_ids, :, 1]
            x_min, x_max, y_min, y_max = self.terrain_bounds
            norm_x = 2.0 * (x_pos - x_min) / (x_max - x_min) - 1.0
            norm_y = 2.0 * (y_pos - y_min) / (y_max - y_min) - 1.0
            grid = torch.stack((norm_y, norm_x), dim=-1).unsqueeze(2)
            terrain_z = (
                F.grid_sample(
                    self.terrain_height_grid[env_ids],
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .squeeze(1)
                .squeeze(-1)
            )
        else:
            # Adjust Z height of randomized obstacles (e.g. trees) to sit on the bottom_wall.urdf profile:
            # X in [-10, 0]: Z = 0
            # X in [0, 10]:  Z = X * tan(35 deg) = 0.7002 * X
            # X in [10, 20]: Z = 7.002
            # X in [20, 30]: Z = (30 - X) * tan(35 deg) = 0.7002 * (30 - X)
            # X in [30, 40]: Z = 0
            terrain_z = torch.zeros_like(x_pos)
            slope_tan = 0.7002075  # tan(35 deg)
            up_mask = (x_pos > 0.0) & (x_pos < 10.0)
            top_mask = (x_pos >= 10.0) & (x_pos <= 20.0)
            down_mask = (x_pos > 20.0) & (x_pos < 30.0)
            terrain_z[up_mask] = x_pos[up_mask] * slope_tan
            terrain_z[top_mask] = 7.002
            terrain_z[down_mask] = (30.0 - x_pos[down_mask]) * slope_tan

        is_movable_obstacle = (
            self.asset_min_state_ratio[env_ids, :, 0] != self.asset_max_state_ratio[env_ids, :, 0]
        )
        self.env_asset_state_tensor[env_ids, :, 2] = torch.where(
            is_movable_obstacle,
            self.env_asset_state_tensor[env_ids, :, 2] + terrain_z,
            self.env_asset_state_tensor[env_ids, :, 2],
        )

        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_asset_state_ratio[env_ids, :, 3:6]
        )
        # put those obstacles not needed in the environment outside
        self.env_asset_state_tensor[env_ids, num_obstacles_per_env:, 0:3] = -1000.0
