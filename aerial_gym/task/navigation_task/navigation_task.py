from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from collections import deque
import os
import time
import torch
import numpy as np
import warp as wp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("navigation_task")


@wp.kernel
def raycast_ground_kernel(
    mesh_ids: wp.array(dtype=wp.uint64),
    robot_positions: wp.array(dtype=wp.vec3),
    ground_heights: wp.array(dtype=wp.float32),
    ground_clearances: wp.array(dtype=wp.float32),
    hit_flags: wp.array(dtype=wp.int32),
    max_distance: float,
):
    env_id = wp.tid()
    origin = robot_positions[env_id]
    direction = wp.vec3(0.0, 0.0, -1.0)

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face = int(0)

    if wp.mesh_query_ray(
        mesh_ids[env_id],
        origin,
        direction,
        max_distance,
        t,
        u,
        v,
        sign,
        normal,
        face,
    ):
        ground_heights[env_id] = origin[2] - t
        ground_clearances[env_id] = t
        hit_flags[env_id] = 1
    else:
        ground_heights[env_id] = 0.0
        ground_clearances[env_id] = 0.0
        hit_flags[env_id] = 0


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class NavigationTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp
        super().__init__(task_config)
        self.device = self.task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for navigation task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        # The navigation target is horizontal; altitude is controlled by the
        # ground-clearance reward instead.
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 2), device=self.device, requires_grad=False
        )

        if not getattr(self.sim_env, "use_warp", False):
            raise RuntimeError(
                "NavigationTask ground raycasting requires use_warp=True."
            )
        self.warp_mesh_ids = wp.array(
            self.sim_env.warp_env.warp_mesh_id_list,
            dtype=wp.uint64,
            device=self.device,
        )
        self.ground_height = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.ground_clearance = torch.zeros_like(self.ground_height)
        self.ground_hit = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )
        self.ground_ray_max_distance = float(
            getattr(self.task_config, "ground_ray_max_distance", 100.0)
        )
        self.target_ground_clearance = float(
            getattr(self.task_config, "target_ground_clearance", 1.5)
        )
        self.ground_height_reward_magnitude = float(
            torch.as_tensor(
                self.task_config.reward_parameters.get(
                    "ground_height_reward_magnitude", 2.0
                )
            ).item()
        )
        self.ground_height_reward_exponent = float(
            torch.as_tensor(
                self.task_config.reward_parameters.get(
                    "ground_height_reward_exponent", 2.0
                )
            ).item()
        )

        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)

        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        self.pos_error_vehicle_frame_prev = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.pos_error_vehicle_frame = torch.zeros_like(
            self.pos_error_vehicle_frame_prev
        )

        if self.task_config.vae_config.use_vae:
            self.vae_model = VAEImageEncoder(config=self.task_config.vae_config, device=self.device)
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        else:
            self.vae_model = lambda x: x

        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["target_position"] = self.target_position
        self.obs_dict["ground_height"] = self.ground_height
        self.obs_dict["ground_clearance"] = self.ground_clearance

        self.episode_clearance_sum = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.episode_clearance_steps = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.episode_reward_sum = torch.zeros(
            self.sim_env.num_envs, device=self.device, requires_grad=False
        )
        self.total_episodes_completed = 0
        self.plot_rolling_window = 100  # Matches Sample Factory stats_avg=100
        self.recent_episode_clearances = deque(maxlen=self.plot_rolling_window)
        self.recent_episode_rewards = deque(maxlen=self.plot_rolling_window)
        self.recent_episode_successes = deque(maxlen=self.plot_rolling_window)
        self.plot_episodes_x = []
        self.plot_clearance_y = []
        self.plot_reward_y = []
        self.plot_success_rate_y = []
        self.last_plot_save_time = 0.0
        self.clearance_plot_path = os.path.join(
            os.getcwd(), "avg_ground_clearance_training.png"
        )
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
                # "image_obs": Box(
                #     low=-1.0,
                #     high=1.0,
                #     shape=(1, 135, 240),
                #     dtype=np.float32,
                # ),
            }
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.task_config.action_space_dim,), dtype=np.float32
        )
        self.action_transformation_function = self.task_config.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            # "priviliged_obs": torch.zeros(
            #     (
            #         self.sim_env.num_envs,
            #         self.task_config.privileged_observation_space_dim,
            #     ),
            #     device=self.device,
            #     requires_grad=False,
            # ),
            # "collisions": torch.zeros(
            #     (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            # ),
            # "rewards": torch.zeros(
            #     (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            # ),
        }

        self.num_task_steps = 0

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.infos = {}
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        target_ratio = torch_rand_float_tensor(self.target_min_ratio, self.target_max_ratio)
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids, :2],
            max=self.obs_dict["env_bounds_max"][env_ids, :2],
            ratio=target_ratio[env_ids, :2],
        )
        self.episode_clearance_sum[env_ids] = 0.0
        self.episode_clearance_steps[env_ids] = 0.0
        self.episode_reward_sum[env_ids] = 0.0
        # logger.warning(f"reset envs: {env_ids}")
        return

    def render(self):
        return self.sim_env.render()

    def logging_sanity_check(self, infos):
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps * torch.ones_like(self.sim_env.sim_steps),
        )
        env_list_for_toc = (time_at_crash < 5).nonzero(as_tuple=False).squeeze(-1)
        crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
        success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
        timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)

        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
            )
        if torch.sum(torch.logical_and(successes, timeouts)) > 0:
            logger.critical("Success and timeout are occuring at the same time")
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(successes, timeouts))}"
            )
        if torch.sum(torch.logical_and(crashes, timeouts)) > 0:
            logger.critical("Crash and timeout are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, timeouts))}"
            )
        return

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + self.crashes_aggregate + self.timeouts_aggregate

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # clamp curriculum_level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, Curriculum progress fraction: {self.curriculum_progress_fraction}"
            )
            logger.warning(
                f"\nSuccess Rate: {success_rate}\nCrash Rate: {crash_rate}\nTimeout Rate: {timeout_rate}"
            )
            logger.warning(
                f"\nSuccesses: {self.success_aggregate}\nCrashes : {self.crashes_aggregate}\nTimeouts: {self.timeouts_aggregate}"
            )
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0

    def process_image_observation(self):
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        if self.task_config.vae_config.use_vae:
            self.image_latents[:] = self.vae_model.encode(image_obs)
        # # comments to make sure the VAE does as expected
        # decoded_image = self.vae_model.decode(self.image_latents[0].unsqueeze(0))
        # image0 = image_obs[0].cpu().numpy()
        # decoded_image0 = decoded_image[0].squeeze(0).cpu().numpy()
        # # save as .png with timestep
        # if not hasattr(self, "img_ctr"):
        #     self.img_ctr = 0
        # self.img_ctr += 1
        # import matplotlib.pyplot as plt
        # plt.imsave(f"image0{self.img_ctr}.png", image0, vmin=0, vmax=1)
        # plt.imsave(f"decoded_image0{self.img_ctr}.png", decoded_image0, vmin=0, vmax=1)

    def step(self, actions):
        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.

        transformed_action = self.action_transformation_function(actions)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        # logger.info(f"Curricluum Level: {self.curriculum_level}")

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        # successes are the sum of the environments which have reached the target (x, y) within a distance threshold
        dist_to_goal = torch.norm(
            self.target_position - self.obs_dict["robot_position"][:, :2], dim=1
        )
        reached_goal = dist_to_goal < 1.0
        timed_out = self.sim_env.sim_steps > self.task_config.episode_len_steps

        self.truncations[:] = torch.where(
            reached_goal | timed_out,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        successes = self.truncations * reached_goal
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash

        self.infos["dist_to_goal"] = dist_to_goal
        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations

        valid_clearance = torch.where(
            self.ground_hit > 0,
            self.ground_clearance,
            torch.zeros_like(self.ground_clearance),
        )
        self.episode_clearance_sum += valid_clearance
        self.episode_clearance_steps += (self.ground_hit > 0).to(
            self.episode_clearance_steps.dtype
        )
        self.episode_reward_sum += self.rewards
        ep_avg_clearance = torch.where(
            self.episode_clearance_steps > 0,
            self.episode_clearance_sum / self.episode_clearance_steps.clamp(min=1.0),
            self.ground_clearance,
        )
        self.infos["ground_clearance"] = self.ground_clearance.clone()
        self.infos["ep_avg_ground_clearance"] = ep_avg_clearance.clone()

        done_mask = (self.terminations > 0) | (self.truncations > 0)
        if torch.any(done_mask):
            finished_clearances = ep_avg_clearance[done_mask].detach().cpu().tolist()
            finished_rewards = self.episode_reward_sum[done_mask].detach().cpu().tolist()
            finished_successes = (
                (successes[done_mask] > 0).to(torch.float32).detach().cpu().tolist()
            )
            self.recent_episode_clearances.extend(finished_clearances)
            self.recent_episode_rewards.extend(finished_rewards)
            self.recent_episode_successes.extend(finished_successes)
            self.total_episodes_completed += len(finished_clearances)
            self.update_and_save_clearance_plot()

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        self.process_image_observation()
        self.post_image_reward_addition()
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def post_image_reward_addition(self):
        image_obs = 10.0 * self.obs_dict["depth_range_pixels"].squeeze(1)
        image_obs[image_obs < 0] = 10.0
        self.min_pixel_dist = torch.amin(image_obs, dim=(1, 2))
        self.rewards[self.terminations < 0] += -exponential_reward_function(
            4.0, 1.0, self.min_pixel_dist[self.terminations < 0]
        )

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        target_vector_world = torch.zeros_like(self.obs_dict["robot_position"])
        target_vector_world[:, :2] = (
            self.target_position - self.obs_dict["robot_position"][:, :2]
        )
        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            target_vector_world,
        )
        perturbed_vec_to_tgt = vec_to_tgt + 0.1 * 2 * (torch.rand_like(vec_to_tgt) - 0.5)
        perturbed_vec_to_tgt[:, 2] = 0.0
        dist_to_tgt = torch.norm(vec_to_tgt, dim=-1)
        perturbed_unit_vec_to_tgt = perturbed_vec_to_tgt / dist_to_tgt.clamp(min=1e-3).unsqueeze(1)
        self.task_obs["observations"][:, 0:3] = perturbed_unit_vec_to_tgt
        self.task_obs["observations"][:, 3] = dist_to_tgt
        # self.task_obs["observation"][:, 3] = self.infos["successes"]
        # self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_vehicle_orientation"]
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        perturbed_euler_angles = euler_angles + 0.1 * (torch.rand_like(euler_angles) - 0.5)
        self.task_obs["observations"][:, 4] = perturbed_euler_angles[:, 0]
        self.task_obs["observations"][:, 5] = perturbed_euler_angles[:, 1]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        if self.task_config.vae_config.use_vae:
            self.task_obs["observations"][:, 17:] = self.image_latents
        # self.task_obs["rewards"] = self.rewards
        # self.task_obs["terminations"] = self.terminations
        # self.task_obs["truncations"] = self.truncations

        # self.task_obs["image_obs"] = self.obs_dict["depth_range_pixels"]

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        target_vector_world = torch.zeros_like(robot_position)
        target_vector_world[:, :2] = self.target_position - robot_position[:, :2]
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, target_vector_world
        )
        self.update_ground_measurement()
        return compute_reward(
            self.pos_error_vehicle_frame,
            self.pos_error_vehicle_frame_prev,
            obs_dict["crashes"],
            obs_dict["robot_actions"],
            obs_dict["robot_prev_actions"],
            self.curriculum_progress_fraction,
            self.task_config.reward_parameters,
            self.ground_clearance,
            self.ground_hit,
            self.target_ground_clearance,
            self.ground_height_reward_magnitude,
            self.ground_height_reward_exponent,
        )

    def update_ground_measurement(self):
        """Cast a vertical ray from each robot and update ground measurements."""
        robot_positions = wp.from_torch(
            self.obs_dict["robot_position"], dtype=wp.vec3
        )
        ground_heights_wp = wp.from_torch(self.ground_height, dtype=wp.float32)
        ground_clearances_wp = wp.from_torch(self.ground_clearance, dtype=wp.float32)
        hit_flags_wp = wp.from_torch(self.ground_hit, dtype=wp.int32)

        wp.launch(
            kernel=raycast_ground_kernel,
            dim=self.sim_env.num_envs,
            inputs=[
                self.warp_mesh_ids,
                robot_positions,
                ground_heights_wp,
                ground_clearances_wp,
                hit_flags_wp,
                self.ground_ray_max_distance,
            ],
            device=self.device,
        )

    def update_and_save_clearance_plot(self):
        """Record rolling mean episode clearance, reward, and success rate and atomically save the training plot."""
        now = time.time()
        if self.last_plot_save_time != 0.0 and (now - self.last_plot_save_time) < 2.0:
            return
        if not self.recent_episode_clearances:
            return

        mean_clearance = float(np.mean(self.recent_episode_clearances))
        mean_reward = float(np.mean(self.recent_episode_rewards))
        mean_success_rate = 100.0 * float(np.mean(self.recent_episode_successes))

        self.plot_episodes_x.append(self.total_episodes_completed)
        self.plot_clearance_y.append(mean_clearance)
        self.plot_reward_y.append(mean_reward)
        self.plot_success_rate_y.append(mean_success_rate)

        fig, axes = plt.subplots(3, 1, figsize=(8, 10.5), sharex=True)

        # 1. Average Height Above Ground
        axes[0].plot(
            self.plot_episodes_x,
            self.plot_clearance_y,
            color="tab:blue",
            linewidth=1.8,
            label=f"Avg Ground Clearance (last {self.plot_rolling_window} eps)",
        )
        axes[0].axhline(
            self.target_ground_clearance,
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
            label=f"Target ({self.target_ground_clearance:.2f} m)",
        )
        axes[0].set_ylabel("Avg Height Above Ground (m)")
        axes[0].set_title("Average Height Above Ground During Training")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 2. Average Episode Reward
        axes[1].plot(
            self.plot_episodes_x,
            self.plot_reward_y,
            color="tab:green",
            linewidth=1.8,
            label=f"Avg Episode Reward (last {self.plot_rolling_window} eps)",
        )
        axes[1].set_ylabel("Avg Episode Reward")
        axes[1].set_title("Average Episode Reward During Training")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # 3. Average Success Rate
        axes[2].plot(
            self.plot_episodes_x,
            self.plot_success_rate_y,
            color="tab:purple",
            linewidth=1.8,
            label=f"Avg Success Rate % (last {self.plot_rolling_window} eps)",
        )
        axes[2].set_ylim(-5.0, 105.0)
        axes[2].set_xlabel("Number of Episodes")
        axes[2].set_ylabel("Success Rate (%)")
        axes[2].set_title("Average Success Rate During Training")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.tight_layout()

        tmp_path = f"{self.clearance_plot_path}.tmp.png"
        fig.savefig(tmp_path, dpi=120)
        plt.close(fig)
        os.replace(tmp_path, self.clearance_plot_path)
        self.last_plot_save_time = now


@torch.jit.script
def exponential_reward_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * torch.exp(-(value * value) * exponent)


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * (torch.exp(-(value * value) * exponent) - 1.0)


@torch.jit.script
def compute_reward(
    pos_error,
    prev_pos_error,
    crashes,
    action,
    prev_action,
    curriculum_progress_fraction,
    parameter_dict,
    ground_clearance,
    ground_hit,
    target_ground_clearance,
    ground_height_reward_magnitude,
    ground_height_reward_exponent,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor], Tensor, Tensor, float, float, float) -> Tuple[Tensor, Tensor]
    MULTIPLICATION_FACTOR_REWARD = 1.0 + (2.0) * curriculum_progress_fraction
    dist = torch.norm(pos_error[:, :2], dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error[:, :2], dim=1)
    pos_reward = exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )
    very_close_to_goal_reward = exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )

    getting_closer = prev_dist_to_goal - dist
    getting_closer_reward = torch.where(
        getting_closer > 0,
        parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
        2.0 * parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
    )

    distance_from_goal_reward = torch.clamp((40.0 - dist) / 40.0, min=0.0)
    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    yawrate_diff_penalty = exponential_penalty_function(
        parameter_dict["yawrate_action_diff_penalty_magnitude"],
        parameter_dict["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    action_diff_penalty = x_diff_penalty + yawrate_diff_penalty
    # absolute action penalty
    x_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["x_absolute_action_penalty_magnitude"],
        parameter_dict["x_absolute_action_penalty_exponent"],
        action[:, 0],
    )
    yawrate_absolute_penalty = curriculum_progress_fraction * exponential_penalty_function(
        parameter_dict["yawrate_absolute_action_penalty_magnitude"],
        parameter_dict["yawrate_absolute_action_penalty_exponent"],
        action[:, 3],
    )
    absolute_action_penalty = x_absolute_penalty + yawrate_absolute_penalty
    total_action_penalty = action_diff_penalty + absolute_action_penalty

    # A missed ray contributes no height reward rather than treating the miss as ground.
    # Keep height reward small and positive (<= 1.5/step) so survival beats crashing (-100)
    # without letting hovering beat forward progress and the goal bonus.
    ground_height_reward = exponential_reward_function(
        ground_height_reward_magnitude,
        ground_height_reward_exponent,
        ground_clearance - target_ground_clearance,
    )
    ground_height_reward = torch.where(
        ground_hit > 0,
        ground_height_reward,
        torch.zeros_like(ground_height_reward),
    )

    goal_reached_reward = torch.where(
        dist < 1.0,
        parameter_dict["goal_reached_bonus"] * torch.ones_like(dist),
        torch.zeros_like(dist),
    )

    # combined reward
    reward = (
        MULTIPLICATION_FACTOR_REWARD
        * (
            pos_reward
            + very_close_to_goal_reward
            + getting_closer_reward
            + distance_from_goal_reward
            + goal_reached_reward
        )
        + total_action_penalty
        + ground_height_reward
    )

    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    return reward, crashes
