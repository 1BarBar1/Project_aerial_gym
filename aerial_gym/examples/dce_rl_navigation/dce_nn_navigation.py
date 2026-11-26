import time
import isaacgym

# isort: on
import torch
from aerial_gym.rl_training.sample_factory.aerialgym_examples.train_aerialgym import (
    parse_aerialgym_cfg,
)
from aerial_gym.utils import get_args
from aerial_gym.registry.task_registry import task_registry


from aerial_gym.examples.dce_rl_navigation.dce_navigation_task import DCE_RL_Navigation_Task
from aerial_gym.examples.dce_rl_navigation.sf_inference_class import NN_Inference_Class

import matplotlib
import numpy as np
from PIL import Image


def sample_command(args):
    use_warp = True
    headless = False
    seg_frames = []
    depth_frames = []
    merged_image_frames = []

    rl_task = task_registry.make_task(
        "dce_navigation_task", seed=42,num_envs=4, use_warp=use_warp, headless=headless
    )
    print("Number of environments", rl_task.num_envs)
    command_actions = torch.zeros((rl_task.num_envs, rl_task.task_config.action_space_dim))
    command_actions[:, 0] = 3.5
    command_actions[:, 1] = 0.0
    command_actions[:, 2] = 0.0
    nn_model = get_network(rl_task.num_envs)
    nn_model.eval()
    nn_model.reset(torch.arange(rl_task.num_envs))
    rl_task.reset()
    success_count = 0
    crash_count = 0
    timeout_count = 0
    total_episodes = 0
    episode_clearances = []

    for i in range(0, 50000):
        start_time = time.time()
        obs, rewards, termination, truncation, infos = rl_task.step(command_actions)

        obs["obs"] = obs["observations"]
        # print(obs["observations"].shape)
        action = nn_model.get_action(obs)
        # print("Action", action, action.shape)
        action = torch.tensor(action).expand(rl_task.num_envs, -1)
        command_actions[:] = action

        if i % 25 == 0 and "ground_clearance" in infos:
            gc = infos["ground_clearance"]
            gc_vals = ", ".join(
                f"env{e}={gc[e].item():.2f}m" for e in range(min(4, rl_task.num_envs))
            )
            print(
                f"[Step {i:5d}] Live Ground Clearance | Mean: {gc.mean().item():.2f}m | {gc_vals}"
            )

        reset_tuples = (termination + truncation).nonzero(as_tuple=True)
        done_env_ids = reset_tuples[0]

        if len(done_env_ids) > 0:
            for env_id_t in done_env_ids:
                eid = env_id_t.item()
                total_episodes += 1
                dist_str = ""
                if "dist_to_goal" in infos:
                    dist_val = infos["dist_to_goal"][eid].item()
                    dist_str = f" | Dist to target: {dist_val:.2f}m"

                clearance_str = ""
                if "ground_clearance" in infos:
                    curr_gc = infos["ground_clearance"][eid].item()
                    if "ep_avg_ground_clearance" in infos:
                        ep_gc = infos["ep_avg_ground_clearance"][eid].item()
                        episode_clearances.append(ep_gc)
                        clearance_str = f" | Clearance: {curr_gc:.2f}m (Ep Avg: {ep_gc:.2f}m)"
                    else:
                        episode_clearances.append(curr_gc)
                        clearance_str = f" | Clearance: {curr_gc:.2f}m"

                # Check outcome:
                # 1. Termination = Collision / Crash
                # 2. Success = Truncation + reached within 1.0m of target without crash
                # 3. Timeout = Truncation + did not reach within 1.0m of target
                if termination[eid]:
                    crash_count += 1
                    status = "CRASH (Termination)"
                elif "successes" in infos and infos["successes"][eid]:
                    success_count += 1
                    status = "SUCCESS (Target Reached)"
                elif "timeouts" in infos and infos["timeouts"][eid]:
                    timeout_count += 1
                    status = "TIMEOUT (Out of Time)"
                elif truncation[eid]:
                    timeout_count += 1
                    status = "TIMEOUT (Truncation)"
                else:
                    status = "UNKNOWN"

                print(f"[Step {i:5d}] Env {eid}: {status}{dist_str}{clearance_str}")

            succ_rate = (success_count / total_episodes) * 100.0
            crash_rate = (crash_count / total_episodes) * 100.0
            tout_rate = (timeout_count / total_episodes) * 100.0
            mean_gc_str = (
                f" | Mean Ep Clearance: {np.mean(episode_clearances):.2f}m"
                if episode_clearances
                else ""
            )
            print(
                f"   ==> [Metrics] Total Episodes: {total_episodes} | "
                f"Success: {success_count} ({succ_rate:.1f}%) | "
                f"Crash: {crash_count} ({crash_rate:.1f}%) | "
                f"Timeout: {timeout_count} ({tout_rate:.1f}%){mean_gc_str}"
            )

            nn_model.reset(reset_tuples)

    # # Uncomment the below lines to save the frames from an episode as a GIF
    #     # save obs to file as a .gif
    #     image1 = (
    #         255.0 * rl_task.obs_dict["depth_range_pixels"][0, 0].cpu().numpy()
    #     ).astype(np.uint8)
    #     seg_image1 = rl_task.obs_dict["segmentation_pixels"][0, 0].cpu().numpy()
    #     seg_image1[seg_image1 <= 0] = seg_image1[seg_image1 > 0].min()
    #     seg_image1_normalized = (seg_image1 - seg_image1.min()) / (
    #         seg_image1.max() - seg_image1.min()
    #     )
    #
    #     # set colormap to plasma in matplotlib
    #     seg_image1_normalized_plasma = matplotlib.cm.plasma(seg_image1_normalized)
    #     seg_image1 = Image.fromarray((seg_image1_normalized_plasma * 255.0).astype(np.uint8))
    #
    #     depth_image1 = Image.fromarray(image1)
    #     image_4d = np.zeros((image1.shape[0], image1.shape[1], 4))
    #     image_4d[:, :, 0] = image1
    #     image_4d[:, :, 1] = image1
    #     image_4d[:, :, 2] = image1
    #     image_4d[:, :, 3] = 255.0
    #     merged_image = np.concatenate((image_4d, seg_image1_normalized_plasma * 255.0), axis=0)
    #     # save frames to array:
    #     seg_frames.append(seg_image1)
    #     depth_frames.append(depth_image1)
    #     merged_image_frames.append(Image.fromarray(merged_image.astype(np.uint8)))
    #     if termination[0] or truncation[0]:
    #         print("i", i)
    #         rl_task.reset()
    #         # save frames as a gif:
    #         seg_frames[0].save(
    #             f"seg_frames_{i}.gif",
    #             save_all=True,
    #             append_images=seg_frames[1:],
    #             duration=100,
    #             loop=0,
    #         )
    #         depth_frames[0].save(
    #             f"depth_frames_{i}.gif",
    #             save_all=True,
    #             append_images=depth_frames[1:],
    #             duration=100,
    #             loop=0,
    #         )
    #         merged_image_frames[0].save(
    #             f"merged_image_frames_{i}.gif",
    #             save_all=True,
    #             append_images=merged_image_frames[1:],
    #             duration=100,
    #             loop=0,
    #         )
    #         seg_frames = []
    #         depth_frames = []
    #         merged_image_frames = []


def get_network(num_envs):
    """Script entry point."""
    # register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    print("CFG is:", cfg)
    nn_model = NN_Inference_Class(num_envs, 3, 81, cfg)
    return nn_model


if __name__ == "__main__":
    task_registry.register_task(
        task_name="dce_navigation_task",
        task_class=DCE_RL_Navigation_Task,
        task_config=task_registry.get_task_config(
            "navigation_task"
        ),  # same config as navigation task
    )
    args = get_args()
    sample_command(args)
