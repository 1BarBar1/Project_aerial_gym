from aerial_gym.utils.logging import CustomLogger
logger = CustomLogger(__name__)

from aerial_gym.sim.sim_builder import SimBuilder
import torch
from aerial_gym.utils.helpers import get_args
from aerial_gym.config.env_config.slope_forest_env import ForestEnvSlopedCfg as EnvCfg

if __name__ == "__main__":
    args = get_args()
    logger.warning("This example demonstrates the use of geometric controllers for a quadrotor.")
    
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="slope_forest_env",
        robot_name="base_quadrotor",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        # num_envs=args.num_envs,
        num_envs= 4,
        headless=args.headless,
        use_warp=args.use_warp,
    )
    
    # Set the target waypoint for all drones: [X, Y, Z, Yaw]
    actions = torch.zeros((env_manager.num_envs, 4)).to("cuda:0")
    actions[:, 0] = 0.0   # Start X at 0
    actions[:, 1] = 0.0   # Stay centered on Y
    actions[:, 2] = 2.0   # Fly at 2 meters height
    actions[:, 3] = 0.0   # 0 Yaw (facing forward)
    env_manager.global_tensor_dict["target_position"] = actions[:, :3]
    
    env_manager.reset()
    
    for i in range(10000):
        # Stop moving the target forward once it reaches the new end of the forest
        if actions[0, 0] < 35.0:
            actions[:, 0] += 0.02  
            
        env_manager.step(actions=actions)