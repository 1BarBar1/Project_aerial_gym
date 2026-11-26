# this is here just to guarantee that isaacgym is imported before PyTorch
# isort: off
# noinspection PyUnresolvedReferences

# isort: on

from aerial_gym.registry.task_registry import task_registry
import sys
from typing import Dict, Optional, Tuple


import gymnasium as gym
import torch


from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.model.core import ModelCore, default_make_core_func
from sample_factory.model.encoder import *
from sample_factory.model.encoder import Encoder, default_make_encoder_func


from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import log, str2bool


from sample_factory.enjoy import enjoy

class AerialGymVecEnv(gym.Env):
    """
    Wrapper for isaacgym environments to make them compatible with the sample factory.
    """

    def __init__(self, aerialgym_env, obs_key):
        self.env = aerialgym_env
        self.num_agents = self.env.num_envs
        self.action_space = convert_space(self.env.action_space)
        self.obs_key = obs_key

        # isaacgym_examples environments actually return dicts
        if obs_key == "obs" or obs_key == "observations":
            raw_spaces = convert_space(self.env.observation_space)
            if obs_key == "obs" and "observations" in raw_spaces.spaces:
                self.observation_space = gym.spaces.Dict({"obs": raw_spaces.spaces["observations"]})
            else:
                self.observation_space = gym.spaces.Dict(raw_spaces)
        else:
            raise ValueError(f"Unknown observation key: {obs_key}")

        self._truncated: Tensor = torch.zeros(self.num_agents, dtype=torch.bool)

    def _remap_obs(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.obs_key == "obs" and "observations" in obs and "obs" not in obs:
            return {"obs": obs["observations"]}
        return obs

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Tensor], Dict]:
        # some IGE envs return all zeros on the first timestep, but this is probably okay
        obs, rew, terminated, truncated, infos = self.env.reset()
        return self._remap_obs(obs), infos

    def step(self, action) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        obs, rew, terminated, truncated, infos = self.env.step(action)
        return self._remap_obs(obs), rew, terminated, truncated, infos

    def render(self):
        pass


def make_aerialgym_env(
    full_task_name: str,
    cfg: Config,
    _env_config=None,
    render_mode: Optional[str] = None,
) -> Env:
    num_envs = cfg.env_agents if getattr(cfg, "env_agents", -1) > 0 else None
    headless = False if render_mode == "human" else None
    return AerialGymVecEnv(
        task_registry.make_task(task_name=full_task_name, num_envs=num_envs, headless=headless),
        "obs",
    )


def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument(
        "--env_agents",
        default=-1,
        type=int,
        help="Num agents in each env (default: -1, means use default value from isaacgymenvs env yaml config file)",
    )
    p.add_argument(
        "--obs_key",
        default="obs",
        type=str,
        help='IsaacGym envs return dicts, some envs return just "obs", and some return "obs" and "states".'
        "States key denotes the full state of the environment, and obs key corresponds to limited observations "
        'available in real world deployment. If we use "states" here we can train will full information '
        "(although the original idea was to use asymmetric training - critic sees full state and policy only sees obs).",
    )
    p.add_argument(
        "--subtask",
        default=None,
        type=str,
        help="Subtask for envs that support it (i.e. AllegroKuka regrasping or manipulation or throw).",
    )
    p.add_argument(
        "--ige_api_version",
        default="preview4",
        type=str,
        choices=["preview3", "preview4"],
        help="We can switch between different versions of IsaacGymEnvs API using this parameter.",
    )
    p.add_argument(
        "--eval_stats",
        default=False,
        type=str2bool,
        help="Whether to collect env stats during evaluation.",
    )
    p.add_argument(
        "--use_visual_lstm",
        default=False,
        type=str2bool,
        help="Pass visual latent features (obs[:, 17:]) through an nn.LSTM before concatenating with the 17-dim drone state vector.",
    )
    p.add_argument(
        "--lstm_hidden_size",
        default=64,
        type=int,
        help="Hidden size of the visual latent nn.LSTM.",
    )
    p.add_argument(
        "--lstm_num_layers",
        default=1,
        type=int,
        help="Number of layers in the visual latent nn.LSTM.",
    )
    p.add_argument(
        "--freeze_lstm",
        default=False,
        type=str2bool,
        help="If True, freeze visual LSTM weights (requires_grad=False, eval mode, no_grad forward, excluded from optimizer).",
    )
    p.add_argument(
        "--lstm_checkpoint_path",
        default=None,
        type=str,
        help="Optional path to a standalone LSTM checkpoint (.pth) to load at startup or evaluation.",
    )


def override_default_params_func(env, parser):
    """Most of these parameters are taken from IsaacGymEnvs default config files."""

    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU
        train_for_env_steps=10000000,
        use_rnn=False,
        adaptive_stddev=True,
        policy_initialization="torch_default",
        env_gpu_actions=True,
        reward_scale=0.1,
        rollout=24,
        max_grad_norm=0.0,
        batch_size=2048,
        num_batches_per_epoch=2,
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        nonlinearity="elu",
        learning_rate=3e-4,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        shuffle_minibatches=True,
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,  # assuming reward from the last step in the episode can generally be ignored
        normalize_input=True,
        normalize_returns=True,  # does not improve results on all envs, but with return normalization we don't need to tune reward scale
        save_best_after=0,
        save_best_every_sec=5,
        save_best_metric="reward",
        serial_mode=True,  # it makes sense to run isaacgym envs in serial mode since most of the parallelism comes from the env itself (although async mode works!)
        async_rl=True,
        use_env_info_cache=False,  # speeds up startup
        kl_loss_coeff=0.1,
        restart_behavior="overwrite",
    )

    # override default config parameters for specific envs
    if env in env_configs:
        parser.set_defaults(**env_configs[env])


# custom default configuration parameters for specific envs
# add more envs here analogously (env names should match config file names in IGE)
env_configs = dict(
    position_setpoint_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        gamma=0.99,
        rollout=16,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=16384,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
    ),
    navigation_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[],
        decoder_mlp_layers=[256, 128, 64],
        use_visual_lstm=True,
        lstm_hidden_size=64,
        lstm_num_layers=1,
        freeze_lstm=False,
        lstm_checkpoint_path=None,
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=64,
        rnn_type="lstm",
        gamma=0.98,
        rollout=32,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=2048,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        wandb_project="quad",
        wandb_user="mihirkulkarni",
    ),
    lidar_navigation_task=dict(
        train_for_env_steps=131000000000,
        # encoder_conv_architecture="resnet_impala",
        # encoder_conv_mlp_layers=[],
        encoder_mlp_layers=[256, 128, 64],
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=128,
        rnn_type="gru",
        gamma=0.98,
        rollout=32,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=1024,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.001,
        with_wandb=True,
        wandb_project="lidar_nav_task",
        wandb_user="mihirkulkarni",
    ),
    radar_navigation_task=dict(
        train_for_env_steps=131000000000,
        # encoder_conv_architecture="resnet_impala",
        # encoder_conv_mlp_layers=[],
        encoder_mlp_layers=[256, 128, 64],
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=128,
        rnn_type="gru",
        gamma=0.98,
        rollout=32,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=1024,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.001,
        with_wandb=True,
        wandb_project="radar_nav_task",
        wandb_user="mihirkulkarni",
    ),

)

class CustomEncoder(Encoder):
    """Just an example of how to use a custom model component."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        # the observatiosn are in the following format:
        # 17 dim state + actions
        # 80 dim lidar readings
        
        # encode lidar readings using conv network and then combine with state vector and pass through MLP
        # all are flattened into "observations" key
        self.encoders = nn.ModuleDict()

        state_action_input_size = 17
        lidar_input_size = 16*20
        out_size = 0
        out_size_cnn = 0
        out_size += obs_space["observations"].shape[0] - lidar_input_size

        # self.encoders["obs_image"] = make_img_encoder(cfg, spaces.Box(low=-1, high=1.5, shape=(1, 16, 20)))
        # out_size += self.encoders["obs_image"].get_out_size()
        ###
        # input is 16 x 20 dims lidar readings
        self.encoders["obs_image"] = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (B, 16, 16, 20)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (B, 16, 8, 10)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (B, 32, 8, 10)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (B, 32, 4, 5)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 4, 5)
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 2, 2)
            nn.Flatten(),  # (B, 64*2*2)
        )
        out_size_cnn += 128
        out_size += out_size_cnn
        ###

        self.encoder_out_size = out_size
        mlp_layers = [256, 128, 64]
        mlp_input_size = out_size
        mlp = []
        for layer_size in mlp_layers:
            mlp.append(nn.Linear(mlp_input_size, layer_size))
            mlp.append(nn.ELU())
            mlp_input_size = layer_size
        self.mlp_head_custom = nn.Sequential(*mlp)
        if len(mlp_layers) > 0:
            self.encoder_out_size = mlp_layers[-1]
        else:
            self.encoder_out_size = out_size
            self.mlp_head_custom = nn.Identity()


    def forward(self, obs_dict):
        x_state_action = obs_dict["observations"][:, :17]
        x_lidar = obs_dict["observations"][:, 17:].unsqueeze(1).view(-1, 1, 16, 20)  # (B, 1, 8, 10)
        x_lidar_encoding = self.encoders["obs_image"](x_lidar)
        x = torch.cat([x_state_action, x_lidar_encoding], dim=-1)
        x = self.mlp_head_custom(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size

def make_custom_encoder(cfg, obs_space):
    return CustomEncoder(cfg, obs_space)


STATE_VEC_DIM = 17


class VisualStateSplitEncoder(Encoder):
    """
    Passes through the 17-dim proprioceptive state and detached visual latent vector
    so the recurrent visual LSTM inside VisualLSTMCore processes only the visual latents.
    """

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.obs_key = "obs" if "obs" in obs_space.spaces else "observations"
        self.obs_dim = obs_space[self.obs_key].shape[0]
        self.state_dim = STATE_VEC_DIM
        assert self.obs_dim > self.state_dim, f"Expected obs_dim > {self.state_dim}, got {self.obs_dim}"
        self.visual_dim = self.obs_dim - self.state_dim

    def forward(self, obs_dict):
        x = obs_dict[self.obs_key]
        x_state = x[:, : self.state_dim]
        # Defense-in-depth: keep upstream frozen VAE features detached from policy gradients.
        x_visual = x[:, self.state_dim :].detach()
        return torch.cat([x_state, x_visual], dim=-1)

    def get_out_size(self) -> int:
        return self.obs_dim


class VisualLSTMCore(ModelCore):
    """
    Sample Factory ModelCore that applies an nn.LSTM strictly to the visual latent slice
    and concatenates the LSTM output with the untouched 17-dim drone state vector.
    """

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.state_dim = STATE_VEC_DIM
        assert input_size > self.state_dim, f"Expected input_size > {self.state_dim}, got {input_size}"
        self.visual_dim = input_size - self.state_dim
        self.hidden_size = int(getattr(cfg, "lstm_hidden_size", cfg.rnn_size))
        self.num_layers = int(getattr(cfg, "lstm_num_layers", cfg.rnn_num_layers))
        self.freeze_lstm = bool(getattr(cfg, "freeze_lstm", False))

        self.visual_lstm = nn.LSTM(self.visual_dim, self.hidden_size, self.num_layers)
        self.core_output_size = self.state_dim + self.hidden_size
        self._apply_freeze_state()

    def _apply_freeze_state(self):
        if self.freeze_lstm:
            for param in self.visual_lstm.parameters():
                param.requires_grad = False
            self.visual_lstm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_lstm:
            self.visual_lstm.eval()
        return self

    def load_lstm_checkpoint(self, path: str, device=None):
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.visual_lstm.load_state_dict(state_dict)
        self._apply_freeze_state()
        log.info("Loaded standalone visual LSTM checkpoint from %s (freeze_lstm=%s)", path, self.freeze_lstm)

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)

        if self.num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        h, c = torch.split(rnn_states, self.hidden_size, dim=2)
        h_in, c_in = h.contiguous(), c.contiguous()

        if is_seq:
            state_data = head_output.data[:, : self.state_dim]
            visual_data = head_output.data[:, self.state_dim :].detach()
            visual_in = PackedSequence(
                visual_data,
                head_output.batch_sizes,
                head_output.sorted_indices,
                head_output.unsorted_indices,
            )
        else:
            state_data = head_output[:, : self.state_dim]
            visual_in = head_output[:, self.state_dim :].detach().unsqueeze(0)

        if self.freeze_lstm:
            with torch.no_grad():
                lstm_out, (h_out, c_out) = self.visual_lstm(visual_in, (h_in, c_in))
        else:
            lstm_out, (h_out, c_out) = self.visual_lstm(visual_in, (h_in, c_in))

        if is_seq:
            concat_data = torch.cat([state_data, lstm_out.data], dim=-1)
            out = PackedSequence(
                concat_data,
                head_output.batch_sizes,
                head_output.sorted_indices,
                head_output.unsorted_indices,
            )
        else:
            out = torch.cat([state_data, lstm_out.squeeze(0)], dim=-1)

        new_rnn_states = torch.cat((h_out, c_out), dim=2)
        if self.num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        if self.freeze_lstm:
            new_rnn_states = new_rnn_states.detach()

        return out, new_rnn_states


def _use_visual_lstm_architecture(cfg) -> bool:
    return bool(getattr(cfg, "use_visual_lstm", False)) and getattr(cfg, "rnn_type", None) == "lstm"


def make_visual_state_split_encoder(cfg, obs_space):
    if _use_visual_lstm_architecture(cfg):
        return VisualStateSplitEncoder(cfg, obs_space)
    return default_make_encoder_func(cfg, obs_space)


def make_visual_lstm_core(cfg, core_input_size):
    if _use_visual_lstm_architecture(cfg):
        return VisualLSTMCore(cfg, core_input_size)
    return default_make_core_func(cfg, core_input_size)


CUSTOM_ENCODER_ENVS = {"lidar_navigation_task"}

def register_aerialgym_custom_components(env=None, cfg=None):
    for env_name in env_configs:
        register_env(env_name, make_aerialgym_env)

    if env in CUSTOM_ENCODER_ENVS:
        global_model_factory().register_encoder_factory(make_custom_encoder)
    else:
        global_model_factory().register_encoder_factory(make_visual_state_split_encoder)
        global_model_factory().register_model_core_factory(make_visual_lstm_core)


def _sync_visual_lstm_cfg(cfg):
    if getattr(cfg, "use_visual_lstm", False):
        cfg.use_rnn = True
        cfg.rnn_type = "lstm"
        cfg.rnn_size = cfg.lstm_hidden_size
        cfg.rnn_num_layers = cfg.lstm_num_layers
        if not getattr(cfg, "decoder_mlp_layers", None):
            cfg.decoder_mlp_layers = [256, 128, 64]
        if hasattr(cfg, "cli_args") and isinstance(cfg.cli_args, dict):
            if any(k in cfg.cli_args for k in ("use_visual_lstm", "lstm_hidden_size", "lstm_num_layers")):
                cfg.cli_args["use_rnn"] = cfg.use_rnn
                cfg.cli_args["rnn_type"] = cfg.rnn_type
                cfg.cli_args["rnn_size"] = cfg.rnn_size
                cfg.cli_args["rnn_num_layers"] = cfg.rnn_num_layers
    return cfg


def parse_aerialgym_cfg(evaluation=False):
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)
    return _sync_visual_lstm_cfg(final_cfg)


from sample_factory.model.actor_critic import create_actor_critic


def main():
    """Script entry point."""
    # Parse cfg first so we know the env name, then conditionally register
    # the custom encoder only for envs that need it.  run_rl() only requires
    # envs/encoders to be registered, not the cfg parse, so this order is safe.
    cfg = parse_aerialgym_cfg()
    register_aerialgym_custom_components(env=cfg.env, cfg=cfg)
    status = run_rl(cfg)
    return status
    

if __name__ == "__main__":
    sys.exit(main())

