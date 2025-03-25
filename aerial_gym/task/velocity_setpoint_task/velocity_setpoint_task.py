from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("velocity_setpoint_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class VelocitySetpointTask(BaseTask):
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
        logger.info("Building environment for velocity setpoint task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
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

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.counter = 0

        # Target velocities instead of positions
        self.target_velocity = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Get the dictionary once from the environment and use it to get the observations later.
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.observation_space = Dict(
            {"observations": Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)}
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )

        self.num_envs = self.sim_env.num_envs
        self.counter = 0

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        # Sample random velocity targets
        # self.target_velocity[:] = 20.0 * torch.rand_like(self.target_velocity) - 10.0  # -10 to 10 m/s
        self.target_velocity[:] = 10.0 * torch.ones_like(self.target_velocity) # 0 to 25 m/s
        self.infos = {}
        self.sim_env.reset()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        # Sample random velocity targets for reset environments
        # self.target_velocity[env_ids] = 20.0 * torch.rand_like(self.target_velocity[env_ids]) - 10.0
        self.target_velocity[env_ids] = 10.0 * torch.ones_like(self.target_velocity[env_ids])
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        return

    def render(self):
        return None

    def step(self, actions):
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        self.sim_env.step(actions=self.actions)

        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self.sim_env.post_reward_calculation_step()

        self.infos = {}

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        return return_tuple

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
        self.task_obs["observations"][:, 0:3] = self.target_velocity - self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        current_velocity = obs_dict["robot_body_linvel"]
        target_velocity = self.target_velocity
        robot_orientation = obs_dict["robot_orientation"]
        angular_velocity = obs_dict["robot_body_angvel"]
        
        # Calculate velocity error
        vel_error = target_velocity - current_velocity
        vel_error_magnitude = torch.norm(vel_error, dim=1)
        
        # Calculate velocity reward (decreasing with error)
        vel_reward = exp_func(vel_error_magnitude, 3.0, 8.0) + exp_func(vel_error_magnitude, 0.5, 1.0)
        
        # Calculate altitude penalty (if needed)
        robot_position = obs_dict["robot_position"]
        altitude_penalty = torch.where(robot_position[:, 2] < 0.5, 
                                      -10.0 * torch.ones_like(vel_reward), 
                                      torch.zeros_like(vel_reward))
        
        # Calculate stability rewards similar to position task
        ups = quat_axis(robot_orientation, 2)
        tiltage = torch.abs(1 - ups[..., 2])
        up_reward = 0.2 / (0.1 + tiltage * tiltage)
        
        spinnage = torch.norm(angular_velocity, dim=1)
        ang_vel_reward = (1.0 / (1.0 + spinnage * spinnage)) * 10
        
        # Calculate action penalties
        previous_action_penalty = torch.sum(
            exp_penalty_func(self.actions - self.prev_actions, 0.02, 10.0), dim=1
        )
        
        absolute_action_penalty = torch.sum(
            exp_penalty_func(self.actions, 0.01, 5.0), dim=1
        )
        
        # Combine rewards
        total_reward = (
            vel_reward + vel_reward * (up_reward + ang_vel_reward) + altitude_penalty + previous_action_penalty + absolute_action_penalty
        )
        
        # Define crashes based on altitude or excessive velocity error
        crashes = torch.zeros_like(vel_reward)
        crashes[:] = torch.where(robot_position[:, 2] < 0.2, 
                               torch.ones_like(crashes), 
                               crashes)
        crashes[:] = torch.where(vel_error_magnitude > 20.0, 
                               torch.ones_like(crashes), 
                               crashes)
        
        # Apply crash penalty
        total_reward[:] = torch.where(
            crashes > 0.0, 
            -20 * torch.ones_like(total_reward), 
            total_reward
        )
        
        return total_reward, crashes


@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * x * x) - 1)
