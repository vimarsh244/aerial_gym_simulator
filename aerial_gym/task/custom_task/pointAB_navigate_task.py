from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger
import torch
import numpy as np
from gymnasium.spaces import Box

logger = CustomLogger("pointAB_navigate_task")


class PointABNavigateTask(BaseTask):
    def __init__(self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None):
        super().__init__(task_config)
        self.device = self.task_config.device
        # Override params if provided

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
            
        
        # Convert reward parameters to tensors for faster computation
        for key in self.task_config.reward_parameters.keys():
            if isinstance(self.task_config.reward_parameters[key], list):
                self.task_config.reward_parameters[key] = torch.tensor(
                    self.task_config.reward_parameters[key], device=self.device
                )
        
        logger.info("Building environment for Point A to B navigation task.")
        logger.info(
            f"Sim Name: {self.task_config.sim_name}, Env Name: {self.task_config.env_name}, "
            f"Robot Name: {self.task_config.robot_name}, Controller Name: {self.task_config.controller_name}"
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
        
        # Define starting position (A) and target position (B)
        self.start_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        
        # Fixed distance between A and B (50 meters in x-direction)
        self.target_distance = 50.0
        
        # Tracking variables for rewards and episode stats
        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0
        
        self.prev_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.prev_velocity = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.prev_distance_to_target = torch.zeros(
            (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
        )
        self.prev_action = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim), device=self.device, requires_grad=False
        )
        
        # Get observation dictionary from the environment
        self.obs_dict = self.sim_env.get_obs()
        
        # Set up termination and reward tensors
        self.terminations = torch.zeros(
            (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
        )
        self.truncations = torch.zeros(
            (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
        )
        self.rewards = torch.zeros(
            (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
        )
        
        # Define the observation and action spaces
        self.observation_space = Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(self.task_config.observation_space_dim,),
            dtype=np.float32
        )
        
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(self.task_config.action_space_dim,), dtype=np.float32
        )
        
        # Define the action transformation function
        self.action_transformation_function = self.task_config.action_transformation_function \
            if hasattr(self.task_config, 'action_transformation_function') else lambda x: x
        
        self.num_envs = self.sim_env.num_envs
        
        # Task observations
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (self.sim_env.num_envs, self.task_config.privileged_observation_space_dim),
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
        
        self.num_task_steps = 0
        self.infos = {}

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # Initialize start position and target position
        # Start position is random within the environment bounds
        random_start_pos = torch_rand_float_tensor(
            lower=self.obs_dict["env_bounds_min"][env_ids],
            upper=self.obs_dict["env_bounds_max"][env_ids]
        )
        
        # For simplicity, set target 50 meters in the positive x direction
        random_direction = torch.zeros_like(random_start_pos)
        random_direction[:, 0] = 1.0  # X direction
        
        # Normalize the direction
        random_direction = random_direction / torch.norm(random_direction, dim=1, keepdim=True)
        
        # Calculate target as start + direction * distance
        self.start_position[env_ids] = random_start_pos
        self.target_position[env_ids] = random_start_pos + random_direction * self.target_distance
        
        # Reset states to start position
        # Reset states to start position
        if "robot_position" in self.obs_dict:
            self.obs_dict["robot_position"][env_ids] = self.start_position[env_ids]
        
        # Handle velocity resets - make sure keys exist first
        # Reset tracking variables
        self.prev_position[env_ids] = self.start_position[env_ids]
        self.prev_velocity[env_ids] = torch.zeros_like(self.prev_velocity[env_ids])
        self.prev_distance_to_target[env_ids] = torch.norm(
            self.target_position[env_ids] - self.start_position[env_ids], dim=1, keepdim=True
        )
        self.prev_action[env_ids] = torch.zeros_like(self.prev_action[env_ids])
        
        self.infos = {}
        # Reset simulation for the specified environments
        self.sim_env.reset_idx(env_ids)

    def render(self):
        return self.sim_env.render()

    def process_obs_for_task(self):
        # Calculate vector to target in robot's frame
        # Safely get observation keys with fallbacks
        robot_position = self.obs_dict.get("robot_position", torch.zeros((self.sim_env.num_envs, 3), device=self.device))
        robot_orientation = self.obs_dict.get("robot_orientation", torch.tensor([[0, 0, 0, 1]], device=self.device).repeat(self.sim_env.num_envs, 1))
        robot_linear_vel = self.obs_dict.get("robot_linear_vel", torch.zeros((self.sim_env.num_envs, 3), device=self.device))
        robot_angular_vel = self.obs_dict.get("robot_angular_vel", torch.zeros((self.sim_env.num_envs, 3), device=self.device))
        # Vector from current position to target in world frame
        vector_to_target = self.target_position - robot_position
        
        # Calculate distance to target
        distance_to_target = torch.norm(vector_to_target, dim=1, keepdim=True)
        
        # Normalize direction vector
        direction_to_target = vector_to_target / torch.clamp(distance_to_target, min=0.001)
        
        # Transform direction vector to robot frame
        q_inv = quat_conjugate(robot_orientation)
        direction_to_target_robot_frame = quat_rotate(q_inv, direction_to_target)
        
        # Normalized distance to target (1.0 at start, 0.0 at target)
        normalized_distance = torch.clamp(distance_to_target / self.target_distance, 0.0, 1.0)
        
        # Combine observations
        self.task_obs["observations"][:, 0:3] = direction_to_target_robot_frame  # Direction to target in robot frame
        self.task_obs["observations"][:, 3:6] = robot_linear_vel                 # Linear velocity
        self.task_obs["observations"][:, 6:9] = robot_angular_vel                # Angular velocity
        self.task_obs["observations"][:, 9:13] = robot_orientation               # Quaternion orientation
        self.task_obs["observations"][:, 13] = normalized_distance.squeeze(-1)  # Normalized distance to target

    def compute_rewards_and_crashes(self):
        # Get current position and velocity
        position = self.obs_dict["robot_position"]
        velocity = self.obs_dict["robot_linear_vel"]
        
        # Calculate distance to target
        vector_to_target = self.target_position - position
        distance_to_target = torch.norm(vector_to_target, dim=1, keepdim=True)
        
        # Check for success (reached target)
        success_threshold = 2.0  # meters
        successes = distance_to_target < success_threshold
        
        # Check for crashes
        crashes = self.obs_dict["crashes"].clone()
        
        # Calculate reward components
        
        # 1. Progress reward (distance decreased toward target)
        progress_reward = (self.prev_distance_to_target - distance_to_target) * 2.0
        
        # 2. Velocity reward (encourage moving fast toward the target)
        # Project velocity onto the direction to target
        direction_to_target = vector_to_target / torch.clamp(distance_to_target, min=0.001)
        vel_projection = torch.sum(velocity * direction_to_target, dim=1, keepdim=True)
        
        # Reward high velocity in the target direction, penalize moving away from target
        velocity_reward = vel_projection * 0.5
        
        # 3. Smoothness reward (penalize large changes in action)
        action_diff = torch.norm(self.prev_action - self.action_tensor, dim=1, keepdim=True)
        smoothness_reward = -action_diff * 0.1
        
        # 4. Success reward (big bonus for reaching target)
        success_reward = successes.float() * 100.0
        
        # 5. Crash penalty
        crash_penalty = crashes.float() * self.task_config.reward_parameters["crash_penalty"]
        
        # Combine rewards
        rewards = progress_reward + velocity_reward + smoothness_reward + success_reward + crash_penalty
        
        # Update previous values
        self.prev_distance_to_target = distance_to_target
        self.prev_position = position.clone()
        self.prev_velocity = velocity.clone()
        
        return rewards, crashes, successes

    def get_return_tuple(self):
        # Update observations before returning
        self.process_obs_for_task()
        
        observations = {
            "observations": self.task_obs["observations"].clone().detach()
        }
        
        return observations, self.rewards.squeeze(-1), self.terminations.squeeze(-1), self.truncations.squeeze(-1), self.infos

    def step(self, actions):
        # Store actions for reward calculation
        self.action_tensor = actions.clone()
        
        # Transform actions if needed (e.g., scale or convert to controller inputs)
        transformed_actions = self.action_transformation_function(actions)
        
        # Execute step in simulation
        self.sim_env.step(actions=transformed_actions)
        
        # Calculate rewards and check for termination conditions
        self.rewards, self.terminations, successes = self.compute_rewards_and_crashes()
        
        # Check for timeouts (episode length exceeded)
        self.truncations = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations)
        )
        
        # Don't count as timeout if already terminated or successful
        self.truncations = torch.where(
            torch.logical_or(self.terminations > 0, successes > 0),
            torch.zeros_like(self.truncations),
            self.truncations
        )
        
        # Set info dict for monitoring
        self.infos["successes"] = successes
        self.infos["crashes"] = self.terminations
        self.infos["timeouts"] = self.truncations
        
        # Track statistics
        self.success_aggregate += torch.sum(successes).item()
        self.crashes_aggregate += torch.sum(self.terminations).item()
        self.timeouts_aggregate += torch.sum(self.truncations).item()
        
        # Reset environments that are done (either terminated or truncated)
        done_envs = torch.logical_or(self.terminations, self.truncations).squeeze(-1).nonzero().flatten()
        if len(done_envs) > 0:
            self.reset_idx(done_envs)
        
        # Update previous action for next step
        self.prev_action = self.action_tensor.clone()
        
        self.num_task_steps += 1
        
        # Return step information
        return self.get_return_tuple()

    def set_task_params(self, target_distance):
        """Allows changing the task parameters dynamically"""
        self.target_distance = target_distance
        # Reset all environments to apply the new distance
        self.reset()