import torch
from aerial_gym import AERIAL_GYM_DIRECTORY

class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_hexa_wing"
    controller_name = "lee_velocity_control"
    args = {}
    num_envs = 16
    use_warp = True
    headless = False
    device = "cuda:0"
    observation_space_dim = 13 + 4 + 64  # root_state + action_dim _+ latent_dims
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 150  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )

    reward_parameters = {
        "pos_reward_magnitude": 5.0,
        "pos_reward_exponent": 1.0 / 3.5,
        "very_close_to_goal_reward_magnitude": 10.0,
        "very_close_to_goal_reward_exponent": 2.0,
        "getting_closer_reward_multiplier": 10.0,
        "x_action_diff_penalty_magnitude": 0.8,
        "x_action_diff_penalty_exponent": 3.333,
        "z_action_diff_penalty_magnitude": 0.8,
        "z_action_diff_penalty_exponent": 5.0,
        "yawrate_action_diff_penalty_magnitude": 0.8,
        "yawrate_action_diff_penalty_exponent": 3.33,
        "x_absolute_action_penalty_magnitude": 0.1,
        "x_absolute_action_penalty_exponent": 0.3,
        "z_absolute_action_penalty_magnitude": 1.5,
        "z_absolute_action_penalty_exponent": 1.0,
        "yawrate_absolute_action_penalty_magnitude": 1.5,
        "yawrate_absolute_action_penalty_exponent": 2.0,
        "collision_penalty": -100.0,
    }