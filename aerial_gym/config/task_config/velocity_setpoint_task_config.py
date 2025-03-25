import torch
from aerial_gym import AERIAL_GYM_DIRECTORY

class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_hexa_wing"
    # controller_name = "lee_velocity_control"
    controller_name = "no_control"
    args = {}
    num_envs = 128
    use_warp = True
    headless = False
    device = "cuda:0"
    observation_space_dim = 13
    privileged_observation_space_dim = 0
    action_space_dim = 6
    episode_len_steps = 200
    return_state_before_reset = False

    reward_parameters = {
        "exp_func_gain1": 3.5,           # used for main velocity error term
        "exp_func_exp1": 8.0,
        "exp_func_gain2": 0.5,           # secondary velocity error term
        "exp_func_exp2": 1.0,
        # "altitude_penalty": -10.0,       # penalty when altitude < 0.5
        # "crash_penalty": -100.0,          # penalty if altitude < 0.2 or vel error > 20
        "action_diff_penalty_gain": [0.02, 10.0],  # for previous_action_penalty
        "absolute_action_penalty_gain": [0.01, 5.0],  # for absolute_action_penalty
    }

