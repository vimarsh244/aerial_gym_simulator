from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.helpers import get_args
from aerial_gym.task.custom_task.pointAB_navigate_task import PointABNavigateTask
from aerial_gym.registry.task_registry import task_registry
import torch
import time

logger = CustomLogger(__name__)

if __name__ == "__main__":
    logger.print_example_message()
    logger.warning("\n\n\nJust an example task interface.\n\n\n")
    start = time.time()
    args = get_args()

    rl_task_env = task_registry.make_task(
        "point_nav_task",
        # headless=args.headless,
        # num_envs=args.num_envs
    )
    rl_task_env.reset()
    actions = torch.zeros(
        (
            rl_task_env.sim_env.num_envs,
            rl_task_env.sim_env.robot_manager.robot.controller_config.num_actions,
        )
    ).to("cuda:0")
    actions[:] = 0.0
    with torch.no_grad():
        for i in range(30000):
            if i == 100:
                start = time.time()
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
    end = time.time()
    
    # Initialize the task
    # task = PointABNavigateTask(
    #     task_config=task_config,
    #     seed=args.seed,
    #     num_envs=args.num_envs,
    #     headless=args.headless,
    #     device=args.device,
    #     use_warp=args.use_warp
    # )
    
    # # Initialize RL agent
    # agent = PPOAgent(
    #     observation_space=task.observation_space,
    #     action_space=task.action_space,
    #     device=args.device
    # )
    
    # Training loop
    # for episode in range(args.num_episodes):
    #     obs = task.reset()
    #     done = False
    #     episode_reward = 0
        
    #     while not done:
    #         # Select action from policy
    #         action = agent.select_action(obs)
            
    #         # Take step in environment
    #         next_obs, reward, done, info = task.step(action)
            
    #         # Store experience in agent's memory
    #         agent.store_transition(obs, action, next_obs, reward, done)
            
    #         # Update agent
    #         if done:
    #             agent.update()
            
    #         obs = next_obs
    #         episode_reward += reward
            
    #     logger.info(f"Episode {episode}, Reward: {episode_reward}")
    
    # Save trained model
    