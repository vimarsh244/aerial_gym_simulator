import os
import time
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.rl_training.hexa_42_controller_ppo import PPOTrainer
import torch
import numpy as np
import matplotlib.pyplot as plt


logger = CustomLogger(__name__)

def create_env():
    os.makedirs("logs", exist_ok=True)
    
    num_envs = 64  # reducucing to 64 from 512 for any decent performance
    headless = True
    use_warp = False
    
    env = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="empty_env",
        robot_name="base_hexa_wing",
        # controller_name="nn_42_controller",
        controller_name="lee_attitude_control",
        args=None,
        device="cuda:0",
        num_envs=num_envs,
        headless=headless,
        use_warp=use_warp,
    )
    
    return env

def main():
    env = create_env()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    save_dir = os.path.join(os.getcwd(), "trained_models")
    trainer = PPOTrainer(env, "nn_42_controller", device, save_path=save_dir)
    
    total_timesteps = 1_000_000
    
    episode_rewards = []
    current_rewards = torch.zeros(env.num_envs, device=device)
    episode_lengths = []
    current_lengths = torch.zeros(env.num_envs, device=device)
    
    log_interval = 10
    
    logger.info("Starting neural network training for 4+2 hexarotor...")
    state = env.reset()
    
    steps = 0
    start_time = time.time()
    
    try:
        while steps < total_timesteps:
            with torch.no_grad():
                action = trainer.controller.update(state)
                # action = action[..., :4]
            
            print(action)
            # print(env.step(action))
            # next_state, reward, terminated, truncated, info = env.step(action)
            data = env.step(action)
            if data is None:
                break
            next_state, reward, terminated, truncated, info = data
            done = torch.tensor(terminated, device=device)
            # next_state, reward, done, info = env.step(action)
            
            steps += env.num_envs
            
            current_rewards += reward.flatten()
            current_lengths += 1
            
            for i, d in enumerate(done.flatten()):
                if d:
                    episode_rewards.append(current_rewards[i].item())
                    episode_lengths.append(current_lengths[i].item())
                    current_rewards[i] = 0
                    current_lengths[i] = 0
            
            if len(episode_rewards) > 0 and len(episode_rewards) % log_interval == 0:
                mean_reward = np.mean(episode_rewards[-log_interval:])
                mean_length = np.mean(episode_lengths[-log_interval:])
                elapsed = time.time() - start_time
                logger.info(f"Step: {steps}, Episodes: {len(episode_rewards)}, Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.2f}, Time: {elapsed:.2f}s")
            
            state = next_state
            
            if steps % 8192 == 0:
                logger.info(f"Training at step {steps}...")
                trainer.train(8192)  
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    trainer.save_model("final")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(2, 1, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_stats.png'))
    
    logger.info(f"Training completed. Model saved to {save_dir} and statistics plotted.")

if __name__ == "__main__":
    main()