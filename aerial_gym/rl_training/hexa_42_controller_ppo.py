import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import os
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.registry.controller_registry import controller_registry

logger = CustomLogger("hexa_42_controller_ppo")

class PPOTrainer:
    def __init__(self, env, controller_name, device, save_path="./models"):
        self.env = env
        self.device = device
        self.save_path = save_path
        
        self.controller, self.controller_config = controller_registry.make_controller(
            controller_name, env.num_envs, device
        )
        
        self.optimizer = optim.Adam(
            self.controller.network.parameters(),
            lr=self.controller_config.learning_rate
        )
        
        self.batch_size = self.controller_config.batch_size
        self.discount = self.controller_config.discount_factor
        self.gae_lambda = self.controller_config.gae_lambda
        self.clip_param = self.controller_config.clip_param
        self.value_loss_coef = self.controller_config.value_loss_coef
        self.entropy_coef = self.controller_config.entropy_coef
        self.max_grad_norm = self.controller_config.max_grad_norm
        
        self.value_network = nn.Sequential(
            nn.Linear(self.controller_config.state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        ).to(device)
        
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(), 
            lr=self.controller_config.learning_rate
        )
        
        os.makedirs(self.save_path, exist_ok=True)
        
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(rewards.size(0))):
            if t == rewards.size(0) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
                
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.discount * next_val * mask - values[t]
            last_gae = delta + self.discount * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae
            
        returns = advantages + values
        return returns, advantages
    
    def collect_rollouts(self, num_steps):
        states = []
        actions = []
        # log_probs = []
        rewards = []
        dones = []
        values = []
        
        state = self.env.reset()
        
        for _ in range(num_steps):
            with torch.no_grad():
                value = self.value_network(state).squeeze(-1)
                action = self.controller.update(state)
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            
            state = next_state
            
        with torch.no_grad():
            next_value = self.value_network(state).squeeze(-1)
            
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        values = torch.stack(values)
        
        returns, advantages = self.compute_gae(rewards, values, dones, next_value)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'values': values,
            'returns': returns,
            'advantages': advantages,
            'next_state': state
        }
    
    def train_epoch(self, rollouts, epochs=10):
        states = rollouts['states']
        actions = rollouts['actions']
        returns = rollouts['returns']
        advantages = rollouts['advantages']
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = states.size(0)
        minibatch_size = dataset_size // self.batch_size
        
        for _ in range(epochs):
            indices = torch.randperm(dataset_size)
            
            for start in range(0, dataset_size, minibatch_size):
                end = min(start + minibatch_size, dataset_size)
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                # mb_advantages = advantages[mb_indices]
                
                mb_values = self.value_network(mb_states).squeeze(-1)
                
                new_actions = self.controller.update(mb_states)
                action_loss = ((new_actions - mb_actions) ** 2).mean()
                
                #value loss
                value_loss = F.mse_loss(mb_values, mb_returns)
                
                # net loss
                loss = action_loss - self.entropy_coef * 0.01 + self.value_loss_coef * value_loss
                
                # updating policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controller.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                #updatimng value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
    
    def train(self, total_timesteps, save_interval=10000):
        timesteps_per_rollout = self.env.num_envs * 2048
        num_updates = total_timesteps // timesteps_per_rollout
        
        for update in range(num_updates):
            rollouts = self.collect_rollouts(timesteps_per_rollout)
            self.train_epoch(rollouts)
            
            if (update + 1) % save_interval == 0:
                self.save_model(update + 1)
                
            logger.info(f"Update {update+1}/{num_updates} completed")
        
        self.save_model("final")
        logger.info("Training completed!")
        
    def save_model(self, suffix):
        model_path = f"{self.save_path}/hexa_nn_{suffix}.pt"
        value_path = f"{self.save_path}/hexa_value_{suffix}.pt"
        
        torch.save(self.controller.network.state_dict(), model_path)
        torch.save(self.value_network.state_dict(), value_path)
        logger.info(f"Saved models to {model_path} and {value_path}")
        
    def load_model(self, suffix):
        model_path = f"{self.save_path}/hexa_nn_{suffix}.pt"
        value_path = f"{self.save_path}/hexa_value_{suffix}.pt"
        
        self.controller.network.load_state_dict(torch.load(model_path, map_location=self.device))
        self.value_network.load_state_dict(torch.load(value_path, map_location=self.device))
        logger.info(f"Loaded models from {model_path} and {value_path}")

# plot horizontal and vertical motor outputs
# use unka ppo implementation dont bother with mine


    