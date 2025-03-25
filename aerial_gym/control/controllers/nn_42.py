import torch
import torch.nn as nn
import torch.nn.functional as F
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("nn_42_controller")

class HexarotorNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HexarotorNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.tanh(x) ## not sure if capping to 0-1 is necessary here cause it's already done in the controller part but still, better perf i hope
        return x

class HexarotorNNController:    
    def __init__(self, cfg, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        
        # state and action dimensions comes from the config
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.num_actions
        self.action_dim = 4
        
        #initiatling neural network
        self.network = HexarotorNeuralNetwork(
            input_size=self.state_dim,
            hidden_size=cfg.hidden_size,
            output_size=self.action_dim
        ).to(device)
        
        #while in inference mode
        self.eval_mode = False
        
        self.action_output = None
        self.noise_tensor = None
        
        logger.info(f"Initialized Hexarotor NN Controller with {self.state_dim} inputs and {self.action_dim} outputs")
    
    def __call__(self, command_actions):
        return self.update(command_actions)
    
    def init_tensors(self, global_tensor_dict=None):
        self.action_output = torch.zeros((self.num_envs, self.action_dim), 
                                        device=self.device, 
                                        requires_grad=False)
        if hasattr(self.cfg, 'exploration_noise') and self.cfg.exploration_noise > 0:
            self.noise_tensor = torch.zeros((self.num_envs, self.action_dim), 
                                           device=self.device, 
                                           requires_grad=False)
        
        if global_tensor_dict is not None:
            global_tensor_dict["nn_controller_actions"] = self.action_output
        
        logger.info("Initialized controller tensors")
        
    
    def update(self, command_actions):
        """
        Compute control actions from state using neural network
        :param command_actions: tensor containing state information or commands
        :return: Control actions for the hexarotor motors
        """
        try:
            with torch.set_grad_enabled(not self.eval_mode):
                if command_actions is None:
                    logger.warning("Received None input in update method")
                    return torch.zeros((self.num_envs, self.action_dim), device=self.device)
                
                state_tensor = command_actions
                
                input_size = state_tensor.shape[-1]
                if input_size != self.network.fc1.in_features:
                    # Option 1: redoing network with correct dimensions (first time only)
                    if self.network.fc1.in_features != input_size:
                        logger.info(f"Reinitializing network with input size {input_size} instead of {self.state_dim}")
                        self.network = HexarotorNeuralNetwork(
                            input_size=input_size,
                            hidden_size=self.cfg.hidden_size,
                            output_size=self.action_dim
                        ).to(self.device)
                
                actions = self.network(state_tensor)
                
                output_scale = getattr(self.cfg, 'output_scaling', 1.0)
                actions = torch.tanh(actions) * output_scale
                
                if self.action_output is not None:
                    self.action_output.copy_(actions)
                    actions = self.action_output
                
                if actions.shape[0] != self.num_envs or actions.shape[1] != self.action_dim:
                    logger.warning(f"Action shape mismatch: {actions.shape}, expected ({self.num_envs}, {self.action_dim})")
                    if actions.shape[0] != self.num_envs:
                        actions = actions.expand(self.num_envs, -1)
                    if actions.shape[1] != self.action_dim:
                        actions = torch.nn.functional.pad(actions, (0, self.action_dim - actions.shape[1]))
                    
                return actions
        except Exception as e:
            logger.error(f"Error in controller: {str(e)}")
            return torch.zeros((self.num_envs, self.action_dim), device=self.device)

    def randomize_params(self, env_ids=None):
        """
        Randomize controller parameters for exploration
        This is called by the robot during reset
        
        :param env_ids: Optional list of environment IDs to randomize
        """
        # Skip if no exploration noise is needed
        if not hasattr(self.cfg, 'exploration_noise') or self.cfg.exploration_noise <= 0:
            return
            
        ## did this entire thing from the ither controller class so not sure how much randomization is needed

        if self.noise_tensor is None:
            self.noise_tensor = torch.zeros((self.num_envs, self.action_dim), 
                                           device=self.device, 
                                           requires_grad=False)
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
            noise_scale = getattr(self.cfg, 'exploration_noise', 0.1)
        
        # Generate random noise
        noise = torch.randn(len(env_ids), self.action_dim, device=self.device) * noise_scale
        self.noise_tensor[env_ids] = noise
        
        logger.debug(f"Randomized exploration noise for {len(env_ids)} environments")
        
    def set_eval_mode(self):
        """Set controller to evaluation mode"""
        self.eval_mode = True
        self.network.eval()
        
    def set_train_mode(self):
        """Set controller to training mode"""
        self.eval_mode = False
        self.network.train()
        
    def save_model(self, path):
        """Save the neural network model"""
        torch.save(self.network.state_dict(), path)
        logger.info(f"Saved model to {path}")
        
    def load_model(self, path):
        """Load the neural network model"""
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Loaded model from {path}")


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from aerial_gym.control.controllers.base_controller import BaseController
# from aerial_gym.utils.logging import CustomLogger

# logger = CustomLogger("hexarotor_nn_controller")

# class HexarotorNNController(BaseController):
#     def __init__(self, config, num_envs, device):
#         super().__init__(config, num_envs, device)
#         # Define the input dimension for the neural network.
#         # Here we assume a feature vector composed of:
#         #   - robot_position: (num_envs, 3)
#         #   - robot_euler_angles: (num_envs, 3)
#         #   - command_actions: (num_envs, 4)
#         # Total input dimension = 3 + 3 + 4 = 10.
#         self.input_dim = 10
        
#         # For a hexarotor with a 4+2 configuration,
#         # we want 6 motor commands (4 for upthrust and 2 for forward propulsion).
#         self.output_dim = 6
        
#         # Define a small neural network with layers [16, 32, 16]
#         self.nn_model = nn.Sequential(
#             nn.Linear(self.input_dim, 16),
#             nn.ReLU(),
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Linear(16, self.output_dim)
#         ).to(device)

#     def init_tensors(self, global_tensor_dict=None):
#         # Initialize any tensors provided by the base controller.
#         super().init_tensors(global_tensor_dict)

#     def update(self, command_actions):
#         """
#         Neural network based controller for direct motor actuation on a hexarotor.
        
#         :param command_actions: tensor of shape (num_envs, 4) representing desired commands 
#                                 (e.g., desired velocities or other task-specific inputs).
#         :return: tensor of shape (num_envs, 6) with direct motor commands for the 4+2 configuration.
#         """
#         # Reset or initialize commands if needed (depends on your base class functionality)
#         self.reset_commands()

#         # Construct a feature vector by concatenating relevant state information with command actions.
#         # Here, we assume that self.robot_position (num_envs, 3) and self.robot_euler_angles (num_envs, 3)
#         # are maintained by the base class.
#         features = torch.cat([self.robot_position, self.robot_euler_angles, command_actions], dim=1)
        
#         # Pass the feature vector through the neural network to obtain motor commands.
#         motor_commands = self.nn_model(features)
        
#         # Optionally, apply an activation (e.g., tanh) to bound the outputs if required:
#         # motor_commands = torch.tanh(motor_commands)
        
#         return motor_commands
