import numpy as np
class control:
    # Controller characteristics
    num_actions = 6  # 6 motors for hexarotor
    state_dim = 24   #total of all diff  (position, orientation, velocities, etc.)
    hidden_size = 16
    
    # Neural network settings
    normalize_inputs = False  # 
    state_mean = 0.0
    state_std = 1.0
    output_scaling = 1.0
    
    # Training parameters
    learning_rate = 3e-4
    batch_size = 64
    discount_factor = 0.99
    gae_lambda = 0.95
    clip_param = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    
    # For 4+2 configuration
    vertical_motors = 4
    horizontal_motors = 2