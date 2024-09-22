import argparse

class posterior_args():
    batch_size = 5_000
    N_epoch = 20_000
    train_ratio = 0.8
    seed = 111
    h_dims_dynamics = [256,256]
    h_dims_posterior = [256,256]
    control_indx = [0]
    power_target_per_dim = .1
    learning_rate = 1e-5
    power_learning_rate = 1e-3
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_eps = 1e-8
    weight_decay = 10. # 0.0001
    power_weight_decay = 0.
    max_grad_norm = 1
    ema_decay = 0.

class VAE_args():
    batch_size = 5_000
    N_epoch = 20_000
    train_ratio = 0.8
    seed = 111
    h_dims_encoder = [256,256]
    h_dims_decoder = [256,256]
    control_indx = [0]
    learning_rate = 1e-5
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_eps = 1e-8
    weight_decay = 10. # 0.0001
    power_weight_decay = 0.
    max_grad_norm = 1
    ema_decay = 0.

class inverse_dynamics_args():
    N_epoch = 20_000
    batch_size = 5_000
    train_ratio = 0.8
    h_dims_inverse_dynamics = [256,256]
    control_indx = [0]
    learning_rate = 1e-5
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_eps = 1e-8
    weight_decay = 10 # 0.01 
    max_grad_norm = 1
    ema_decay = 0.