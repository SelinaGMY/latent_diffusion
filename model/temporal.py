from flax import linen as nn
import jax.numpy as np
from jax import random, vmap, jit, grad, value_and_grad, lax, scipy
from einops.layers.flax import Rearrange
import einops
from functools import partial
import gym
import copy

from model.helpers import (
    residual_temporal_block,
    sinusoidal_pos_emb,
    downsample1d,
    upsample1d,
    conv1d_block,
    mish,
    residual
)

class temporal_unet(nn.Module):
    """
    based on https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
    """
    args: dict

    def setup(self):

        self.transition_dim = self.args.transition_dim

        dims = [self.transition_dim, *map(lambda m: self.args.u_net_dim * m, self.args.u_net_dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # print(f'U-Net channel dimensions: {in_out}')

        self.time_mlp = nn.Sequential([sinusoidal_pos_emb(self.args.u_net_dim), nn.Dense(self.args.u_net_dim * 4), mish(), nn.Dense(self.args.u_net_dim)])

        # with each block/resolution, the state-action dimensionality increases while the horizon decreases (due to 'downsample1d')
        num_resolutions = len(in_out)
        self.downs = [[residual_temporal_block(dim_in, dim_out),
                       residual_temporal_block(dim_out, dim_out),
                       residual(dim_out, self.args.u_net_attention),
                       downsample1d(dim_out) if not (ind >= (num_resolutions - 1)) else lambda x: x] 
                       for ind, (dim_in, dim_out) in enumerate(in_out)]

        mid_dim = dims[-1]
        self.mid_block1 = residual_temporal_block(mid_dim, mid_dim)
        self.mid_attn = residual(mid_dim, self.args.u_net_attention)
        self.mid_block2 = residual_temporal_block(mid_dim, mid_dim)

        self.ups = [[residual_temporal_block(dim_out * 2, dim_in),
                     residual_temporal_block(dim_in, dim_in),
                     residual(dim_in, self.args.u_net_attention),
                     upsample1d(dim_in) if not ind >= (num_resolutions - 1) else lambda x: x] 
                     for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:]))]

        self.final_conv = nn.Sequential([conv1d_block(out_channels = self.args.u_net_dim, kernel_size = 5, n_groups = 8),
                                         nn.Conv(self.transition_dim, kernel_size = [1])])

    def __call__(self, x, time):
        '''
            x: [batch x horizon x transition]
        '''

        t = self.time_mlp(time)
        h = []
        for resnet, resnet2, attn, downsample in self.downs:
            
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            
            x = np.concatenate((x, h.pop()), axis = 2) # skip connection for aggregating multi-scale information
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        
        return x

class value_function(nn.Module):
    """
    based on https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
    """
    args: dict

    def setup(self):

        self.transition_dim = self.args.transition_dim

        dims = [self.transition_dim, *map(lambda m: self.args.u_net_dim * m, self.args.u_net_dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # print(f'U-Net channel dimensions: {in_out}')

        self.time_mlp = nn.Sequential([sinusoidal_pos_emb(self.args.u_net_dim), nn.Dense(self.args.u_net_dim * 4), mish(), nn.Dense(self.args.u_net_dim)])
        # self.goal_mlp = nn.Sequential([nn.Dense(self.args.u_net_dim), mish(), nn.Dense(self.args.u_net_dim * 4), mish(), nn.Dense(self.args.u_net_dim)])

        num_resolutions = len(in_out)
        self.blocks = [[residual_temporal_block(dim_in, dim_out),
                        residual_temporal_block(dim_out, dim_out),
                        downsample1d(dim_out)] 
                        for ind, (dim_in, dim_out) in enumerate(in_out)]

        horizon = self.args.horizon
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if not (ind >= (num_resolutions - 1)):
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4

        self.mid_block1 = residual_temporal_block(mid_dim, mid_dim_2)
        self.mid_down1 = downsample1d(mid_dim_2)
        horizon = horizon // 2

        self.mid_block2 = residual_temporal_block(mid_dim_2, mid_dim_3)
        self.mid_down2 = downsample1d(mid_dim_3)
        horizon = horizon // 2

        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential([nn.Dense(fc_dim // 2), mish(), nn.Dense(1)])

    def __call__(self, x, time):
        '''
            x: [batch x horizon x transition]
        '''

        t = self.time_mlp(time)
        # t = np.concatenate((self.time_mlp(time), self.goal_mlp(goal)), axis = -1)
        for resnet, resnet2, downsample in self.blocks:
            
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)

        x = x.reshape(len(x), -1)
        out = self.final_block(np.concatenate((x, t), axis = -1))
        
        return out
    
class value_function_with_goal(nn.Module):
    """
    based on https://github.com/jannerm/diffuser/blob/main/diffuser/models/temporal.py
    """
    args: dict

    def setup(self):

        self.transition_dim = self.args.transition_dim

        dims = [self.transition_dim, *map(lambda m: self.args.u_net_dim * m, self.args.u_net_dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # print(f'U-Net channel dimensions: {in_out}')

        self.time_mlp = nn.Sequential([sinusoidal_pos_emb(self.args.u_net_dim), nn.Dense(self.args.u_net_dim * 4), mish(), nn.Dense(self.args.u_net_dim)])
        self.goal_mlp = nn.Sequential([nn.Dense(self.args.u_net_dim), mish(), nn.Dense(self.args.u_net_dim * 4), mish(), nn.Dense(self.args.u_net_dim)])

        num_resolutions = len(in_out)
        self.blocks = [[residual_temporal_block(dim_in, dim_out),
                        residual_temporal_block(dim_out, dim_out),
                        downsample1d(dim_out)] 
                        for ind, (dim_in, dim_out) in enumerate(in_out)]

        horizon = self.args.horizon
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if not (ind >= (num_resolutions - 1)):
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4

        self.mid_block1 = residual_temporal_block(mid_dim, mid_dim_2)
        self.mid_down1 = downsample1d(mid_dim_2)
        horizon = horizon // 2

        self.mid_block2 = residual_temporal_block(mid_dim_2, mid_dim_3)
        self.mid_down2 = downsample1d(mid_dim_3)
        horizon = horizon // 2

        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential([nn.Dense(fc_dim // 2), mish(), nn.Dense(1)])

    def __call__(self, x, time, goal):
        '''
            x: [batch x horizon x transition]
        '''

        # t = self.time_mlp(time)
        t = np.concatenate((self.time_mlp(time), self.goal_mlp(goal)), axis = -1)
        for resnet, resnet2, downsample in self.blocks:
            
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)

        x = x.reshape(len(x), -1)
        out = self.final_block(np.concatenate((x, t), axis = -1))
        
        return out