from jax import numpy as np
from flax import linen as nn
import einops
from einops.layers.flax import Rearrange
from flax.linen.initializers import zeros_init, ones_init # https://flax.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.param
import math
import argparse

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class residual_temporal_block(nn.Module):
    inp_channels: int
    out_channels: int

    def setup(self):
        
        self.blocks = [conv1d_block(self.out_channels, kernel_size = 5, n_groups = 8), 
                       conv1d_block(self.out_channels, kernel_size = 5, n_groups = 8)]
        
        self.time_mlp = nn.Sequential([mish(), nn.Dense(self.out_channels), Rearrange('batch t -> batch 1 t')])
        
        self.residual_conv = nn.Conv(self.out_channels, kernel_size = [1]) if self.inp_channels != self.out_channels else lambda x: x

    def __call__(self, x, t):
        '''
            x: [batch_size x horizon x inp_channels]
            t: [batch_size x embed_dim]
            returns:
            out: [batch_size x horizon x out_channels]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)

class mish(nn.Module):

    @nn.compact
    def __call__(self, x):

        return x * nn.tanh(nn.softplus(x))

class sinusoidal_pos_emb(nn.Module):
    dim: int

    def __call__(self, x):
        
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = np.concatenate((np.sin(emb), np.cos(emb)), axis = -1)

        return emb

class downsample1d(nn.Module):
    dim: int

    def setup(self):
        
        self.conv = nn.Conv(self.dim, kernel_size = [3], strides = 2, padding = 1)

    def __call__(self, x):

        return self.conv(x)

class upsample1d(nn.Module):
    dim: int
    
    def setup(self):
        
        self.conv = nn.ConvTranspose(self.dim, kernel_size = 4, strides = [2], padding = 2)

    def __call__(self, x):

        return self.conv(x)

class conv1d_block(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''
    out_channels: int
    kernel_size: int
    n_groups: int

    def setup(self):

        self.block = nn.Sequential([nn.Conv(self.out_channels, kernel_size = [self.kernel_size], padding = self.kernel_size // 2),
                                    Rearrange('batch channels horizon -> batch channels 1 horizon'),
                                    nn.GroupNorm(num_groups = self.n_groups),
                                    Rearrange('batch channels 1 horizon -> batch channels horizon'),
                                    mish()])

    def __call__(self, x):

        return self.block(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class residual(nn.Module):
    dim: int
    attention: bool

    def setup(self):

        if self.attention:

            self.fn = pre_norm(self.dim)

        else:

            self.fn = lambda x: x

    def __call__(self, x):

        return self.fn(x) + x

class layer_norm(nn.Module):
    dim: int

    def setup(self, eps = 1e-5):

        self.eps = eps
        self.g = self.param('layer_norm_gamma', ones_init(), (1, 1, self.dim))
        self.b = self.param('layer_norm_bias', zeros_init(), (1, 1, self.dim))

    def __call__(self, x):

        x_var = np.var(x, axis = 2, keepdims = True)
        x_mean = np.mean(x, axis = 2, keepdims = True)

        return (x - x_mean) / np.sqrt(x_var + self.eps) * self.g + self.b

class pre_norm(nn.Module):
    dim: int

    def setup(self):
        
        self.layer_norm = layer_norm(self.dim)
        self.linear_attention = linear_attention(self.dim)

    def __call__(self, x):

        x = self.layer_norm(x)

        return self.linear_attention(x)

class linear_attention(nn.Module):
    dim: int

    def setup(self):

        self.heads = 4
        dim_head = 32
        hidden_dim = dim_head * self.heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Conv(hidden_dim * 3, kernel_size = [1], use_bias = False)
        self.to_out = nn.Conv(self.dim, kernel_size = [1])

    def __call__(self, x):
        
        qkv = np.split(self.to_qkv(x), 3, axis = 2)
        q, k, v = map(lambda t: einops.rearrange(t, 'b c (h d) -> b c h d', h = self.heads), qkv)
        q = q * self.scale

        k = nn.softmax(k, axis = 1)
        context = np.einsum('b n h d, b n h e -> b h d e', k, v)

        out = np.einsum('b h d e, b n h d -> b n h e', context, q)
        out = einops.rearrange(out, 'b c h d -> b c (h d)')
        
        return self.to_out(out)