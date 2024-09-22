import optax
from flax.training import train_state
from jax import vmap, random
rng = random.PRNGKey(2024)
from jax import numpy as jnp

import pickle 
import numpy as np
from model.z_posterior import MLP_Gaussian

from config import VAE_args

args = VAE_args()

def get_trans_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        actions = data['actions'][:,:26,:]
        norm_actions = (actions - actions.mean(axis=(0,1),keepdims=True)) / actions.std(axis=(0,1),keepdims=True)

    return norm_actions[:,:25,:].reshape((-1,6))

actions = get_trans_data("/nfs/ghome/live/mgao/latent_diffusion/data/obj")

with open("/nfs/ghome/live/mgao/latent_diffusion/VAE/params_VAE.pickle", "rb") as input_file:
    params_VAE = pickle.load(input_file)['params']
    encoder_model = MLP_Gaussian(h_dims = args.h_dims_encoder,
                                 out_dim = len(args.control_indx))
    decoder_model = MLP_Gaussian(h_dims = args.h_dims_decoder,
                                 out_dim = 6)

z_mean, z_log_var = encoder_model.apply({'params': params_VAE['params']['encoder']}, actions)
a_mean, a_log_var = decoder_model.apply({'params': params_VAE['params']['decoder']}, z_mean)

np.save('/nfs/ghome/live/mgao/latent_diffusion/VAE/new_1daction.npy', np.array(z_mean).reshape((1000,25,1)))
np.save('/nfs/ghome/live/mgao/latent_diffusion/VAE/new_6daction.npy', np.array(a_mean).reshape((1000,25,6)))