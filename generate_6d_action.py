import optax
from flax.training import train_state
from jax import vmap, random
rng = random.PRNGKey(2024)
from jax import numpy as jnp

import pickle 
import numpy as np
from model.z_posterior import dynamics, infer, precoder

class Args:
    N_epochs = 50_000
    batch_size = 200
    early_stop_start = 150_000
    train_ratio = 0.8
    seed = 111
    control_indx = [0]
    h_dims_dynamics = [256,256]
    horizon = 4  # time steps
    transition_dim = 15  # spatial dimensionality
    u_net_dim = 32
    u_net_dim_mults = [1, 4, 8]
    u_net_attention = True
    learning_rate = 2e-4
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_eps = 1e-8
    weight_decay = 0 # 0.0001
    max_grad_norm = 1
    ema_decay = 0.
    gamma = 0.997

args = Args()

def get_trans_data(filename, control_indx):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        obs = data['obs'][:,:26,:]
        norm_obs = (obs - obs.mean(axis=(0,1),keepdims=True)) / obs.std(axis=(0,1),keepdims=True)
        trans_obs = norm_obs[:,:25,:].reshape((-1,9))
        trans_y_prime = norm_obs[:,1:26,control_indx].reshape((-1,1))

    return trans_obs, trans_y_prime

def get_mean_std(filename,var_name):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        var = data[var_name][:,:26,:]
        length = var.shape[-1]
        mean = var.mean(axis=(0,1)).reshape((-1,length))
        std = var.std(axis=(0,1)).reshape((-1,length))

    return mean, std

action_mean, action_std = get_mean_std("/nfs/ghome/live/mgao/latent_diffusion/data/obj_sigmoid_actions.pickle",'actions')
obs_mean, _ = get_mean_std("/nfs/ghome/live/mgao/latent_diffusion/data/obj_sigmoid_actions.pickle",'obs')
trans_obs, trans_y_prime = get_trans_data("/nfs/ghome/live/mgao/latent_diffusion/data/obj_sigmoid_actions.pickle", args.control_indx)

with open("/nfs/ghome/live/mgao/latent_diffusion/dynamics/params_dynamics_0822.pickle", "rb") as input_file:
    params_dyn = pickle.load(input_file)
    dynamics_model = dynamics(args.h_dims_dynamics, args.control_indx)
    tx = optax.chain(optax.adamw(learning_rate = args.learning_rate,
                        b1 = args.adam_b1,
                        b2 = args.adam_b2,
                        eps = args.adam_eps,
                        weight_decay = args.weight_decay),
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.ema(decay = args.ema_decay))
    state_dynamics = train_state.TrainState.create(apply_fn = dynamics_model.apply, params = params_dyn['params'], tx = tx)

with open("/nfs/ghome/live/mgao/latent_diffusion/empowerment/params_posterior.pickle", "rb") as input_file:
    params_pos = pickle.load(input_file)
    posterior = infer(h_dims_posterior = args.h_dims_dynamics,
                      control_variables = args.control_indx)

my_precoder = precoder(action_mean.reshape(6), action_std.reshape(6))
synergy = vmap(my_precoder.apply, in_axes=(None,0,None))({}, trans_obs, state_dynamics)
z_mean, z_log_var = posterior.apply({'params': params_pos['params']['posterior_model']}, trans_obs, trans_y_prime)
new_action = jnp.einsum("...ij,...j->...i", synergy, z_mean).reshape((1000,25,6))

# from matplotlib import pyplot as plt

# zr = z_mean.reshape((1000,25))
# plt.plot(zr[0,:],'b')
# # plt.plot(z_mean[:1000,0])
# plt.plot(z_mean[:25,0],'r--')
# plt.show()

# breakpoint()

# dots = [jnp.inner(synergy[i,:,0],synergy[i+1,:,0]) for i in range(1000)]
# plt.hist(dots)
# plt.show()

# breakpoint()

# with open("/nfs/ghome/live/mgao/latent_diffusion/data/obj", 'rb') as handle:
#     data = pickle.load(handle)
#     actions = data['actions'][:,:25,:].reshape((-1,6))

# z_action = jnp.einsum("...i,...i->...", synergy[:,:,0], actions).reshape((1000,25))
# plt.plot(z_action[0,:],'r--')
# plt.show()

# breakpoint()

np.save('/nfs/ghome/live/mgao/latent_diffusion/empowerment/new_1daction.npy', np.array(z_mean).reshape((1000,25,1)))
np.save('/nfs/ghome/live/mgao/latent_diffusion/empowerment/new_6daction.npy', new_action)