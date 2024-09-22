from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from jax import numpy as jnp
from jax import random, jit, value_and_grad
rng = random.PRNGKey(0)
from flax.training.train_state import TrainState
from flax import linen as nn
from orbax.checkpoint import AsyncCheckpointer, PyTreeCheckpointHandler, CheckpointManager, CheckpointManagerOptions
options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
mngr = CheckpointManager('/nfs/ghome/live/mgao/latent_diffusion/inverse_dynamics', AsyncCheckpointer(PyTreeCheckpointHandler()), options)

import optax
import numpy as np
import matplotlib.pyplot as plt
import pickle

from config import inverse_dynamics_args
from data_process import full_trans_data_loader
from model.z_posterior import MLP_Gaussian

def log_likelihood_diagonal_Gaussian(x, mu, log_var):
    log_likelihood = -0.5 * (log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var))
    return log_likelihood

def loss_fn(params, state, obs, action, next_obs):
    action_mean, action_log_var = state.apply_fn(params, jnp.concatenate((obs, next_obs),-1))
    log_likelihood = log_likelihood_diagonal_Gaussian(action, action_mean, action_log_var)
    loss = -jnp.mean(log_likelihood)
    return loss

@jit
def update_inverse_dynamics(state, obs, action, next_obs):
    loss, grads = value_and_grad(loss_fn)(state.params, state, obs, action, next_obs)
    state = state.apply_gradients(grads=grads)
    return loss, state

validate_loss = jit(loss_fn)

args = inverse_dynamics_args()

options = {
    'fraction_for_validation': 1-args.train_ratio,
    'batch_size_validate': args.batch_size,
    'batch_size_train': args.batch_size,
    'tfds_shuffle_data': True,
    'tfds_seed': 42,
}

# load data
train_trans, val_trans = full_trans_data_loader('/nfs/ghome/live/mgao/latent_diffusion/data/obj', options, args.control_indx) # data['actions'] = nn.sigmoid((data['actions']-0.5)*5.)

# initialisation
inverse_dynamics_model = MLP_Gaussian(h_dims=args.h_dims_inverse_dynamics,
                              out_dim=6)

obs = jnp.zeros((args.batch_size, 9))
rng, step_rng = random.split(rng)
params_inverse_dynamics = inverse_dynamics_model.init(rng, jnp.concatenate((obs, obs),-1))
tx = optax.chain(optax.adamw(learning_rate = args.learning_rate,
                            b1 = args.adam_b1,
                            b2 = args.adam_b2,
                            eps = args.adam_eps,
                            weight_decay = args.weight_decay),
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.ema(decay = args.ema_decay))
state_inverse_dynamics = TrainState.create(apply_fn = inverse_dynamics_model.apply, params = params_inverse_dynamics, tx = tx)

# train the model
train_size = 1_000 * 25 * 0.8 # args.train_ratio
val_size = 1_000 * 25 * 0.2 # (1-args.train_ratio)
N_train_batch = int(train_size // args.batch_size)
N_val_batch = int(val_size // args.batch_size)

losses = []
mean_losses = []
val_losses = []
mean_val_losses = []

for k in range(args.N_epoch):
    
    # train step
    shuffle_train = iter(train_trans)
    for i in range(N_train_batch):
        batch_obs, batch_actions, batch_obs_prime = next(shuffle_train)
        loss, state_inverse_dynamics = update_inverse_dynamics(state_inverse_dynamics, jnp.array(batch_obs), jnp.array(batch_actions), jnp.array(batch_obs_prime))
        losses.append(loss)
    mean_losses.append(jnp.mean(jnp.array(losses)))
    losses = []

    # validation step
    shuffle_val = iter(val_trans)
    for i in range(N_val_batch):
        val_batch_obs, val_batch_actions, val_batch_obs_prime = next(shuffle_val)
        val_loss = validate_loss(state_inverse_dynamics.params, state_inverse_dynamics, jnp.array(val_batch_obs), jnp.array(val_batch_actions), jnp.array(val_batch_obs_prime))
        val_losses.append(val_loss)
    mean_val_losses.append(jnp.mean(jnp.array(val_losses)))
    val_losses = []

    if (k+1) % 100 == 0:
        print("Dynamics: epoch %d \t, Train Loss %f " % (k+1, mean_losses[-1]))
        print("Dynamics: epoch %d \t, Validation Loss %f " % (k+1, mean_val_losses[-1]))
        mngr.save(k+1, state_inverse_dynamics , metrics=np.array(mean_val_losses[-1]).tolist())

mngr.wait_until_finished()
state_inverse_dynamics = mngr.restore(mngr.best_step())
print(mngr.best_step())

with open('/nfs/ghome/live/mgao/latent_diffusion/inverse_dynamics/params_inverse_dynamics_0822.pickle', 'wb') as handle:
    pickle.dump(state_inverse_dynamics, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.figure()
plt.plot(mean_losses, label = 'train_loss')
plt.plot(mean_val_losses, label = 'validation_loss')
plt.legend()
plt.title('Loss: dynamic model')
plt.savefig('/nfs/ghome/live/mgao/latent_diffusion/inverse_dynamics/inverse_dynamic_0822.pdf')

np.save('/nfs/ghome/live/mgao/latent_diffusion/inverse_dynamics/inv_dyn_loss_0822.npy', np.array(mean_losses))
np.save('/nfs/ghome/live/mgao/latent_diffusion/inverse_dynamics/inv_dyn_val_loss_0822.npy', np.array(mean_val_losses))