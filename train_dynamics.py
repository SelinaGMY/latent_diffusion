from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from jax import numpy as jnp
from jax import random, jit, value_and_grad
rng = random.PRNGKey(0)
from flax.training.train_state import TrainState
from flax import linen as nn
from orbax.checkpoint import AsyncCheckpointer, PyTreeCheckpointHandler, CheckpointManager, CheckpointManagerOptions
options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
mngr = CheckpointManager('/nfs/ghome/live/mgao/latent_diffusion/dynamics', AsyncCheckpointer(PyTreeCheckpointHandler()), options)

import optax
import numpy as np
import matplotlib.pyplot as plt
import pickle

from data_process import trans_data_loader
from model.z_posterior import dynamics

def log_likelihood_diagonal_Gaussian(x, mu, log_var):
    log_likelihood = -0.5 * (log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var))
    return log_likelihood

def loss_fn(params, state, obs, action, control_var, control_indx):
    delta_y_mean, delta_y_log_var = state.apply_fn(params, obs, action)
    log_likelihood = log_likelihood_diagonal_Gaussian(control_var - obs[:,control_indx], delta_y_mean, delta_y_log_var)
    loss = -jnp.mean(jnp.sum(log_likelihood, axis=-1)) / len(control_var)
    return loss

@jit
def update_dynamics(state, obs, action, control_var, control_indx):
    loss, grads = value_and_grad(loss_fn)(state.params, state, obs, action, control_var, control_indx)
    state = state.apply_gradients(grads=grads)
    return loss, state

validate_loss = jit(loss_fn)

class Args():
    N_epoch = 100_000
    batch_size = 5_000
    train_ratio = 0.8
    h_dims_dynamics = [256,256]
    control_indx = [0]
    learning_rate = 1e-5
    adam_b1 = 0.9
    adam_b2 = 0.999
    adam_eps = 1e-8
    weight_decay = 10 # 0.01 
    max_grad_norm = 1
    ema_decay = 0.

args = Args()

options = {
    'fraction_for_validation': 1-args.train_ratio,
    'batch_size_validate': args.batch_size,
    'batch_size_train': args.batch_size,
    'tfds_shuffle_data': True,
    'tfds_seed': 42,
}

# load data
# train_trans, val_trans = trans_data_loader('/nfs/nhome/live/jheald/mingyang_latent_diffusion/data/obj', options, args.control_indx)
train_trans, val_trans = trans_data_loader('/nfs/ghome/live/mgao/latent_diffusion/data/obj_sigmoid_actions.pickle', options, args.control_indx) # data['actions'] = nn.sigmoid((data['actions']-0.5)*5.)

# initialisation
dynamics_model = dynamics(h_dims_dynamics=args.h_dims_dynamics,
                          control_variables=[args.control_indx])

obs = jnp.zeros((args.batch_size, 9))
action = jnp.zeros((args.batch_size, 6))
rng, step_rng = random.split(rng)
params_dynamics = dynamics_model.init(rng, obs, action)
tx = optax.chain(optax.adamw(learning_rate = args.learning_rate,
                            b1 = args.adam_b1,
                            b2 = args.adam_b2,
                            eps = args.adam_eps,
                            weight_decay = args.weight_decay),
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.ema(decay = args.ema_decay))
state_dynamics = TrainState.create(apply_fn = dynamics_model.apply, params = params_dynamics, tx = tx)

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
        batch_obs, batch_actions, batch_y_prime = next(shuffle_train)
        loss, state_dynamics = update_dynamics(state_dynamics, jnp.array(batch_obs), jnp.array(batch_actions), jnp.array(batch_y_prime), args.control_indx)
        losses.append(loss)
    mean_losses.append(jnp.mean(jnp.array(losses)))
    losses = []

    # validation step
    shuffle_val = iter(val_trans)
    for i in range(N_val_batch):
        val_batch_obs, val_batch_actions, val_batch_y_prime = next(shuffle_val)
        val_loss = validate_loss(state_dynamics.params, state_dynamics, jnp.array(val_batch_obs), jnp.array(val_batch_actions), jnp.array(val_batch_y_prime), args.control_indx)
        val_losses.append(val_loss)
    mean_val_losses.append(jnp.mean(jnp.array(val_losses)))
    val_losses = []

    if (k+1) % 100 == 0:
        print("Dynamics: epoch %d \t, Train Loss %f " % (k+1, mean_losses[-1]))
        print("Dynamics: epoch %d \t, Validation Loss %f " % (k+1, mean_val_losses[-1]))
        mngr.save(k+1, state_dynamics , metrics=np.array(mean_val_losses[-1]).tolist())

mngr.wait_until_finished()
state_dynamics = mngr.restore(mngr.best_step())
print(mngr.best_step())

with open('/nfs/ghome/live/mgao/latent_diffusion/dynamics/params_dynamics_0822.pickle', 'wb') as handle:
    pickle.dump(state_dynamics, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.figure()
plt.plot(mean_losses, label = 'train_loss')
plt.plot(mean_val_losses, label = 'validation_loss')
plt.legend()
plt.title('Loss: dynamic model')
plt.savefig('/nfs/ghome/live/mgao/latent_diffusion/dynamics/dynamic_0822.pdf')

np.save('/nfs/ghome/live/mgao/latent_diffusion/dynamics/dyn_loss_0822.npy', np.array(mean_losses))
np.save('/nfs/ghome/live/mgao/latent_diffusion/dynamics/dyn_val_loss_0822.npy', np.array(mean_val_losses))