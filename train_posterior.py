from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from jax import numpy as jnp
from jax import random, value_and_grad, jit, tree_map
rng = random.PRNGKey(2024)
from flax.training.train_state import TrainState
import optax
import haiku as hk
from orbax.checkpoint import AsyncCheckpointer, PyTreeCheckpointHandler, CheckpointManager, CheckpointManagerOptions
options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
mngr = CheckpointManager('/nfs/ghome/live/mgao/latent_diffusion/empowerment', AsyncCheckpointer(PyTreeCheckpointHandler()), options)

import pickle
import matplotlib.pyplot as plt
import numpy as np

from data_process import get_mean_std, obs_data_loader
from model.z_posterior import train_posterior, dynamics, reconstruction
from config import posterior_args

from flax import linen as nn

def loss_fn(params, state_pos, state_dynamics, obs, y_prime, rng):
    posterior_loss, kl = state_pos.apply_fn(params, obs, y_prime, state_dynamics, rng)
    return posterior_loss.mean(), kl

@jit
def update(state_pos, state_dynamics, obs, y_prime, rng):
    (loss, kl), grads = value_and_grad(loss_fn, has_aux=True)(state_pos.params, state_pos, state_dynamics, obs, y_prime, rng)
    state_pos = state_pos.apply_gradients(grads=grads)
    return loss, state_pos, kl

validate_loss = jit(loss_fn)

# load args and options
args = posterior_args()
options = {
    'fraction_for_validation': 1-args.train_ratio,
    'batch_size_validate': args.batch_size,
    'batch_size_train': args.batch_size,
    'tfds_shuffle_data': True,
    'tfds_seed': 42,
}

train_obs, val_obs = obs_data_loader('/nfs/ghome/live/mgao/latent_diffusion/data/obj_sigmoid_actions.pickle', options, args.control_indx)

# load trained model
with open("/nfs/ghome/live/mgao/latent_diffusion/dynamics/params_dynamics_0822.pickle", "rb") as input_file:
    params_dyn = pickle.load(input_file)
    dynamics_model = dynamics(args.h_dims_dynamics, args.control_indx)

    # random params
    # obs = jnp.zeros((args.batch_size, 9))
    # action = jnp.zeros((args.batch_size, 6))
    # rng, step_rng = random.split(rng)
    # params_dyn = {'params': dynamics_model.init(rng, obs, action)}

    tx = optax.chain(optax.adamw(learning_rate = args.learning_rate,
                        b1 = args.adam_b1,
                        b2 = args.adam_b2,
                        eps = args.adam_eps,
                        weight_decay = args.weight_decay),
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.ema(decay = args.ema_decay))
    state_dynamics = TrainState.create(apply_fn = dynamics_model.apply, params = params_dyn['params'], tx = tx)

# posterior p(z|y',s) used to infer the 1D latent action z
# initialisation
mean, std = get_mean_std('/nfs/ghome/live/mgao/latent_diffusion/data/obj_sigmoid_actions.pickle', 'actions')
posterior_model = train_posterior(h_dims_posterior = args.h_dims_posterior,
                                  control_variables = args.control_indx,
                                  mean_action = mean.reshape(6),
                                  std_action = std.reshape(6))


rng, step_rng = random.split(rng)
obs = jnp.zeros((args.batch_size, 9))
y_prime = jnp.zeros((args.batch_size, len(args.control_indx)))
params_posterior = posterior_model.init(rng, obs, y_prime, state_dynamics, rng)

mask = hk.data_structures.map(lambda module_name, name, value: name == 'power_param', params_posterior)
not_mask = tree_map(lambda x: not x, mask)


tx_q = optax.chain(optax.adamw(learning_rate = args.learning_rate,
                            b1 = args.adam_b1,
                            b2 = args.adam_b2,
                            eps = args.adam_eps,
                            weight_decay = args.weight_decay),
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.ema(decay = args.ema_decay))

# don't use weight decay on power param
tx_power = optax.chain(optax.adamw(learning_rate = args.power_learning_rate,
                            b1 = args.adam_b1,
                            b2 = args.adam_b2,
                            eps = args.adam_eps,
                            weight_decay = args.power_weight_decay),
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.ema(decay = args.ema_decay))

state_posterior = TrainState.create(apply_fn = posterior_model.apply, params = params_posterior, 
        tx = optax.chain(
        # only applies to leaves where mask is True:
        optax.masked(tx_power, mask), 
        # only applies to leaves where not_mask is True:
        optax.masked(tx_q, not_mask)
        ) )

def get_true_delta_y(filename, control_indx):

    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        obs = data['obs'][:,:26,:]

        # normalisation
        norm_obs = (obs - obs.mean(axis=(0,1))) / obs.std(axis=(0,1))

        norm_delta_y = (norm_obs[:,1:,control_indx] - norm_obs[:,:25,control_indx]).reshape((-1,1))

    return norm_delta_y

# useful for comparing delta_y in dataset with delta_y's sampled in train_posterior
# for the posterior to be useful, they should be similar
true_delta_y = get_true_delta_y('/nfs/ghome/live/mgao/latent_diffusion/data/obj_sigmoid_actions.pickle', args.control_indx)

reconstruction_model = reconstruction(h_dims_posterior = args.h_dims_posterior,
                                      control_variables = args.control_indx,
                                      mean_action = mean.reshape(6),
                                      std_action = std.reshape(6))

reconstruction_jit = jit(reconstruction_model.apply)

# train
train_size = 1_000 * 25 * 0.8 # args.train_ratio
val_size = 1_000 * 25 * 0.2 # (1-args.train_ratio)
N_train_batch = int(train_size // args.batch_size)
N_val_batch = int(val_size // args.batch_size)

losses = []
mean_losses = []
kls = []
mean_kls = []
reconstruction_errors = []
mean_reconstruction_errors = []
val_losses = []
val_kls = []
mean_val_losses = []
mean_val_kls = []
val_reconstruction_errors = []
mean_val_reconstruction_errors = []

for k in range(args.N_epoch):
    # train step
    shuffle_train = iter(train_obs)
    for i in range(N_train_batch):
        batch_obs, batch_y_prime = next(shuffle_train)
        loss, state_posterior, kl = update(state_posterior, state_dynamics, jnp.array(batch_obs), jnp.array(batch_y_prime), step_rng)
        losses.append(loss)
        kls.append(kl)
        reconstruction_error = reconstruction_jit(state_posterior.params, jnp.array(batch_obs), jnp.array(batch_y_prime), state_dynamics)
        reconstruction_errors.append(reconstruction_error)
    mean_losses.append(jnp.mean(jnp.array(losses)))
    mean_kls.append(jnp.mean(jnp.array(kls)))
    mean_reconstruction_errors.append(jnp.mean(jnp.array(reconstruction_errors)))
    losses = []
    kls = []
    reconstruction_errors = []

    # validation step
    shuffle_val = iter(val_obs)
    for i in range(N_val_batch):
        val_batch_obs, val_batch_y_prime = next(shuffle_val)
        val_loss, val_kl = validate_loss(state_posterior.params, state_posterior, state_dynamics, jnp.array(val_batch_obs), jnp.array(val_batch_y_prime), step_rng)
        val_losses.append(val_loss)
        val_kls.append(val_kl)
        val_reconstruction_error = reconstruction_jit(state_posterior.params, jnp.array(val_batch_obs), jnp.array(val_batch_y_prime), state_dynamics)
        val_reconstruction_errors.append(val_reconstruction_error)
    mean_val_losses.append(jnp.mean(jnp.array(val_losses)))
    mean_val_kls.append(jnp.mean(jnp.array(val_kls)))
    mean_val_reconstruction_errors.append(jnp.mean(jnp.array(val_reconstruction_errors)))
    val_losses = []
    val_kls = []
    val_reconstruction_errors = []

    if (k+1) % 100 == 0:
        print("Posterior: epoch %d \t, Train Loss %f " % (k+1, mean_losses[-1]))
        print("Posterior: epoch %d \t, Train log p(y'|x,z) %f " % (k+1, -mean_losses[-1] + mean_kls[-1]))
        print("Posterior: epoch %d \t, Train kl(q||p) %f " % (k+1, mean_kls[-1]))
        print("Posterior: epoch %d \t, Train Reconstruction Error %f " % (k+1, mean_reconstruction_errors[-1]))
        print("Posterior: epoch %d \t, Validation Loss %f " % (k+1, mean_val_losses[-1]))
        print("Posterior: epoch %d \t, Validation log p(y'|x,z) %f " % (k+1, -mean_val_losses[-1] + mean_val_kls[-1]))
        print("Posterior: epoch %d \t, Validation kl(q||p) %f " % (k+1, mean_val_kls[-1]))
        print("Posterior: epoch %d \t, Validation Reconstruction Error %f " % (k+1, mean_val_reconstruction_errors[-1]))
        print("Posterior: epoch %d \t, Std z %f " % (k+1, nn.softplus(state_posterior.params['params']['power_param'][0])))
        mngr.save(k+1, state_posterior , metrics=np.array(mean_val_losses[-1]).tolist())

mngr.wait_until_finished()
state_dynamics = mngr.restore(mngr.best_step())
print(mngr.best_step())

with open('/nfs/ghome/live/mgao/latent_diffusion/empowerment/params_posterior.pickle', 'wb') as handle:
    pickle.dump(state_posterior.params, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.figure()
plt.plot(mean_losses, label = 'train_loss')
plt.plot(mean_val_losses, alpha = 0.3, label = 'validation_loss')
plt.gca().set_ylim(top=10.)
plt.legend()
plt.title('Loss: posterior model')
plt.savefig('//nfs/ghome/live/mgao/latent_diffusion/empowerment/posterior.pdf')

plt.figure()
plt.plot(mean_reconstruction_errors, label = 'train_reconstruction_error')
plt.plot(mean_val_reconstruction_errors, label = 'validation_reconstruction_error')
plt.legend()
plt.title('Reconstruction error: posterior model')
plt.savefig('/nfs/ghome/live/mgao/latent_diffusion/empowerment/reconstruction_error.pdf')

plt.figure()
plt.plot(-jnp.array(mean_losses) + jnp.array(mean_kls), label = 'train_log_p(y''|x,z)')
plt.plot(-jnp.array(mean_val_losses) + jnp.array(mean_val_kls), label = 'validation_log_p(y''|x,z)')
plt.gca().set_ylim(bottom=-10.)
plt.legend()
plt.title('Decoder likelihood: posterior model')
plt.savefig('/nfs/ghome/live/mgao/latent_diffusion/empowerment/decoder_likelihood.pdf')

plt.figure()
plt.plot(mean_kls, label = 'train_kl(q||p)')
plt.plot(mean_val_kls, label = 'validation_kl(q||p)')
plt.legend()
plt.title('KL(q||p): posterior model')
plt.savefig('/nfs/ghome/live/mgao/latent_diffusion/empowerment/kl.pdf')

np.save('//nfs/ghome/live/mgao/latent_diffusion/empowerment/pos_loss.npy', np.array(mean_losses))
np.save('/nfs/ghome/live/mgao/latent_diffusion/empowerment/pos_val_loss.npy', np.array(mean_val_losses))
np.save('/nfs/ghome/live/mgao/latent_diffusion/empowerment/pos_reconstruction_error.npy', np.array(mean_reconstruction_errors))
np.save('/nfs/ghome/live/mgao/latent_diffusion/empowerment/pos_val_reconstruction_error.npy', np.array(mean_val_reconstruction_errors))
np.save('/nfs/ghome/live/mgao/latent_diffusion/empowerment/pos_kl.npy', np.array(mean_kls))
np.save('/nfs/ghome/live/mgao/latent_diffusion/empowerment/pos_val_kl.npy', np.array(mean_val_kls))