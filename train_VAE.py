
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from jax import numpy as jnp
from jax import random, value_and_grad, jit, tree_map
rng = random.PRNGKey(2024)
from flax.training.train_state import TrainState
import optax
import haiku as hk
from orbax.checkpoint import AsyncCheckpointer, PyTreeCheckpointHandler, CheckpointManager, CheckpointManagerOptions
options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
mngr = CheckpointManager('/nfs/ghome/live/mgao/latent_diffusion/VAE', AsyncCheckpointer(PyTreeCheckpointHandler()), options)

import pickle
import matplotlib.pyplot as plt
import numpy as np

from data_process import trans_data_loader
from model.z_posterior import train_VAE
from config import VAE_args

from flax import linen as nn

def loss_fn(params, state_VAE, actions, rng):
    VAE_loss, kl = state_VAE.apply_fn(params, actions, rng)
    return VAE_loss.mean(), kl

@jit
def update(state_VAE, actions, rng):
    (loss, kl), grads = value_and_grad(loss_fn, has_aux=True)(state_VAE.params, state_VAE, actions, rng)
    state_VAE = state_VAE.apply_gradients(grads=grads)
    return loss, state_VAE, kl

validate_loss = jit(loss_fn)

# load args and options
args = VAE_args()
options = {
    'fraction_for_validation': 1-args.train_ratio,
    'batch_size_validate': args.batch_size,
    'batch_size_train': args.batch_size,
    'tfds_shuffle_data': True,
    'tfds_seed': 42,
}

train_trans, val_trans = trans_data_loader('/nfs/ghome/live/mgao/latent_diffusion/data/obj', options, args.control_indx) # data['actions'] = nn.sigmoid((data['actions']-0.5)*5.)

VAE_model = train_VAE(h_dims_encoder = args.h_dims_encoder,
                      h_dims_decoder = args.h_dims_decoder,
                      control_variables = args.control_indx,
                      action_dim = 6)

rng, step_rng = random.split(rng)
actions = jnp.zeros((args.batch_size, 6))
params_VAE = VAE_model.init(rng, actions, rng)
tx = optax.chain(optax.adamw(learning_rate = args.learning_rate,
                            b1 = args.adam_b1,
                            b2 = args.adam_b2,
                            eps = args.adam_eps,
                            weight_decay = args.weight_decay),
                            optax.clip_by_global_norm(args.max_grad_norm),
                            optax.ema(decay = args.ema_decay))

state_VAE = TrainState.create(apply_fn = VAE_model.apply, params = params_VAE, tx = tx)

# train
train_size = 1_000 * 25 * 0.8 # args.train_ratio
val_size = 1_000 * 25 * 0.2 # (1-args.train_ratio)
N_train_batch = int(train_size // args.batch_size)
N_val_batch = int(val_size // args.batch_size)

losses = []
mean_losses = []
kls = []
mean_kls = []
val_losses = []
val_kls = []
mean_val_losses = []
mean_val_kls = []

for k in range(args.N_epoch):
    # train step
    shuffle_train = iter(train_trans)
    for i in range(N_train_batch):
        _, batch_actions, _ = next(shuffle_train)
        loss, state_VAE, kl = update(state_VAE, jnp.array(batch_actions), step_rng)
        losses.append(loss)
        kls.append(kl)
    mean_losses.append(jnp.mean(jnp.array(losses)))
    mean_kls.append(jnp.mean(jnp.array(kls)))
    losses = []
    kls = []

    # validation step
    shuffle_val = iter(val_trans)
    for i in range(N_val_batch):
        _, val_batch_actions, _ = next(shuffle_val)
        val_loss, val_kl = validate_loss(state_VAE.params, state_VAE, jnp.array(val_batch_actions), step_rng)
        val_losses.append(val_loss)
        val_kls.append(val_kl)
    mean_val_losses.append(jnp.mean(jnp.array(val_losses)))
    mean_val_kls.append(jnp.mean(jnp.array(val_kls)))
    val_losses = []
    val_kls = []

    if (k+1) % 100 == 0:
        print("Posterior: epoch %d \t, Train Loss %f " % (k+1, mean_losses[-1]))
        print("Posterior: epoch %d \t, Train log p(y'|x,z) %f " % (k+1, -mean_losses[-1] + mean_kls[-1]))
        print("Posterior: epoch %d \t, Train kl(q||p) %f " % (k+1, mean_kls[-1]))
        print("Posterior: epoch %d \t, Validation Loss %f " % (k+1, mean_val_losses[-1]))
        print("Posterior: epoch %d \t, Validation log p(y'|x,z) %f " % (k+1, -mean_val_losses[-1] + mean_val_kls[-1]))
        print("Posterior: epoch %d \t, Validation kl(q||p) %f " % (k+1, mean_val_kls[-1]))
        mngr.save(k+1, state_VAE , metrics=np.array(mean_val_losses[-1]).tolist())

mngr.wait_until_finished()
state_VAE = mngr.restore(mngr.best_step())
print(mngr.best_step())

with open('/nfs/ghome/live/mgao/latent_diffusion/VAE/params_VAE.pickle', 'wb') as handle:
    pickle.dump(state_VAE, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.figure()
plt.plot(mean_losses, label = 'train_loss')
plt.plot(mean_val_losses, alpha = 0.3, label = 'validation_loss')
plt.gca().set_ylim(top=10.)
plt.legend()
plt.title('Loss: VAE model')
plt.savefig('/nfs/ghome/live/mgao/latent_diffusion/VAE/VAE.pdf')

plt.figure()
plt.plot(-jnp.array(mean_losses) + jnp.array(mean_kls), label = 'train_log_p(y''|x,z)')
plt.plot(-jnp.array(mean_val_losses) + jnp.array(mean_val_kls), label = 'validation_log_p(y''|x,z)')
plt.gca().set_ylim(bottom=-10.)
plt.legend()
plt.title('Decoder likelihood: VAE model')
plt.savefig('/nfs/ghome/live/mgao/latent_diffusion/VAE/decoder_likelihood.pdf')

plt.figure()
plt.plot(mean_kls, label = 'train_kl(q||p)')
plt.plot(mean_val_kls, label = 'validation_kl(q||p)')
plt.legend()
plt.title('KL(q||p): VAE model')
plt.savefig('/nfs/ghome/live/mgao/latent_diffusion/VAE/kl.pdf')

np.save('/nfs/ghome/live/mgao/latent_diffusion/VAE/VAE_loss.npy', np.array(mean_losses))
np.save('/nfs/ghome/live/mgao/latent_diffusion/VAE/VAE_val_loss.npy', np.array(mean_val_losses))
np.save('/nfs/ghome/live/mgao/latent_diffusion/VAE/VAE_kl.npy', np.array(mean_kls))
np.save('/nfs/ghome/live/mgao/latent_diffusion/VAE/VAE_val_kl.npy', np.array(mean_val_kls))