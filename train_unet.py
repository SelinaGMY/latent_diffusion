import jax.numpy as jnp
from jax import jit, value_and_grad
import jax.random as random
from functools import partial
import optax
from flax.training import train_state
from flax.training.early_stopping import EarlyStopping
from flax.serialization import to_state_dict
from orbax.checkpoint import AsyncCheckpointer, PyTreeCheckpointHandler, CheckpointManager, CheckpointManagerOptions
options = CheckpointManagerOptions(max_to_keep=3, best_fn=lambda metrics: metrics, best_mode='min')
rng = random.PRNGKey(2024)

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'false'

from model.temporal import temporal_unet
from data_process import new_traj_data_loader, state_only_traj_data_loader # get_new_traj_data
from utils import mean_factor, var

import jax
print(jax.devices())

# model_type = 'empowerment'
# model_type = 'synergy_manifold'
# model_type = 'VAE'
model_type = 'VAE_6d'
# model_type = 'state_only'
# model_type = 'baseline'

def lossfun_score(params, model, rng, batch):

    rng, step_rng = random.split(rng)
    N_batch = batch.shape[0]
    t = random.uniform(step_rng, (N_batch, ), minval=1e-5, maxval=1)
    mean_coeff = mean_factor(t)

    vs = var(t)
    stds = jnp.sqrt(vs)

    rng, step_rng = random.split(rng)
    noise = random.normal(step_rng, batch.shape)

    xt = batch * mean_coeff[:,None,None] + noise * stds[:,None,None]
    xt = xt.at[:,0,:9].set(batch[:,0,:9])

    output = model.apply(params, xt, t*999)

    score_loss = (noise + output * stds[:,None,None])**2
    score_loss = score_loss.at[:,0,:9].set(jnp.zeros(score_loss[:,0,:9].shape))
    final_loss = jnp.mean(score_loss)

    return final_loss

@partial(jit, static_argnums=[3])
def update_score(rng, batch, opt_state, model):
    vals, grads = value_and_grad(lossfun_score)(opt_state.params, model, rng, batch)
    opt_state = opt_state.apply_gradients(grads = grads)
    return vals, opt_state

@partial(jit, static_argnums=[3])
def evaluate_score(rng, batch, state, model):
    loss = lossfun_score(state.params, model, rng, batch)
    return loss

class Args:
    batch_size = 200
    N_epochs = 50_000
    train_ratio = 0.8
    early_stop_start = 150_000
    horizon = 4  # time steps
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

args = Args()

# options = {
#     'fraction_for_validation': 1-args.train_ratio,
#     'batch_size_validate': args.batch_size,
#     'batch_size_train': args.batch_size,
#     'tfds_shuffle_data': True,
#     'tfds_seed': 42,
# }

mngr = CheckpointManager('/nfs/ghome/live/mgao/latent_diffusion' + model_type + '/diffusion', AsyncCheckpointer(PyTreeCheckpointHandler()), options)

if model_type == 'state_only':
    args.transition_dim = 9
    train_traj, val_traj = state_only_traj_data_loader("/nfs/ghome/live/mgao/latent_diffusion/data/obj", args.train_ratio, 111)
elif model_type == 'empowerment' or model_type == 'VAE':
    args.transition_dim = 10
    new_action = np.load('/nfs/ghome/live/mgao/latent_diffusion/' + model_type + '/new_1daction.npy')
    train_traj, val_traj = new_traj_data_loader("/nfs/nhome/live/jheald/mingyang_latent_diffusion/data/obj", new_action, args.train_ratio, 111)
elif model_type == 'synergy_manifold':
    args.transition_dim = 15
    new_action = np.load('/nfs/ghome/live/mgao/latent_diffusion/empowerment/new_6daction.npy')
    train_traj, val_traj = new_traj_data_loader("/nfs/ghome/live/mgao/latent_diffusion/data/obj", new_action, args.train_ratio, 111)
elif model_type == 'VAE_6d':
    args.transition_dim = 15
    new_action = np.load('/nfs/ghome/live/mgao/latent_diffusion/VAE/new_6daction.npy')
    train_traj, val_traj = new_traj_data_loader("/nfs/ghome/live/mgao/latent_diffusion/data/obj", new_action, args.train_ratio, 111)
elif model_type == 'baseline':
    args.transition_dim = 15
    with open("/nfs/ghome/live/mgao/latent_diffusion/data/obj", 'rb') as handle:
        data = pickle.load(handle)
    actions = data['actions'][:,:26,:]
    train_traj, val_traj = new_traj_data_loader("/nfs/ghome/live/mgao/latent_diffusion/data/obj", actions, args.train_ratio, 111)

# initialisation
rng, step_rng = random.split(rng)
x = jnp.zeros((args.batch_size, args.horizon, args.transition_dim)) # [batch x horizon x transition]
time = random.uniform(step_rng, (args.batch_size, ), minval=1e-5, maxval=1)
score_model = temporal_unet(args)
params_s = score_model.init(rng, x, time)
optimizer_s = optax.chain(optax.adamw(learning_rate = args.learning_rate,
                        b1 = args.adam_b1,
                        b2 = args.adam_b2,
                        eps = args.adam_eps,
                        weight_decay = args.weight_decay),
                        optax.clip_by_global_norm(args.max_grad_norm),
                        optax.ema(decay = args.ema_decay))
state_s = train_state.TrainState.create(apply_fn = score_model.apply, params = params_s, tx = optimizer_s)

# train the model
train_size = train_traj.shape[0] # number of different episodes
val_size = val_traj.shape[0] # number of different episodes
time_step = train_traj.shape[1]
train_Nbatch = train_size // args.batch_size
val_Nbatch = val_size // args.batch_size
losses = []
mean_losses = []
val_losses = []
mean_val_losses = []
# early_stop = EarlyStopping(min_delta=1e-3, patience=3)

for k in range(args.N_epochs):
    
    # train step
    rng, step_rng = random.split(rng)
    shuffle_indx = random.permutation(step_rng, train_size)
    shuffle_train = train_traj[shuffle_indx]
    for i in range(train_Nbatch):
        # random start point
        rng, step_rng = random.split(rng)
        init_t = random.randint(step_rng, (args.batch_size, ), 0, time_step-args.horizon)  
        batch = shuffle_train[i*args.batch_size:(i+1)*args.batch_size, :, :]
        batch = batch[jnp.arange(args.batch_size)[:,None], init_t[:,None] + jnp.arange(args.horizon)[None,:], :]
        rng, step_rng = random.split(rng)
        loss, state_s = update_score(step_rng, batch, state_s, score_model)
        losses.append(loss)
    mean_losses.append(jnp.mean(jnp.array(losses)))
    losses = []
    
    # validation step
    rng, step_rng = random.split(rng)
    val_shuffle_indx = random.permutation(step_rng, val_size)
    shuffle_val = val_traj[val_shuffle_indx]
    for j in range(val_Nbatch):
        # random start point
        rng, step_rng = random.split(rng)
        init_t = random.randint(step_rng, (args.batch_size, ), 0, time_step-args.horizon)  
        val_batch = shuffle_val[j*args.batch_size:(j+1)*args.batch_size, :, :]
        val_batch = val_batch[jnp.arange(args.batch_size)[:,None], init_t[:,None] + jnp.arange(args.horizon)[None,:], :]
        rng, step_rng = random.split(rng)
        val_loss = evaluate_score(step_rng, val_batch, state_s, score_model)
        val_losses.append(val_loss)
    mean_val_losses.append(jnp.mean(jnp.array(val_losses)))
    val_losses = []

    # mngr.save(k+1, state_s , metrics=np.array(mean_val_losses[-1]).tolist())
    
    if (k+1) % 1000 == 0:
        print("Score: epoch %d \t, Train Loss %f " % (k+1, mean_losses[-1]))
        print("Score: epoch %d \t, Validation Loss %f " % (k+1, mean_val_losses[-1]))
        mngr.save(k+1, state_s , metrics=np.array(mean_val_losses[-1]).tolist())
        # if (k+1) >= args.early_stop_start:
        #     early_stop = early_stop.update(val_loss)
        #     if early_stop.should_stop:
        #         # if early stopping criteria met, break
        #         print('Early stopping criteria met, breaking...')
        #         break
mngr.wait_until_finished()
state_s = mngr.restore(mngr.best_step())

with open('/nfs/ghome/live/mgao/latent_diffusion/' + model_type + '/diffusion/params_score_0912.pickle', 'wb') as handle:
    pickle.dump(state_s, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.figure()
plt.plot(mean_losses, label = 'train_loss')
plt.plot(mean_val_losses, alpha = 0.3, label = 'validation_loss')
plt.legend()
plt.savefig('/nfs/ghome/live/mgao/latent_diffusion/' + model_type + '/diffusion/scoreLoss0912.pdf')

np.save('/nfs/ghome/live/mgao/latent_diffusion/' + model_type + '/diffusion/mean_loss0912.npy', np.array(mean_losses))
np.save('/nfs/ghome/live/mgao/latent_diffusion/' + model_type + '/diffusion/mean_val_loss0912.npy', np.array(mean_val_losses))
