import jax.numpy as jnp
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], device_type = 'GPU')

# obs_keys: ['qpos', 'qvel', 'pose_err', 'act']

# load trajectory data (unet)
def new_traj_data_loader(filename, actions, train_ratio, seed):
    with open(filename, 'rb') as handle: 
        data = pickle.load(handle)
        obs = data['obs'][:,:26,:]

        # normalisation
        norm_obs = (obs - obs.mean(axis=(0,1),keepdims=True)) / obs.std(axis=(0,1),keepdims=True)
        norm_actions = (actions - actions.mean(axis=(0,1),keepdims=True)) / actions.std(axis=(0,1),keepdims=True)

        # concatenate obs and actions
        traj = jnp.concatenate((norm_obs[:,:25,:], norm_actions[:,:25,:]),axis = -1)
        train_traj, val_traj = train_test_split(traj, train_size=train_ratio, random_state=seed)

    return train_traj, val_traj

def state_only_traj_data_loader(filename, train_ratio, seed):
    with open(filename, 'rb') as handle: 
        data = pickle.load(handle)
        obs = data['obs'][:,:26,:]

        # normalisation
        norm_obs = (obs - obs.mean(axis=(0,1),keepdims=True)) / obs.std(axis=(0,1),keepdims=True)
        
        train_traj, val_traj = train_test_split(norm_obs[:,:25,:], train_size=train_ratio, random_state=seed)

    return train_traj, val_traj

def traj_data_loader(filename, train_ratio, seed):
    with open(filename, 'rb') as handle: 
        data = pickle.load(handle)
        obs = data['obs'][:,:25,:] 
        actions = data['actions'][:,:25,:]
        rwd = data['rwd_dense'][:,:25]

        # normalisation
        norm_obs = (obs - obs.mean(axis=(0,1),keepdims=True)) / obs.std(axis=(0,1),keepdims=True)
        norm_goal = (norm_obs[:,:,0] + norm_obs[:,:,2])[:,0] 
        norm_obs = jnp.delete(obs, 2, axis=2) # remove pose_err
        norm_actions = (actions - actions.mean(axis=(0,1),keepdims=True)) / actions.std(axis=(0,1),keepdims=True)
        norm_rwd = (rwd - rwd.mean(axis=(0,1), keepdims=True)) / rwd.std(axis=(0,1), keepdims=True)

        # concatenate obs and actions
        traj = jnp.concatenate((norm_obs, norm_actions, norm_rwd[:,:,None]),axis = -1)
        train_traj, val_traj, train_goal, val_goal = train_test_split(traj, norm_goal, train_size=train_ratio, random_state=seed)

        obs_size = norm_obs.shape[-1]
        acts_size = norm_actions.shape[-1]

        train_rwd = train_traj[:,:,-1]
        val_rwd = val_traj[:,:,-1]
        train_traj = train_traj[:,:,:obs_size+acts_size]
        val_traj = val_traj[:,:,:obs_size+acts_size]
    
    return train_traj, val_traj, train_rwd, val_rwd, train_goal, val_goal

# data loader for dynamic model and posterior
def transform_train_dataset(dataset, options):

    if options['tfds_shuffle_data']:
        dataset = dataset.shuffle(tf.data.experimental.cardinality(dataset).numpy(), seed = options['tfds_seed'], reshuffle_each_iteration = True)
    dataset = dataset.batch(options['batch_size_train'], drop_remainder = True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
def transform_validate_dataset(dataset, options):

    dataset = dataset.batch(options['batch_size_validate'], drop_remainder = True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
def create_tf_dataset(x, options):
    full_train_set = tf.data.Dataset.from_tensor_slices((x))
    n_data = tf.data.experimental.cardinality(full_train_set).numpy()
    n_data_validate = tf.cast(n_data * (jnp.round(options['fraction_for_validation'],1)), tf.int64)
    n_data_train = tf.cast(n_data * jnp.round((1. - options['fraction_for_validation']),1), tf.int64)

    train_dataset = full_train_set.take(n_data_train)
    validate_dataset = full_train_set.skip(n_data_train).take(n_data_validate)
    train_dataset = transform_train_dataset(train_dataset, options)
    validate_dataset = transform_validate_dataset(validate_dataset, options)

    return tfds.as_numpy(train_dataset), tfds.as_numpy(validate_dataset)

# load transition data
def trans_data_loader(filename, options, control_indx):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        obs = data['obs'][:,:26,:]
        acts = data['actions'][:,:26,:]

        # normalisation
        norm_obs = (obs - obs.mean(axis=(0,1))) / obs.std(axis=(0,1))
        norm_acts = (acts - acts.mean(axis=(0,1))) / acts.std(axis=(0,1))

        trans_obs = norm_obs[:,:25,:].reshape((-1,9))
        trans_acts = norm_acts[:,:25,:].reshape((-1,6))
        trans_y_prime = norm_obs[:,1:,control_indx].reshape((-1,1))
        train_trans, val_trans = create_tf_dataset((trans_obs, trans_acts, trans_y_prime), options)

    return train_trans, val_trans

def full_trans_data_loader(filename, options, control_indx):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        obs = data['obs'][:,:26,:]
        acts = data['actions'][:,:26,:]

        # normalisation
        norm_obs = (obs - obs.mean(axis=(0,1))) / obs.std(axis=(0,1))
        norm_acts = (acts - acts.mean(axis=(0,1))) / acts.std(axis=(0,1))

        trans_obs = norm_obs[:,:25,:].reshape((-1,9))
        trans_acts = norm_acts[:,:25,:].reshape((-1,6))
        trans_obs_prime = norm_obs[:,1:,:].reshape((-1,9))
        train_trans, val_trans = create_tf_dataset((trans_obs, trans_acts, trans_obs_prime), options)

    return train_trans, val_trans

# load observation
def obs_data_loader(filename, options, control_indx):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        obs = data['obs'][:,:26,:]
        
        # normalisation
        norm_obs = (obs - obs.mean(axis=(0,1),keepdims=True)) / obs.std(axis=(0,1),keepdims=True)

        trans_obs = norm_obs[:,:25,:].reshape((-1,9))
        trans_y_prime = norm_obs[:,1:,control_indx].reshape((-1,1))

        train_obs, val_obs = create_tf_dataset((trans_obs, trans_y_prime), options)

    return train_obs, val_obs

# get mean and sd
def get_mean_std(filename,var_name):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
        var = data[var_name][:,:26,:]
        length = var.shape[-1]
        mean = var.mean(axis=(0,1)).reshape((-1,length))
        std = var.std(axis=(0,1)).reshape((-1,length))

    return mean, std



