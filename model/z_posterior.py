from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from jax import numpy as jnp
from jax import random, vmap, jacrev
rng = random.PRNGKey(2024)

from flax import linen as nn
import optax
from flax.training.train_state import TrainState
from flax.linen.initializers import zeros_init, ones_init, normal, orthogonal, constant

class dynamics(nn.Module):
    h_dims_dynamics: List
    control_variables: List

    def setup(self):

        dynamics = [nn.Sequential([nn.Dense(features=h_dim), nn.relu]) for h_dim in self.h_dims_dynamics]
        dynamics.append(nn.Dense(features=len(self.control_variables)*2))
        self.dynamics = dynamics

    def __call__(self, obs, action):

        def log_likelihood_diagonal_Gaussian(x, mu, log_var):
            """
            Calculate the log likelihood of x under a diagonal Gaussian distribution
            var_min is added to the variances for numerical stability
            """
            log_likelihood = -0.5 * (log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var))
            return log_likelihood

        def get_log_var(x):
            """
            sigma = log(1 + exp(x))
            """
            sigma = nn.softplus(x) + 1e-6
            log_var = 2 * jnp.log(sigma)
            return log_var

        x = jnp.concatenate((obs, action), axis=-1)
        for i, fn in enumerate(self.dynamics):
            x = fn(x)
        y_prime_mean, y_prime_scale = jnp.split(x, 2, axis=-1)
        y_prime_log_var = get_log_var(y_prime_scale)

        return y_prime_mean, y_prime_log_var

class precoder(nn.Module):
    mean_action: Any
    std_action: Any

    def setup(self):

        None

    def __call__(self, obs, dynamics_state):

        def dynamics_forward_pass(action, obs, dynamics_state):

            # squash the control inputs before passing them through the dynamics model
            action = (nn.sigmoid((action-0.5)*5.) - self.mean_action) / self.std_action

            delta_y_mean, delta_y_log_var = dynamics_state.apply_fn(dynamics_state.params, obs, action)

            return delta_y_mean

        get_jacobian = jacrev(dynamics_forward_pass)

        mu_u_source = jnp.zeros(self.mean_action.shape)

        Jac = get_jacobian(mu_u_source, obs, dynamics_state)

        # economic SVD of Jac (full_matrices=False) automatically excludes the right singular vectors that are associated with zero singular values
        U, S, Vh = jnp.linalg.svd(Jac, full_matrices=False)

        # resolve sign ambiguity of right singular vectors to avoid sign flips
        Jac_z = Jac @ Vh.T
        Vh = Vh * jnp.where(jnp.diag(Jac_z) < 0, -1, 1)[:,None]

        A = Vh.T

        return A

class infer(nn.Module):
    h_dims_posterior: List
    control_variables: List

    def setup(self):

        self.z_dim = len(self.control_variables)

        inverse = [nn.Sequential([nn.Dense(features=h_dim), nn.relu]) for h_dim in self.h_dims_posterior]
        inverse.append(nn.Dense(features=self.z_dim*2))
        self.inverse = inverse

    def __call__(self, obs, y_prime):

        def get_log_var(x):
            """
            sigma = log(1 + exp(x))
            """
            sigma = nn.softplus(x) + 1e-6
            log_var = 2 * jnp.log(sigma)
            return log_var

        x = jnp.concatenate((obs, y_prime), axis=-1) # y_prime not delta
        for fn in self.inverse:
            x = fn(x)
        z_mean, z_scale = jnp.split(x, 2, axis=-1)
        z_log_var = get_log_var(z_scale)

        return z_mean, z_log_var

class MLP_Gaussian(nn.Module):
    h_dims: List
    out_dim: int

    def setup(self):

        mlp = [nn.Sequential([nn.Dense(features=h_dim), nn.relu]) for h_dim in self.h_dims]
        mlp.append(nn.Dense(features=self.out_dim*2))
        self.mlp = mlp

    def __call__(self, x):

        def get_log_var(x):
            """
            sigma = log(1 + exp(x))
            """
            sigma = nn.softplus(x) + 1e-6
            log_var = 2 * jnp.log(sigma)
            return log_var

        for fn in self.mlp:
            x = fn(x)
        x_mean, x_scale = jnp.split(x, 2, axis=-1)
        x_log_var = get_log_var(x_scale)

        return x_mean, x_log_var

# class train_power(nn.Module):
#     control_variables: List
#     mean_action: Any
#     std_action: Any

#     def setup(self):

#         self.power_param = self.param('power_param', normal(), (1,))

#         self.z_dim = len(self.control_variables)

#         self.synergies = precoder(mean_action=self.mean_action,
#                                   std_action=self.std_action) 

#     def __call__(self, obs, y_prime, dynamics_state, key):

#         def sample_diag_Gaussian(mean, log_var, key):
#             """
#             sample from a diagonal Gaussian distribution
#             """
#             return mean + jnp.exp(0.5 * log_var) * random.normal(key, mean.shape)

#         def log_likelihood_diagonal_Gaussian(x, mu, log_var):
#             """
#             Calculate the log likelihood of x under a diagonal Gaussian distribution
#             var_min is added to the variances for numerical stability
#             """
#             log_likelihood = -0.5 * (log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var))
#             return log_likelihood

#         def dynamics_forward_pass(action, obs, dynamics_state):

#             # squash the control inputs before passing them through the dynamics model
#             action = (nn.sigmoid((action-0.5)*5.) - self.mean_action) / self.std_action

#             delta_y_mean, delta_y_log_var = dynamics_state.apply_fn(dynamics_state.params, obs, action)

#             return delta_y_mean, delta_y_log_var

#         def get_log_var(x):
#             """
#             sigma = log(1 + exp(x))
#             """
#             sigma = nn.softplus(x) + 1e-6
#             log_var = 2 * jnp.log(sigma)
#             return log_var

#         def y_prime_log_likelihood(obs, y_prime, dynamics_state, key):

#             # sample 1D high-level action
#             key, subkey = random.split(key,2)
#             z = sample_diag_Gaussian(jnp.zeros((self.z_dim,)), get_log_var(self.power_param), subkey) # equal power allocation assumed here

#             # get synergy vector
#             A = self.synergies(obs, dynamics_state)

#             # project 1D high-level action to 6D low-level action via synergy vector A
#             action = A @ z

#             # sample a change in the control variable y (elbow joint angle) from the dynamics model
#             delta_y_mean, delta_y_log_var = dynamics_forward_pass(action, obs, dynamics_state)

#             ll_y_prime = log_likelihood_diagonal_Gaussian(y_prime - obs[self.control_variables], delta_y_mean, delta_y_log_var)

#             return ll_y_prime, delta_y_mean, delta_y_log_var
    
#         batch_y_prime_log_likelihood = vmap(y_prime_log_likelihood, in_axes=(0,0,None,0))

#         keys = random.split(key, obs.shape[0])
#         ll_y_prime, delta_y_mean, delta_y_log_var = batch_y_prime_log_likelihood(obs, y_prime, dynamics_state, keys)
#         loss = -jnp.sum(ll_y_prime, axis=-1) / self.z_dim

#         return loss, delta_y_mean, delta_y_log_var

class train_posterior(nn.Module):
    control_variables: List
    h_dims_posterior: List
    mean_action: Any
    std_action: Any

    def setup(self):

        self.power_param = self.param('power_param', constant(jnp.log(jnp.exp(1.) - 1)), (1,)) # initialise with prior std of 1

        self.z_dim = len(self.control_variables)

        self.synergies = precoder(mean_action=self.mean_action,
                                  std_action=self.std_action) 

        self.posterior_model = infer(h_dims_posterior=self.h_dims_posterior,
                                     control_variables=self.control_variables)

    def __call__(self, obs, y_prime, dynamics_state, key):

        def get_log_var(x):
            """
            sigma = log(1 + exp(x))
            """
            sigma = nn.softplus(x) + 1e-6
            log_var = 2 * jnp.log(sigma)
            return log_var

        def KL_diagonal_Gaussians(mu_1, log_var_1, mu_0, log_var_0):
            """
            KL(q||p), where q is posterior and p is prior
            mu_1, log_var_1 is the mean and log variances of the posterior
            mu_0, log_var_0 is the mean and log variances of the prior
            """
            return jnp.mean(0.5 * (log_var_0 - log_var_1 + jnp.exp(log_var_1 - log_var_0) 
                                 - 1.0 + (mu_1 - mu_0)**2 / jnp.exp(log_var_0)))

        def sample_diag_Gaussian(mean, log_var, key):
            """
            sample from a diagonal Gaussian distribution
            """
            return mean + jnp.exp(0.5 * log_var) * random.normal(key, mean.shape)

        def log_likelihood_diagonal_Gaussian(x, mu, log_var):
            """
            Calculate the log likelihood of x under a diagonal Gaussian distribution
            var_min is added to the variances for numerical stability
            """
            log_likelihood = jnp.mean(-0.5 * (log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var)))
            return log_likelihood

        def dynamics_forward_pass(action, obs, dynamics_state):

            # squash the control inputs before passing them through the dynamics model
            action = (nn.sigmoid((action-0.5)*5.) - self.mean_action) / self.std_action

            delta_y_mean, delta_y_log_var = dynamics_state.apply_fn(dynamics_state.params, obs, action)

            return delta_y_mean, delta_y_log_var

        def loss(obs, y_prime, dynamics_state, key):

            # get the posterior distribution over the high-level action given the obs and next control variable y (elbow joint angle)
            z_mean, z_log_var = self.posterior_model(obs, y_prime)

            # sample 1D high-level action
            z_i = sample_diag_Gaussian(z_mean, z_log_var, key)

            # get synergy vector
            A = self.synergies(obs, dynamics_state)

            # project 1D high-level action to 6D low-level action via synergy vector A
            action = A @ z_i

            # sample a change in the control variable y (elbow joint angle) from the dynamics model
            delta_y_mean, delta_y_log_var = dynamics_forward_pass(action, obs, dynamics_state)

            ll_y_prime = log_likelihood_diagonal_Gaussian(y_prime - obs[self.control_variables], delta_y_mean, delta_y_log_var)

            kl = KL_diagonal_Gaussians(z_mean, z_log_var, jnp.zeros((self.z_dim,)), get_log_var(self.power_param)) # equal power allocation assumed here

            return ll_y_prime, kl
    
        batch_loss = vmap(loss, in_axes=(0,0,None,0))

        keys = random.split(key, obs.shape[0])
        ll_y_prime, kl = batch_loss(obs, y_prime, dynamics_state, keys)
        elbo = ll_y_prime - kl
        negative_elbo = -elbo

        return negative_elbo, kl

class train_VAE(nn.Module):
    control_variables: List
    h_dims_encoder: List
    h_dims_decoder: List
    action_dim: int

    def setup(self):

        self.z_dim = len(self.control_variables) 

        self.encoder = MLP_Gaussian(h_dims=self.h_dims_encoder,
                                    out_dim=self.z_dim)

        self.decoder = MLP_Gaussian(h_dims=self.h_dims_decoder,
                                    out_dim=self.action_dim)

    def __call__(self, actions, key):

        def KL_diagonal_Gaussians(mu_1, log_var_1, mu_0, log_var_0):
            """
            KL(q||p), where q is posterior and p is prior
            mu_1, log_var_1 is the mean and log variances of the posterior
            mu_0, log_var_0 is the mean and log variances of the prior
            """
            return jnp.mean(0.5 * (log_var_0 - log_var_1 + jnp.exp(log_var_1 - log_var_0) 
                                 - 1.0 + (mu_1 - mu_0)**2 / jnp.exp(log_var_0)))

        def sample_diag_Gaussian(mean, log_var, key):
            """
            sample from a diagonal Gaussian distribution
            """
            return mean + jnp.exp(0.5 * log_var) * random.normal(key, mean.shape)

        def log_likelihood_diagonal_Gaussian(x, mu, log_var):
            """
            Calculate the log likelihood of x under a diagonal Gaussian distribution
            var_min is added to the variances for numerical stability
            """
            log_likelihood = jnp.mean(-0.5 * (log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var)))
            return log_likelihood

        def loss(action, key):

            # get the posterior distribution over the high-level action given the low-level action
            z_mean, z_log_var = self.encoder(action)

            # sample 1D high-level action
            z_i = sample_diag_Gaussian(z_mean, z_log_var, key)

            # decode the low-level action from the high-level action
            a_mean, a_log_var = self.decoder(z_i)

            ll_a = log_likelihood_diagonal_Gaussian(action, a_mean, a_log_var)

            kl = KL_diagonal_Gaussians(z_mean, z_log_var, jnp.zeros((self.z_dim,)), jnp.zeros((self.z_dim,)))

            return ll_a, kl
    
        batch_loss = vmap(loss, in_axes=(0,0))

        keys = random.split(key, actions.shape[0])
        ll_a, kl = batch_loss(actions, keys)
        elbo = ll_a - kl
        negative_elbo = -elbo

        return negative_elbo, kl

    @nn.compact
    def get_posterior_model_and_synergies(self):
        return self.posterior_model, self.synergies

class reconstruction(nn.Module):
    control_variables: List
    h_dims_posterior: List
    mean_action: Any
    std_action: Any

    def setup(self):

        self.z_dim = len(self.control_variables)

        self.synergies = precoder(mean_action=self.mean_action,
                                  std_action=self.std_action) 

        self.posterior_model = infer(h_dims_posterior=self.h_dims_posterior,
                                     control_variables=self.control_variables)

    def __call__(self, obs, y_prime, dynamics_state):

        def dynamics_forward_pass(action, obs, dynamics_state):

            # squash the control inputs before passing them through the dynamics model
            action = (nn.sigmoid((action-0.5)*5.) - self.mean_action) / self.std_action

            delta_y_mean, delta_y_log_var = dynamics_state.apply_fn(dynamics_state.params, obs, action)

            return delta_y_mean, delta_y_log_var

        def reconstruct(obs, y_prime, dynamics_state):

            # infer high-level action z
            z_mean, _ = self.posterior_model(obs, y_prime)
            
            # get synergy vector
            A = self.synergies(obs, dynamics_state)

            # project 1D high-level action to 6D low-level action via synergy vector A
            action = A @ z_mean

            # sample a change in the control variable y (elbow joint angle) from the dynamics model
            delta_y_mean, delta_y_log_var = dynamics_forward_pass(action, obs, dynamics_state)
            y_prime_mean = obs[self.control_variables] + delta_y_mean # y_prime not delta_y

            reconstruction_error = jnp.linalg.norm(y_prime_mean-y_prime)

            return reconstruction_error
    
        batch_reconstruct = vmap(reconstruct, in_axes=(0,0,None))

        reconstruction_error = batch_reconstruct(obs, y_prime, dynamics_state)

        return reconstruction_error