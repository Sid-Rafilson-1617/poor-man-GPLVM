'''
Helper functions for the M-step in the EM algorithm
'''

import jax
import jax.numpy as jnp
from jax import jit, vmap
import jax.scipy as jscipy

# new==
@jit
def get_tuning_linear(params,basis):
    '''
    params: n_feat (basis) x n_neuron
    basis: n_tuning_state x n_basis
    '''
    return basis.dot(params)

@jit
def get_tuning_softplus(params,basis):
    '''
    params: n_feat (basis) x n_neuron
    basis: n_tuning_state x n_basis
    '''
    return jax.nn.softplus(get_tuning_linear(params,basis))
#==

def get_statistics(log_posterior_probs,y,):
    '''
    get posterior weighted observation, and posterior weighted time, for each latent bin
    posterior_probs: n_time x n_latent 
    y: n_time x n_neuron
    return:
    y_weighted: n_latent x n_neuron (A matrix)
    t_weighted: n_latent  (B vector)
    '''
    posterior_probs = jnp.exp(log_posterior_probs)
    y_weighted = jnp.einsum('tl,tn->ln',posterior_probs,y)
    t_weighted = posterior_probs.sum(axis=0) # n_latent,
    return y_weighted, t_weighted

@jit
def gaussian_m_step_analytic(hyperparam,basis_mat,y_weighted,t_weighted):
    '''
    basis_mat: n_latent x n_basis
    y_weighted: n_latent x n_neuron
    t_weighted: n_latent
    
    '''
    n_latent,n_basis = basis_mat.shape
    n_neuron = y_weighted.shape[1]
    noise_var = hyperparam['noise_std']**2
    param_prior_std = hyperparam['param_prior_std']

    G = jnp.einsum('qd,q,qb->db',basis_mat,t_weighted,basis_mat)
    H = G / noise_var + jnp.eye(n_basis) / (param_prior_std**2)    # compute the covariance matrix
    RHS = basis_mat.T @ y_weighted / noise_var
    w = jnp.linalg.solve(H,RHS)
    return w

def get_statistics_gain(log_posterior_probs, y, gain):
    '''
    get posterior weighted observation, time, and gain for each latent bin
    posterior_probs: n_time x n_latent 
    y: n_time x n_neuron
    gain: n_time
    return:
    y_weighted: n_latent x n_neuron (A matrix)
    t_weighted: n_latent  (B vector)
    gain_weighted: n_latent (posterior-weighted average gain for each latent)
    '''
    posterior_probs = jnp.exp(log_posterior_probs)
    y_weighted = jnp.einsum('tl,tn->ln', posterior_probs, y)
    t_weighted = posterior_probs.sum(axis=0) # n_latent,
    gain_weighted = jnp.einsum('tl,t->l', posterior_probs, gain) / (t_weighted + 1e-20)
    return y_weighted, t_weighted, gain_weighted

@jit
def get_gain_mstep_single_time(y_t, log_posterior_t, tuning):
    '''
    M-step for gain at a single time point: total spike / total predicted rate
    y_t: n_neuron
    log_posterior_t: n_latent
    tuning: n_latent x n_neuron
    '''
    posterior_t = jnp.exp(log_posterior_t)
    expected_rate = jnp.sum(posterior_t[:, None] * tuning, axis=0)  # n_neuron
    total_spikes = jnp.sum(y_t)
    total_expected_rate = jnp.sum(expected_rate)
    gain_new = total_spikes / (total_expected_rate + 1e-20)
    return gain_new

@jit 
def get_gain_mstep(y, log_posterior, tuning):
    '''
    M-step for gain: total spike / total predicted rate for each time
    y: n_time x n_neuron
    log_posterior: n_time x n_latent
    tuning: n_latent x n_neuron
    '''
    gain_new = vmap(get_gain_mstep_single_time, in_axes=(0, 0, None))(
        y, log_posterior, tuning)
    return gain_new

def get_gain_mstep_chunk(y, log_posterior, tuning, n_time_per_chunk=10000):
    '''
    Chunked version of get_gain_mstep for large arrays
    '''
    n_time_tot = y.shape[0]
    n_chunks = int(jnp.ceil(n_time_tot / n_time_per_chunk))
    
    gain_new_l = []
    for n in range(n_chunks):
        sl = slice(n * n_time_per_chunk, (n+1) * n_time_per_chunk)
        y_chunk = y[sl]
        log_posterior_chunk = log_posterior[sl]
        
        gain_new_chunk = get_gain_mstep(y_chunk, log_posterior_chunk, tuning)
        gain_new_l.append(gain_new_chunk)
    
    gain_new = jnp.concatenate(gain_new_l, axis=0)
    return gain_new

def poisson_m_step_objective(param,hyperparam,basis_mat,y_weighted,t_weighted):
    '''
    param: n_basis x n_neuron
    basis_mat: n_latent x n_basis
    y_weighted: n_latent x n_neuron
    t_weighted: n_latent
    
    return:
    negative log joint
    '''
    param_prior_std = hyperparam['param_prior_std']
    pf_hat = get_tuning_softplus(param,basis_mat) # n_latent x n_neuron
    # log_likelihood = jax.scipy.stats.poisson.logpmf(y_weighted,yhat * t_weighted[:,None]).sum()

    norm_term = pf_hat * t_weighted[:,None] # n_latent x n_neuron
    fit_term = vmap(jscipy.special.xlogy,in_axes=(1,1),out_axes=1)(y_weighted,pf_hat+1e-20) # n_latent x n_neuron
    log_likelihood = jnp.sum(fit_term - norm_term) # crucial, this is different from poisson logpmf(s_b_one, pf_one*t_b)!!!!
    log_prior = jax.scipy.stats.norm.logpdf(param,0,param_prior_std).sum()
    return -log_likelihood - log_prior

def poisson_m_step_objective_gain(weight, hyperparam, basis_mat, y_weighted, t_weighted, gain_weighted):
    '''
    Modified objective function with gain
    weight: n_basis x n_neuron (only weight is optimized, not gain)
    basis_mat: n_latent x n_basis
    y_weighted: n_latent x n_neuron
    t_weighted: n_latent
    gain_weighted: n_latent
    
    return:
    negative log joint
    '''
    param_prior_std = hyperparam['param_prior_std']
    tuning_base = get_tuning_softplus(weight, basis_mat) # n_latent x n_neuron
    pf_hat = tuning_base * gain_weighted[:, None] # n_latent x n_neuron

    norm_term = pf_hat * t_weighted[:, None] # n_latent x n_neuron
    fit_term = vmap(jscipy.special.xlogy, in_axes=(1,1), out_axes=1)(y_weighted, pf_hat+1e-20) # n_latent x n_neuron
    log_likelihood = jnp.sum(fit_term - norm_term)
    log_prior = jax.scipy.stats.norm.logpdf(weight, 0, param_prior_std).sum()
    return -log_likelihood - log_prior

import optax
from jax import tree_util
from poor_man_gplvm.fit_tuning_helper import make_adam_runner, tree_l2_norm



