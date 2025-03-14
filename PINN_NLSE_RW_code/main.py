#import os
#os.environ["JAX_PLATFORMS"] = 'cpu'

import jax, jax.numpy as jnp
from jax import jit, vmap, grad
from jax import value_and_grad
from jax import jacfwd, jvp
from jax import jacrev, vjp
from jax import random
import matplotlib.pyplot as plt

import optax

import pickle

from networks import MLP
from batch_generator import sampler 
from batch_generator import update_attention
#from validation_pinn_nls import spectral_solver

# Constants from your original problem
L=10. ; xmin, xmax = -L,L
N_x=600
dx=2*L/N_x
dt=(0.5*dx**2)
T=7.
N_t=250#int(T/dt)
tmin,tmax=0,T
g=-1
alpha=1
x = jnp.linspace(xmin, xmax, N_x)
t = jnp.linspace(tmin, tmax, N_t)

domain_x = (xmin, xmax, N_x) # xmin, xmax, Nx
domain_t = (tmin, tmax, N_t) # tmin, tmax, Nt

# batch sizes

batch_size_eq = 1024
batch_size_bc = 200
batch_size_ic = 300

# residual based attention

init_rba = 0.1
gamma = 0.8
eta_star = 1.0 - gamma


#-Model for ( u, v)-----------------------------------------

in_dim, features = 3,8*[48] + [2] 
model = MLP(features, activation='tanh', factorization=True)

# for readability
def forward(params, X):  
    x,t=X
    #return model.apply(params, X)
    #return model.apply(params, jnp.array((x/L,jnp.tanh(x/L),t)))
    return model.apply(params, jnp.array((jnp.cos(jnp.pi*x/L), jnp.sin(jnp.pi*x/L),t))) #add +jnp.pi as a phase for the 2 dark soliton case


#-define eq residuum and vectorize-----------------------------------
def hvp(f, primals, tangents): #hessian vector product 
  return jvp(grad(f), primals, tangents)[1]

def V_potential(X):
    x, t = X
    t_s=t-T/2
    # Define the potential directly for t in [0,T]
    return (4*(x**2 - (t_s )**2) - 1)/((x**2 + (t_s )**2 + 0.25)**2) - 2
    

def compute_residuum(params, X):

    (u,v),(u_t,v_t)=jvp(lambda X: forward(params,X),(X,),(jnp.array([0.,1.]),))
    
    u_fn = lambda X: forward(params, X)[0]
    v_fn = lambda X: forward(params, X)[1]
    
    u_xx, _ = hvp(u_fn, (X,), (jnp.array((1.,0.)),))
    v_xx, _ = hvp(v_fn, (X,), (jnp.array((1.,0.)),))

    psi_squared = u**2 + v**2
    V =0# -V_potential(X)
    residual_real = -v_t + (alpha/2)*u_xx + V*u + g*psi_squared*u
    residual_imag = u_t + (alpha/2)*v_xx + V*v + g*psi_squared*v
    return jnp.array((residual_real,residual_imag))



#-Vectorize what needed----------------------------------------------

v_forward = vmap(forward, in_axes=[None,0])
v_compute_residuum = vmap(compute_residuum, in_axes=[None,0])


#-loss function------------------------------------------------------
def compute_residuum_loss(params, X_batch):
    # Get residuals (X_batch is already sorted by time)
    residuum = v_compute_residuum(params, X_batch)
    residuum_sq = jnp.sum(residuum**2, axis=1)
    residuum_sq = residuum_sq.reshape(16, -1) #divisor of batch_size_eq
    
    # Compute partial losses
    #partial_losses = jnp.mean(residuum_sq.reshape(-1, 1), keepdims=True) 
    partial_losses = jnp.mean(residuum_sq, axis=1)
    
    # Compute causal weights
    causal_weights = jax.lax.stop_gradient(
        jnp.exp(-5. * (jnp.cumsum(partial_losses) - partial_losses))
    )
    
    residuum_loss = jnp.mean(causal_weights * partial_losses)
    
    return residuum_loss, residuum

def compute_ic_loss(params, X_0_batch, u_0_batch,v_0_batch):
    
    fwd = v_forward(params, X_0_batch)

    u_0_pred = fwd[:,0]
    v_0_pred = fwd[:,1]
     
    ic_loss = jnp.mean(
        (u_0_pred - u_0_batch)**2 +
        (v_0_pred - v_0_batch)**2 )              # float

    return ic_loss

def compute_bc_loss(params, X_b_batch):
    """Compute loss for periodic boundary conditions on the derivative
    d_x psi(L,t) = d_x psi(-L,t)=0
    """
    
    # Get derivatives at both boundaries
    jvp_fun = lambda X: jvp(lambda X: forward(params,X), (X,), (jnp.array((1.0,0.0)),))[1]
    ux_vx = vmap(jvp_fun)(X_b_batch)
    ux,vx = ux_vx[:,0],ux_vx[:,1]

    # Compute MSE between left and right derivatives
    bc_loss = jnp.mean(
        ux**2+vx**2) #neumann zero slope bc
    """fwd = v_forward(params, X_b_batch)
    u, v = fwd[:, 0], fwd[:, 1]
    bc_loss = (u[0] - u[-1])**2 + (v[0] - v[-1])**2+jnp.mean(ux**2+vx**2)""" #dirichlet-neumann bcs

    
    return bc_loss

def compute_conservation_loss(params, X_batch):
    fwd = v_forward(params, X_batch)

    u= fwd[:,0]
    v= fwd[:,1]
    return jnp.mean(jnp.abs(u[-1,:]**2-u[1,:]**2)+v[-1,:]**2-v[1,:])

chi = jnp.sqrt(1/2)  # velocity
nu = jnp.sqrt(1-chi**2)  # darkness parameter
x0 = -4.0  # initial position
psi0=(nu*jnp.tanh(nu*(x-x0))+1j*chi)*(nu*jnp.tanh(nu*(x+x0))-1j*chi)
# Compute analytical solution on the same grid
"""def analytical_rogue_wave(x, t, shift=T/2):
    t_shifted = t - shift  # Reintroduce time shift
    denominator = 4 * (x**2 + t_shifted**2) + 1
    psi = (1 - 4 * (1 + 2j * t_shifted) / denominator) * jnp.exp(1j * t_shifted)
    return psi.real, psi.imag

def analytical_dark_soliton(x,t):
    psi=(nu*jnp.tanh(nu*(x-x0-chi*t))+1j*chi)*jnp.exp(-1j*t)
    return psi.real, psi.imag

# Vectorize over spatial and temporal points
v_analytical = jax.vmap(jax.vmap(analytical_rogue_wave, in_axes=(0, None)), in_axes=(None, 0))
u_exact, v_exact = v_analytical(x, t)


def compute_l2_error(params, x, t): #with analytical solution
   xx, tt = jnp.meshgrid(x, t)
   fwd = v_forward(params, jnp.hstack((xx.flatten()[:,None], tt.flatten()[:,None])))
   u_pred, v_pred = fwd[:, 0].reshape(N_t, N_x), fwd[:, 1].reshape(N_t, N_x)
   psi_squared_pred = u_pred**2 + v_pred**2
   
   u_exact, v_exact = v_analytical(x, t) 
   psi_squared_true = u_exact**2 + v_exact**2
   
   return jnp.sqrt(jnp.mean((psi_squared_pred - psi_squared_true)**2))/jnp.sqrt(jnp.mean(psi_squared_true**2))"""

 



#-training step------------------------------------------------------

def fit(params, optimizer, batch_generator, rba_weights, epochs=15001):
    
    opt_state = optimizer.init(params)
    key = random.key(1234) # for SGD

    metrics = {'loss history': [], 'l2_errors': []}

    @jit
    def step(params, opt_state, rba_weights_batch, X_batch, X_0_batch, u_0_batch, v_0_batch, X_b_batch, epoch):
        (residuum_loss, residuum), residuum_grads = value_and_grad(
            compute_residuum_loss, has_aux=True)(params, X_batch)
    
        ic_loss, ic_grads = value_and_grad(
            compute_ic_loss)(params, X_0_batch, u_0_batch, v_0_batch)
    
        bc_loss, bc_grads = value_and_grad(
            compute_bc_loss)(params, X_b_batch)
        #cons_loss, cons_grads=value_and_grad(compute_conservation_loss)(params, X_batch, X_0_batch)

        """mR, m0, mb,mc = 1.0, 10.0, 10.0,0.0

        loss_weights=jnp.array((mR,m0,mb,mc))

        # Combine all grads
        grads = jax.tree.map(
            lambda rg, icg, bcg,cqg: mR*rg + m0*icg + mb*bcg +mc*cqg,
            residuum_grads, ic_grads, bc_grads, cons_grads)
        loss = mR*residuum_loss + m0*ic_loss + mb*bc_loss +mc*cons_loss"""
        
        mR, m0, mb= 1.0, 1.0, 0.0

        loss_weights=jnp.array((mR,m0,mb))

        # Combine all grads
        grads = jax.tree.map(
            lambda rg, icg, bcg: mR*rg + m0*icg + mb*bcg ,
            residuum_grads, ic_grads, bc_grads)

        
       
        
        # Total loss
        
        loss = mR*residuum_loss + m0*ic_loss + mb*bc_loss 
    
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    
        residuum_abs = jnp.sum(jnp.abs(residuum), axis=1)
        residuum_max = jnp.max(residuum_abs)
        rba_weights_batch = gamma*rba_weights_batch + (1-gamma)*(residuum_abs / residuum_max)
    
        return params, opt_state, loss, rba_weights_batch, loss_weights

    for epoch in range(epochs):

        key, new_key = random.split(key, num=2)

        idxs, rba_weights_batch, X_batch, X_0_batch, u_0_batch, v_0_batch, X_b_batch = batch_generator(new_key, rba_weights)

        params, opt_state, loss, rba_weights_batch, loss_weights = step(
            params, opt_state, rba_weights_batch, X_batch, X_0_batch, u_0_batch, v_0_batch, X_b_batch, epoch)
        
        rba_weights = update_attention(idxs, rba_weights_batch, rba_weights) # allocating???

        # update metrics
        metrics['loss history'].append(loss)

        if epoch % 5 == 0:
            print(f'step {epoch}, loss: {loss}, weights: {loss_weights}')
        """if epoch % 100 == 0:  # Compute every 100 epochs
              # Compute L2 error every 100 epochs
            # Force recompute on first iteration of each epoch
            l2_error = compute_l2_error(params, x, t)
            metrics['l2_errors'].append(l2_error)
            print(f"Epoch {epoch}, Relative L2 Error: {l2_error:.6f}")"""


    return params, metrics


#-Execute all--------------------------------------------------------

if __name__=='__main__':
    
    key = random.key(0)
    params = model.init(key, jnp.ones(shape=(in_dim,)))

    batch_generator = sampler(domain_x, domain_t,
                              batch_size_eq, batch_size_ic, batch_size_bc
                              )
    
    rba_weights = init_rba*jnp.ones(
        shape = (domain_x[-1]*domain_t[-1],))

    lr = optax.exponential_decay(init_value=1e-3, transition_steps=5000, decay_rate=0.9)
    """lr = optax.warmup_cosine_decay_schedule(
    init_value=1e-5,
    peak_value=1e-3,
    warmup_steps=1000,
    decay_steps=5000,
    end_value=1e-6)"""

    optimizer = optax.adam(learning_rate=lr)
    
    params, metrics = fit(params, optimizer, batch_generator, rba_weights, epochs=15001)
    
    # save params
    with open('data/8x48.pkl', 'wb') as f:
        pickle.dump(
            {'params': params,
             'metrics': metrics}, f)
        
