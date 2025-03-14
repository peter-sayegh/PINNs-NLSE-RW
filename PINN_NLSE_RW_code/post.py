import os
os.environ["JAX_PLATFORMS"] = 'cpu'
import jax, jax.numpy as jnp
import pickle
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.titlesize': 18, 'axes.labelsize': 18, 
                    'xtick.labelsize': 14, 'ytick.labelsize': 14})

import scipy.io

from networks import MLP


# Load results
with open('data/8x48.pkl', 'rb') as f:
    data = pickle.load(f)
params = data['params']
loss_history = data['metrics']['loss history']

# Setup
in_dim, features = 2,5*[48] + [2] 
model = MLP(features, activation='tanh', factorization=True)
def forward(params, X):  
    #x,t=X
    return model.apply(params, X)
    #return model.apply(params, jnp.array((x/L,jnp.tanh(x/L),t)))
    #return model.apply(params, jnp.array((jnp.cos(jnp.pi*x/L), jnp.sin(jnp.pi*x/L ),t)))

v_forward = jax.vmap(forward, in_axes=[None,0])

# Domain setup
L, N_x =2.5*jnp.pi, 512
T, N_t = 1.25, 256
x = jnp.linspace(-L, L, N_x)
t = jnp.linspace(0, 2*T, N_t)
xx, tt = jnp.meshgrid(x, t)
X_batch = jnp.hstack((xx.flatten()[:, None], tt.flatten()[:, None]))  # Should be (N_t * N_x, 2)

# Get PINN solution
fwd = v_forward(params, X_batch)
u, v = fwd[:, 0].reshape(N_t, N_x), fwd[:, 1].reshape(N_t, N_x)
psi_squared_pred = u**2 + v**2

# Analytical solution
chi = jnp.sqrt(1/2)
nu = jnp.sqrt(1-chi**2)

@jax.vmap
def analytical_solution(x, t):

    #psi = (nu*jnp.tanh(nu*(x-x0-chi*t)) + 1j*chi)*jnp.exp(-1j*t)
    t_shifted = t - T
    
    # Rogue wave solution formula from equation (9) in the paper
    denominator = 4 * (x**2 + t_shifted**2) + 1
    psi = (1 - 4 * (1 + 2j*t_shifted) / denominator) * jnp.exp(1j * t_shifted)
    return jnp.abs(psi)**2

psi_squared_true = analytical_solution(xx,tt).reshape(N_t, N_x)

# Compute errors
l2_error = jnp.sqrt(jnp.mean((psi_squared_pred - psi_squared_true)**2))/jnp.sqrt(jnp.mean(psi_squared_true**2))
print(f"Relative L2 error of |ψ|²: {l2_error:.2e}")

temporal_error = jnp.sqrt(jnp.mean((psi_squared_pred - psi_squared_true)**2, axis=1))/jnp.sqrt(jnp.mean(psi_squared_true**2, axis=1))

# Plotting
def plot_solution():
    time_indices = jnp.array([0, N_t//4, N_t//2, 3*N_t//4, N_t-1])
    plt.figure(figsize=(10, 8))
    for i, tidx in enumerate(time_indices):
        plt.plot(x, psi_squared_pred[tidx], label=f'PINN (t = {t[tidx]:.2f})', linestyle='dashed')
        plt.plot(x, psi_squared_true[tidx], label=f'Analytical (t = {t[tidx]:.2f})', linestyle='solid')
    plt.xlabel('x')
    plt.ylabel('|ψ|²')
    plt.title('Comparison of PINN and Analytical Solutions')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_nls.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics():
   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
   
   # Loss vs iterations
   iterations = range(len(data['metrics']['loss history']))
   ax1.plot(iterations, data['metrics']['loss history'])
   ax1.set_yscale('log')
   ax1.set_xlabel('Iterations')
   ax1.set_ylabel('Training Loss')
   ax1.grid(True)
   
   # Loss vs time
   ax2.plot(t, data['metrics']['loss history'][::len(data['metrics']['loss history'])//N_t][:N_t])
   ax2.set_yscale('log')
   ax2.set_xlabel('Time')
   ax2.set_ylabel('Training Loss')
   ax2.grid(True)
   
   # L2 error vs iterations 
   l2_iterations = range(0, len(iterations), 100)
   ax3.plot(l2_iterations, data['metrics']['l2_errors'])
   ax3.set_yscale('log')
   ax3.set_xlabel('Iterations')
   ax3.set_ylabel('L2 Error')
   ax3.grid(True)
   
   # L2 error vs time
   ax4.plot(t, temporal_error)
   ax4.set_yscale('log')
   ax4.set_xlabel('Time')
   ax4.set_ylabel('L2 Error')
   ax4.grid(True)
   
   plt.tight_layout()
   plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
   plt.close()


l2_errors = data['metrics']['l2_errors'] 
plot_solution()
plot_metrics()

params_matlab = {}
for i, (key, value) in enumerate(params.items()):
    try:
        params_matlab[f'param_{i}'] = jnp.array(value)  # Ensure numeric format
    except:
        print(f"Skipping non-numeric param: {key}")
# Save to a .mat file for MATLAB
scipy.io.savemat('pinn_results_rw_foc.mat', {
    'params': params_matlab,  # Now correctly formatted
    'loss_history': jnp.array(loss_history),
    'psi_squared_pred': jnp.array(psi_squared_pred),
    'l2_errors': jnp.array(l2_errors)
})