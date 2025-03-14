import os
os.environ["JAX_PLATFORMS"] = 'cpu'
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import pickle
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import numpy as np
import scipy.io
plt.rcParams.update({'axes.titlesize': 18})
plt.rcParams.update({'axes.labelsize': 18})
plt.rcParams.update({'xtick.labelsize': 14})
plt.rcParams.update({'ytick.labelsize': 14})

from networks import MLP

# Constants from your original problem
L=10. ; xmin, xmax = -L,L
N_x=2048
dx=2*L/N_x
dt=(0.5*dx**2)#*(N_x/2048)**2
T=7.
N_t=256
tmin,tmax=0,T
g=-1
alpha=1
x = jnp.linspace(xmin, xmax, N_x)
t = jnp.linspace(tmin, tmax, N_t)
N_t=int(T/dt)

# Load the trained model
with open('data/8x48.pkl', 'rb') as f:
    data = pickle.load(f)
params = data['params']

# Setup model
in_dim, features = 3, 8*[48] + [2] 
model = MLP(features, activation='tanh', factorization=True)

def forward(params, X):  
    x,t=X
    #return model.apply(params, X)
    #return model.apply(params, jnp.array((x/L,jnp.tanh(x/L),t)))
    return model.apply(params, jnp.array((jnp.cos(jnp.pi*x/L ), jnp.sin(jnp.pi*x/L ),t)))


v_forward = vmap(forward, in_axes=[None, 0])



# Function to compute spatial derivatives
def compute_derivatives(params, x, t):
    # Create batch of points
    X = jnp.stack([x, jnp.full_like(x, t)], axis=1)
    
    # Compute psi and its derivatives
    def psi_fn(x_):
        return forward(params, (x_[0], x_[1]))
    
    # Get gradients with respect to x
    grad_psi = jax.vmap(lambda x_: jax.jacfwd(psi_fn)(x_))(X)
    
    # Extract components
    psi = v_forward(params, X)
    psi_x = grad_psi[:, :, 0]  # derivative with respect to x
    
    return psi, psi_x

# Function to compute conserved quantities
def compute_conserved_quantities(params, x, t):
    psi, psi_x = compute_derivatives(params, x, t)
    
    # Extract real and imaginary parts
    u, v = psi[:, 0], psi[:, 1]
    u_x, v_x = psi_x[:, 0], psi_x[:, 1]
    
    # Particle number N = ∫|ψ|²dx
    particle_density = u**2 + v**2
    N = sum(particle_density)* dx
    
    # Hamiltonian H = ∫[(α/2)|∇ψ|² - (g/2)|ψ|⁴]dx
    H_density = (alpha/2) * (u_x**2 + v_x**2) - 0.5*g*particle_density**2
    H = sum(H_density)* dx
    
    # Momentum P = Im[∫ψ*∇ψdx]
    P_density = u*v_x - v*u_x
    P = sum(P_density)* dx
    
    return N, H, P

# Compute conserved quantities for all times
Ns = []
Hs = []
Ps = []

for current_t in t:
    N, H, P = compute_conserved_quantities(params, x, current_t)
    Ns.append(float(N))
    Hs.append(float(H))
    Ps.append(float(P))

# Convert lists to arrays for plotting
Ns = jnp.array(Ns)
Hs = jnp.array(Hs)
Ps = jnp.array(Ps)

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

ax1.plot(t, Ns)
ax1.set_ylabel('Particle Number N')
ax1.grid(True)

ax2.plot(t, Hs)
ax2.set_ylabel('Hamiltonian H')
ax2.grid(True)

ax3.plot(t, Ps)
ax3.set_ylabel('Momentum P')
ax3.set_xlabel('Time t')
ax3.grid(True)

plt.tight_layout()
plt.savefig('conservation_laws.png', dpi=300, bbox_inches='tight')
plt.close()

# Print statistics
print(f"Particle number - Mean: {jnp.mean(Ns):.6f}, Std: {jnp.std(Ns):.6f}")
print(f"Hamiltonian - Mean: {jnp.mean(Hs):.6f}, Std: {jnp.std(Hs):.6f}")
print(f"Momentum - Mean: {jnp.mean(Ps):.6f}, Std: {jnp.std(Ps):.6f}")

# Also print relative variations
print("\nRelative variations:")
print(f"Particle number: {jnp.std(Ns)/jnp.mean(Ns):.6f}")
print(f"Hamiltonian: {jnp.std(Hs)/jnp.mean(Hs):.6f}")
print(f"Momentum: {jnp.std(Ps)/jnp.mean(Ps):.6f}")

def compute_l2_error(psi1_u, psi1_v, psi2_u, psi2_v):
    
    # Compute the L2 norm of the first solution
    norm1 = jnp.sqrt(jnp.sum(psi1_u**2 + psi1_v**2))
    
    # Compute the L2 norm of the difference
    norm_diff = jnp.sqrt(jnp.sum((psi1_u - psi2_u)**2 + (psi1_v - psi2_v)**2))
    
    # Return the relative error
    return norm_diff / norm1


def analytical_rogue_wave(x, t, shift=T):
    """
    Compute the Peregrine soliton solution for given space-time grid.
    
    Parameters:
    x: spatial grid (array)
    t: time grid (array or scalar)
    shift: time shift parameter (default: T)
    
    Returns:
    Complex-valued solution psi(x, t)
    """
    # Ensure `t` is an array, but allow scalars
    t = jnp.asarray(t)  # Convert scalars to arrays
    t_shifted = t - shift

    # If `t` is a 1D array, reshape it to broadcast with `x`
    if t.ndim == 1:
        t_shifted = t[:, None]  # Shape (N_t, 1), to broadcast over x

    denominator = 4 * (x**2 + t_shifted**2) + 1
    psi = (1 - 4 * (1 + 2j * t_shifted) / denominator) * jnp.exp(1j * t_shifted)
    return psi


def compute_analytical_solution(x, t, shift=T):
   
    psi = analytical_rogue_wave(x, t, shift)  # Compute over full grid
    u_analytical = psi.real
    v_analytical = psi.imag
    return u_analytical, v_analytical


# Function to implement a properly configured spectral solver
def spectral_solver(x, t, alpha=1.0, g=-1.0):
    """
    Spectral solver for the nonlinear Schrödinger equation with external potential.
    Uses split-step Fourier method with Peregrine soliton initial condition.
    
    Parameters:
    x: spatial grid
    t: time array
    alpha: coefficient of diffusion term (default: 1.0)
    g: coefficient of nonlinear term (default: -1.0)
    
    Returns:
    u_spectral, v_spectral: real and imaginary parts of solution at all time points
    """
    N_x = len(x)
    N_t = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    
    # Set up spectral grid
    k = 2.0 * np.pi * np.fft.fftfreq(N_x, dx)
    k2 = k**2
    
    # Initial condition (t=0, corresponding to t_shifted=-2)
    #psi0 = analytical_rogue_wave(x, 0.)
    x0=-4.
    chi=jnp.sqrt(1/2)
    nu=jnp.sqrt(1-chi**2)
    psi0=(nu*jnp.tanh(nu*(x-x0))+1j*chi)*(nu*jnp.tanh(nu*(x+x0))-1j*chi)
    psi = psi0.copy()
    
    # Solution storage
    u_spectral = np.zeros((N_t, N_x))
    v_spectral = np.zeros((N_t, N_x))
    
    # Store initial condition
    u_spectral[0] = psi.real
    v_spectral[0] = psi.imag
    
    # Potential function matching the analytical solution
    def V_potential(x, t):
        t_shifted = t - T
        return (4*(x**2 - t_shifted**2) - 1) / ((x**2 + t_shifted**2 + 0.25)**2) - 2
    
    # Linear operator
    def apply_D_operator(psi_hat, dt):
        return np.exp(-1j * (alpha/2) * k2 * dt) * psi_hat
    
    # Nonlinear operator
    def apply_N_operator(psi, x, t, dt):
        V = 0#V_potential(x, t)
        nonlinear_phase = (-V + g * np.abs(psi)**2) * dt
        return np.exp(1j * nonlinear_phase) * psi
    
    # Main time-stepping loop
    for n in range(1, N_t):
        current_t = t[n-1]
        
        # Half-step nonlinear operator
        psi = apply_N_operator(psi, x, current_t, dt/2)
        
        # Full-step linear operator
        psi_hat = fft(psi)
        psi_hat = apply_D_operator(psi_hat, dt)
        psi = ifft(psi_hat)
        
        # Half-step nonlinear operator
        psi = apply_N_operator(psi, x, current_t + dt/2, dt/2)
        
        # Store solution
        u_spectral[n] = psi.real
        v_spectral[n] = psi.imag
    
    return u_spectral, v_spectral

# Compute the PINN solution
psi_pinn = []
for current_t in t:
    psi, _ = compute_derivatives(params, x, current_t)
    psi_pinn.append(psi)
psi_pinn = jnp.array(psi_pinn)
u_pinn = psi_pinn[:,:,0]
v_pinn = psi_pinn[:,:,1]

# Compute analytical solution
#u_analytical, v_analytical = compute_analytical_solution(x, t, shift=T)

# Compute spectral solution
u_spectral, v_spectral = spectral_solver(x, t, alpha=alpha, g=g)



# Create comparison plots
compare_times = np.linspace(0, len(t) - 1, 5, dtype=int)
fig, axes = plt.subplots(len(compare_times), 1, figsize=(10, 15))
fig.suptitle('Comparison of PINN, Spectral, and Analytical Solutions')

for i, tidx in enumerate(compare_times):
    current_t = t[tidx]
    
    # Plot |ψ|² for all solutions
    psi_sq_pinn = u_pinn[tidx]**2 + v_pinn[tidx]**2
    psi_sq_spectral = u_spectral[tidx]**2 + v_spectral[tidx]**2
    #psi_sq_analytical = u_analytical[tidx]**2 + v_analytical[tidx]**2
    
    axes[i].plot(x, psi_sq_pinn, 'b-', label='PINN')
    axes[i].plot(x, psi_sq_spectral, 'r--', label='Spectral')
    #axes[i].plot(x, psi_sq_analytical, 'g-.', label='Analytical')
    axes[i].set_ylabel(f'|ψ|² (t = {current_t:.2f})')
    axes[i].grid(True)
    axes[i].legend()

axes[-1].set_xlabel('x')
plt.tight_layout()
plt.savefig('solution_comparison_2ds.png', dpi=300, bbox_inches='tight')
plt.close()

# Compute L2 errors
l2_errors_pinn_vs_spectral = np.zeros(len(t))
#l2_errors_pinn_vs_analytical = np.zeros(len(t))
#l2_errors_spectral_vs_analytical = np.zeros(len(t))

for tidx in range(len(t)):
    # Error between PINN and spectral
    l2_errors_pinn_vs_spectral[tidx] = compute_l2_error(
        u_pinn[tidx], v_pinn[tidx],
        u_spectral[tidx], v_spectral[tidx]
    )
    
    # Error between PINN and analytical
    """l2_errors_pinn_vs_analytical[tidx] = compute_l2_error(
        u_pinn[tidx], v_pinn[tidx],
        u_analytical[tidx], v_analytical[tidx]
    )
    
    # Error between spectral and analytical
    l2_errors_spectral_vs_analytical[tidx] = compute_l2_error(
        u_spectral[tidx], v_spectral[tidx],
        u_analytical[tidx], v_analytical[tidx]
    )"""

# Print mean errors
print("\nMean relative L2 errors:")
print(f"PINN vs. Spectral: {jnp.mean(l2_errors_pinn_vs_spectral):.6f}")
#print(f"PINN vs. Analytical: {jnp.mean(l2_errors_pinn_vs_analytical):.6f}")
#print(f"Spectral vs. Analytical: {jnp.mean(l2_errors_spectral_vs_analytical):.6f}")

# Create error comparison plot
plt.figure(figsize=(10, 6))
plt.plot(t, l2_errors_pinn_vs_spectral, 'b-', label="PINN vs. Spectral")
#plt.plot(t, l2_errors_pinn_vs_analytical, 'g-', label="PINN vs. Analytical")
#plt.plot(t, l2_errors_spectral_vs_analytical, 'r--', label="Spectral vs. Analytical")
plt.xlabel("Time t")
plt.ylabel("Relative L2 Error")
plt.title("L2 Error Over Time")
plt.grid(True)
plt.legend()
plt.yscale('log')  # Using log scale for better visualization
plt.savefig("l2_error_fft.png", dpi=300, bbox_inches="tight")
plt.close()
"""
params_matlab = {}
for i, (key, value) in enumerate(params.items()):
    try:
        params_matlab[f'param_{i}'] = jnp.array(value)
    except:
        print(f"Skipping non-numeric param: {key}")
# Update MATLAB output
scipy.io.savemat('pinn_results_2ds.mat', 
    {'params': params_matlab,
     'loss_history': data['metrics']['loss history'],
     'l2_errors': jnp.array(data['metrics']['l2_errors']),
     'l2_errors_pinn_vs_spectral': jnp.array(l2_errors_pinn_vs_spectral),
     'psi_squared_pred': u_pinn**2+v_pinn**2,
     'psi_squared_fft': jnp.array(u_spectral**2 + v_spectral**2),
     'x': x,
     't': t})

print("✅ Data successfully converted to pinn_results_2ds.mat with correctly aligned solutions")"""