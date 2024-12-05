import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

# Parameters
size = 100 # Grid size
Du, Dv = 0.04, 0.012# Diffusion coefficients
F, k = 0.035, 0.055 # Feed and kill rates
time_steps =  10000  # Number of iterations
dt = 1.0 # Time step

# Initialize the grid
U = np.ones((size, size))
V = np.zeros((size, size))

# Random initial conditions in the center
r = 10 # Radius of the initial disturbance
center = size // 2
for i in range(size):
    for j in range(size):
        if (i - center)**2 + (j - center)**2 < r**2:
            U[i, j] = 0.50 + 0.02 * np.random.random()
            V[i, j] = 0.25 + 0.02 * np.random.random()

# Function to compute Laplacian (diffusion)
def compute_laplacian(Z):
    return laplace(Z, mode="wrap")

# Simulation
for t in range(time_steps):
    Ulap = compute_laplacian(U)
    Vlap = compute_laplacian(V)

    # Gray-Scott reaction-diffusion equations
    UV2 = U * V**2
    U += (Du * Ulap - UV2 + F * (1 - U)) * dt
    V += (Dv * Vlap + UV2 - (F + k) * V) * dt

    # Visualization at intervals
    if t % 1000 == 0:
        plt.figure(figsize=(6, 6))
        plt.imshow(U, cmap="viridis")
        plt.title(f"Gray-Scott Model at t={t}")
        plt.axis("off")
        plt.show()
