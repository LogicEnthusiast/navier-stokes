import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx, ny = 200, 100
dx, dy = 4.0/(nx-1), 2.0/(ny-1)
dt = 0.001
Re = 100  # Reynolds number

# Initialize grids
u = np.ones((ny, nx))  # x-velocity
v = np.zeros((ny, nx))  # y-velocity
p = np.zeros((ny, nx))  # pressure

# Cylinder parameters
cx, cy, r = 1.0, 1.0, 0.2
x = np.linspace(0, 4, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
cylinder = (X - cx)**2 + (Y - cy)**2 <= r**2

# Solve to steady state
for _ in range(3000):
    # Momentum equations with pressure gradient
    u_new = u.copy()
    v_new = v.copy()
    
    # Pressure correction (simplified)
    p[1:-1, 1:-1] = 0.25 * (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 
                            dx**2 * ((u[1:-1, 2:] - u[1:-1, :-2])/(2*dx) + 
                                    (v[2:, 1:-1] - v[:-2, 1:-1])/(2*dy)))
    
    # Momentum equations
    u_new[1:-1, 1:-1] = u[1:-1, 1:-1] - dt * (
        u[1:-1, 1:-1] * (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
        v[1:-1, 1:-1] * (u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy) +
        (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dx) -
        (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (Re*dx**2) -
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (Re*dy**2)
    )
    
    v_new[1:-1, 1:-1] = v[1:-1, 1:-1] - dt * (
        u[1:-1, 1:-1] * (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx) +
        v[1:-1, 1:-1] * (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy) +
        (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dy) -
        (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / (Re*dx**2) -
        (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / (Re*dy**2)
    )
    
    # Boundary conditions
    u_new[0, :] = u_new[-1, :] = 0.8
    u_new[:, 0] = u_new[:, -1] = 0.8
    v_new[0, :] = v_new[-1, :] = 0.0
    v_new[:, 0] = v_new[:, -1] = 0.0
    
    # Cylinder boundary (no-slip)
    u_new[cylinder] = 0
    v_new[cylinder] = 0
    
    u, v = u_new, v_new

# Visualization
velocity_mag = np.sqrt(u**2 + v**2)
plt.figure(figsize=(12, 6))
plt.streamplot(X, Y, u, v, density=2, color='blue', linewidth=0.8)
plt.contourf(X, Y, velocity_mag, levels=30, alpha=0.8, cmap='RdYlBu_r', 
             vmin=0.0, vmax=velocity_mag.max())
plt.colorbar(label='Velocity Magnitude')

# Draw cylinder
circle = plt.Circle((cx, cy), r, color='black', fill=True)
plt.gca().add_patch(circle)

plt.xlim(0, 4)
plt.ylim(0, 2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Navier-Stokes Flow Around Cylinder (Stabilized)')
plt.axis('equal')
plt.tight_layout()
plt.show()
