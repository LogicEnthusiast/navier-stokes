import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

class NavierStokesSimulation:
    def __init__(self, nx=200, ny=100, Re=100):
        """
        Initialize the Navier-Stokes simulation for flow around a cylinder.
        
        Parameters:
        nx, ny: Grid dimensions
        Re: Reynolds number
        """
        self.nx = nx
        self.ny = ny
        self.Re = Re
        
        # Physical parameters
        self.Lx = 4.0  # Domain length
        self.Ly = 2.0  # Domain height
        self.dx = self.Lx / (nx - 1)
        self.dy = self.Ly / (ny - 1)
        self.dt = 0.001  # Time step
        
        # Cylinder parameters
        self.cx = 1.0  # Cylinder center x
        self.cy = 1.0  # Cylinder center y
        self.r = 0.15  # Cylinder radius
        
        # Initialize velocity and pressure fields
        self.u = np.ones((ny, nx))  # x-velocity
        self.v = np.zeros((ny, nx))  # y-velocity
        self.p = np.zeros((ny, nx))  # pressure
        
        # Create mask for cylinder
        self.create_cylinder_mask()
        
        # Set inlet velocity
        self.U_inf = 1.0
        self.u[:, 0] = self.U_inf
        
    def create_cylinder_mask(self):
        """Create a boolean mask for the cylinder obstacle."""
        self.mask = np.zeros((self.ny, self.nx), dtype=bool)
        for i in range(self.ny):
            for j in range(self.nx):
                x = j * self.dx
                y = i * self.dy
                if (x - self.cx)**2 + (y - self.cy)**2 <= self.r**2:
                    self.mask[i, j] = True
    
    def apply_boundary_conditions(self):
        """Apply boundary conditions to velocity fields."""
        # Inlet (left boundary)
        self.u[:, 0] = self.U_inf
        self.v[:, 0] = 0
        
        # Outlet (right boundary) - zero gradient
        self.u[:, -1] = self.u[:, -2]
        self.v[:, -1] = self.v[:, -2]
        
        # Top and bottom walls - free slip
        self.v[0, :] = 0
        self.v[-1, :] = 0
        self.u[0, :] = self.u[1, :]
        self.u[-1, :] = self.u[-2, :]
        
        # Cylinder surface - no slip
        self.u[self.mask] = 0
        self.v[self.mask] = 0
    
    def solve_momentum(self):
        """Solve momentum equations (predictor step)."""
        u_new = self.u.copy()
        v_new = self.v.copy()
        nu = self.U_inf * 2 * self.r / self.Re  # Kinematic viscosity
        
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if self.mask[i, j]:
                    continue
                
                # Convection terms
                u_conv = (self.u[i, j] * (self.u[i, j+1] - self.u[i, j-1]) / (2*self.dx) +
                         self.v[i, j] * (self.u[i+1, j] - self.u[i-1, j]) / (2*self.dy))
                
                v_conv = (self.u[i, j] * (self.v[i, j+1] - self.v[i, j-1]) / (2*self.dx) +
                         self.v[i, j] * (self.v[i+1, j] - self.v[i-1, j]) / (2*self.dy))
                
                # Diffusion terms
                u_diff = nu * ((self.u[i, j+1] - 2*self.u[i, j] + self.u[i, j-1]) / self.dx**2 +
                              (self.u[i+1, j] - 2*self.u[i, j] + self.u[i-1, j]) / self.dy**2)
                
                v_diff = nu * ((self.v[i, j+1] - 2*self.v[i, j] + self.v[i, j-1]) / self.dx**2 +
                              (self.v[i+1, j] - 2*self.v[i, j] + self.v[i-1, j]) / self.dy**2)
                
                # Pressure gradient
                dpx = (self.p[i, j+1] - self.p[i, j-1]) / (2*self.dx)
                dpy = (self.p[i+1, j] - self.p[i-1, j]) / (2*self.dy)
                
                # Update velocities
                u_new[i, j] = self.u[i, j] + self.dt * (-u_conv + u_diff - dpx)
                v_new[i, j] = self.v[i, j] + self.dt * (-v_conv + v_diff - dpy)
        
        return u_new, v_new
    
    def solve_pressure(self, u_star, v_star):
        """Solve pressure Poisson equation (corrector step)."""
        p_new = self.p.copy()
        
        for _ in range(50):  # Pressure iterations
            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    if self.mask[i, j]:
                        continue
                    
                    # Divergence of predicted velocity
                    div = ((u_star[i, j+1] - u_star[i, j-1]) / (2*self.dx) +
                           (v_star[i+1, j] - v_star[i-1, j]) / (2*self.dy))
                    
                    # Pressure Poisson equation
                    p_new[i, j] = 0.25 * (p_new[i, j+1] + p_new[i, j-1] +
                                         p_new[i+1, j] + p_new[i-1, j] -
                                         self.dx * self.dy * div / self.dt)
            
            # Pressure boundary conditions
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = 0  # Reference pressure at outlet
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
        
        return p_new
    
    def step(self):
        """Perform one time step of the simulation."""
        # Predictor: solve momentum equations
        u_star, v_star = self.solve_momentum()
        
        # Corrector: solve pressure and correct velocities
        self.p = self.solve_pressure(u_star, v_star)
        
        # Velocity correction
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if not self.mask[i, j]:
                    dpx = (self.p[i, j+1] - self.p[i, j-1]) / (2*self.dx)
                    dpy = (self.p[i+1, j] - self.p[i-1, j]) / (2*self.dy)
                    self.u[i, j] = u_star[i, j] - self.dt * dpx
                    self.v[i, j] = v_star[i, j] - self.dt * dpy
        
        # Apply boundary conditions
        self.apply_boundary_conditions()

def visualize_simulation(sim, num_frames=300):
    """Create animated visualization of the flow field."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Create meshgrid
    x = np.linspace(0, sim.Lx, sim.nx)
    y = np.linspace(0, sim.Ly, sim.ny)
    X, Y = np.meshgrid(x, y)
    
    # Calculate velocity magnitude
    speed = np.sqrt(sim.u**2 + sim.v**2)
    
    # Left plot: velocity magnitude
    im1 = ax1.contourf(X, Y, speed, levels=20, cmap='jet')
    circle1 = Circle((sim.cx, sim.cy), sim.r, color='white', zorder=10)
    ax1.add_patch(circle1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Velocity Magnitude')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Right plot: streamlines and vorticity
    vorticity = np.zeros_like(sim.u)
    for i in range(1, sim.ny - 1):
        for j in range(1, sim.nx - 1):
            vorticity[i, j] = ((sim.v[i, j+1] - sim.v[i, j-1]) / (2*sim.dx) -
                              (sim.u[i+1, j] - sim.u[i-1, j]) / (2*sim.dy))
    
    im2 = ax2.contourf(X, Y, vorticity, levels=20, cmap='RdBu', vmin=-10, vmax=10)
    strm = ax2.streamplot(X, Y, sim.u, sim.v, color='black', linewidth=0.5, density=1.5)
    circle2 = Circle((sim.cx, sim.cy), sim.r, color='white', zorder=10)
    ax2.add_patch(circle2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Vorticity and Streamlines')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Vorticity')
    
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def update(frame):
        # Run simulation steps
        for _ in range(5):
            sim.step()
        
        # Update velocity magnitude plot
        speed = np.sqrt(sim.u**2 + sim.v**2)
        ax1.clear()
        ax1.contourf(X, Y, speed, levels=20, cmap='jet')
        circle1 = Circle((sim.cx, sim.cy), sim.r, color='white', zorder=10)
        ax1.add_patch(circle1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Velocity Magnitude')
        ax1.set_aspect('equal')
        
        # Update vorticity plot
        vorticity = np.zeros_like(sim.u)
        for i in range(1, sim.ny - 1):
            for j in range(1, sim.nx - 1):
                vorticity[i, j] = ((sim.v[i, j+1] - sim.v[i, j-1]) / (2*sim.dx) -
                                  (sim.u[i+1, j] - sim.u[i-1, j]) / (2*sim.dy))
        
        ax2.clear()
        ax2.contourf(X, Y, vorticity, levels=20, cmap='RdBu', vmin=-10, vmax=10)
        ax2.streamplot(X, Y, sim.u, sim.v, color='black', linewidth=0.5, density=1.5)
        circle2 = Circle((sim.cx, sim.cy), sim.r, color='white', zorder=10)
        ax2.add_patch(circle2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Vorticity and Streamlines')
        ax2.set_aspect('equal')
        
        time_text = ax1.text(0.02, 0.95, f'Time: {frame*5*sim.dt:.2f}s', 
                            transform=ax1.transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return ax1, ax2
    
    anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                  interval=50, blit=False)
    plt.tight_layout()
    plt.show()

# Run simulation
if __name__ == "__main__":
    print("Initializing Navier-Stokes simulation...")
    sim = NavierStokesSimulation(nx=200, ny=100, Re=100)
    
    print("Running simulation and generating visualization...")
    print("Left panel: Velocity magnitude")
    print("Right panel: Vorticity (color) and streamlines")
    
    visualize_simulation(sim, num_frames=300)