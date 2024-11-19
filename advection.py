import numpy as np
import matplotlib.pyplot as plt
import imageio
from copy import deepcopy

# CONSTANTS: Initial setup parameters:
L:int = 1 # Length of the domain
n_x:int = 101 # Number of grid points
c:float = 0.01 # Wave speed
t_initial:float = 0.0 # Initial time
t_final:float = 1.0 # Final time
dt:float = 0.1 # Time step size

x, dx = np.linspace(0, L, n_x, retstep=True) # Grid points and grid spacing

def initial_condition(x:np.ndarray) -> np.ndarray:
    '''
    This function sets the initial condition.
    The initial condition is a step function between 0.1 and 0.3.

    Parameters:
    x: The grid points

    Returns:
    u_x_0: The velocity field at time t = 0
    '''
    u_x_0 = np.zeros_like(x) # Setting all the values in the velocity field to zero

    # Looping over all the grid points and setting the velocity field to 1 between 0.1 and 0.3
    for i in range(n_x):
        if 0.1 <= x[i] <= 0.3:
            u_x_0[i] = 1
    return u_x_0

def analytical_solution(x:np.ndarray, t:float) -> np.ndarray:
    '''
    This function calculates the analytical solution of the advection equation at time t.
    
    Parameters:
    x: The grid points
    t: The time at which the solution is calculated
    
    Returns:
    u_x_t: The velocity field at time t

    Functionality:
    The analytical solution of the advection equation is given by:
    c = velocity of the wave
    c * t = distance travelled by the wave

    if distance traveled by the wave is in the domain [0.1, 0.3]
    at a spatial location x, then we can say that the wave is at that location. 
    Hence, set the velocity field to 1 at that location.
    '''
    u_x_t = np.zeros_like(x) # Setting all the values in the velocity field to zero

    # Looping over all the grid points and setting the velocity field to 1 between 0.1 and 0.3
    for i in range(n_x):
        if 0.1 <= x[i] - c*t <= 0.3:
            u_x_t[i] = 1
    return u_x_t

def numerical_solution(u_old:np.ndarray) -> np.ndarray:
    '''
    This function calculates the numerical solution of the advection equation using the backward difference scheme.

    Parameters:
    u_old: The velocity fields at the previous time step

    Returns:
    u_new: The velocity fields at the current time step

    Functionality:
    The backward difference scheme is given by:
    du/dt + c * du/dx = 0
    '''
    u_new = np.zeros_like(u_old)
    for i in range(1, n_x-1):
        u_new[i] = u_old[i] - c * dt / dx * (u_old[i] - u_old[i-1]) # Using the backward difference scheme
    return u_new

def animate_combined_solutions(x: np.ndarray) -> None:
    '''
    Create a single GIF combining initial condition, analytical solution,
    and numerical solution with different colors.

    Parameters:
    x: The grid points

    Returns:
    None
    '''
    # Initialize solutions
    u_initial = initial_condition(x)  # Initial condition
    u_numerical = deepcopy(u_initial)  # Temporary variable for numerical solution
    
    # Lists to store frames
    images_list = []
    time_list = np.arange(t_initial, t_final + dt, dt).round(1)
    
    for t in time_list:
        # Compute solutions
        u_analytical = analytical_solution(x, t)
        u_numerical = numerical_solution(u_numerical)
        
        # Plot solutions for the current time step
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(x, u_initial, linewidth=3, label='Initial Condition', color='blue')  # Initial condition
        ax.plot(x, u_analytical, linewidth=3, label='Analytical Solution', color='green')  # Analytical solution
        ax.plot(x, u_numerical, linewidth=3, label='Numerical Solution', color='red')  # Numerical solution
        ax.grid()
        ax.set(xlabel='x', ylabel=f'u(x, {t})', title=f'Advection Equation Solutions at Time t = {t}')
        ax.legend()

        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images_list.append(image)
        plt.close(fig)  # Close the figure to save memory
    
    # Save the GIF
    imageio.mimsave('advection_combined.gif', images_list, fps=1)

if __name__ == '__main__':
    animate_combined_solutions(x)