import numpy
import matplotlib.pyplot as plt
import imageio
from copy import deepcopy

# CONSTANTS: Initial setup parameters:
temperature_left: float = 300.0 # Temperature at the left boundary in Kelvin
temperature_right: float = 273.0 # Temperature at the right boundary in Kelvin
L: float = 1.0 # Length of the rod in meters
alpha: int = 1 # Thermal diffusivity
dt: float = 0.001 # Time step size
dx: float = 0.05  # Grid spacing
t_initial: float = 0.0 # Initial time
t_final: float = 0.05 # Final time

x = numpy.arange(0, L+dx, dx).round(2) # Grid points and grid spacing

def initial_condition(x:numpy.ndarray) -> numpy.ndarray:
    temperature_initial = numpy.full_like(x, 350.0)
    return temperature_initial

def numerical_solution(temperature_old:numpy.ndarray) -> numpy.ndarray:
    '''
    Compute the numerical solution using the central difference scheme.

    Parameters:
    temp_old: The temperature distribution at the previous time step

    Returns:
    temp_new: The temperature distribution at the current time step

    Equation:
    dT/dt = alpha * d^2T/dx^2 

    Discretization: (Central difference scheme)
    t_new = t_old + dt * alpha * (t_old[i+1] - 2*t_old[i] + t_old[i-1]) / dx^2
    '''
    temperature_new = deepcopy(temperature_old)
    for i in range(1, len(temperature_old) - 1):
        temperature_new[i] = temperature_old[i] + alpha * dt * (temperature_old[i+1] - 2*temperature_old[i] + temperature_old[i-1]) / dx**2

    # Setting Dirichlet boundary conditions:
    temperature_new[0] = temperature_left
    temperature_new[-1] = temperature_right

    return temperature_new

def animate_combined_solutions(x: numpy.ndarray) -> None:
    '''
    Create a single GIF combining initial condition,
    and numerical solution with different colors.

    Parameters:
    x: The grid points

    Returns:
    None
    '''
    # Initialize solutions
    temperature_initial = initial_condition(x)  # Initial condition
    temperature_numerical = deepcopy(temperature_initial)  # Temporary variable for numerical solution
    
    # Lists to store frames
    images_list = []
    time_list = numpy.arange(t_initial, t_final + dt, dt).round(3)
    
    for t in time_list:
        # Compute solutions
        temperature_numerical = numerical_solution(temperature_numerical)
        
        # Plot solutions for the current time step
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(x, temperature_initial, linewidth=3, label='Initial Condition', color='blue')  # Initial condition
        ax.plot(x, temperature_numerical, linewidth=3, label='Numerical Solution', color='red')  # Numerical solution
        ax.grid()
        ax.set(xlabel='x', ylabel=f'T(x, {t})', title=f'Diffusion Equation Solutions at Time t = {t}')
        ax.legend()

        # Convert plot to image
        fig.canvas.draw()
        image = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images_list.append(image)
        plt.close(fig)  # Close the figure to save memory
    
    # Save the GIF
    imageio.mimsave('diffusion_combined.gif', images_list, fps=5)

if __name__ == '__main__':
    animate_combined_solutions(x)