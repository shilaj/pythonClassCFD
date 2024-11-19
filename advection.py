import numpy as np
import matplotlib.pyplot as plt
import imageio

# CONSTANTS: Initial setup parameters:
L = 1 # Length of the domain
n_x = 101 # Number of grid points
c = 0.01 # Wave speed
t_initial = 0 # Initial time
t_final = 1 # Final time
dt = 0.1 # Time step size

x, dx = np.linspace(0, L, n_x, retstep=True) # Grid points and grid spacing

def initial_condition(x):
    u_x_0 = np.zeros_like(x) # Setting all the values in the velocity field to zero

    # Looping over all the grid points and setting the velocity field to 1 between 0.1 and 0.3
    for i in range(n_x):
        if 0.1 <= x[i] <= 0.3:
            u_x_0[i] = 1
    return u_x_0

def analytical_solution(x, t):
    u_x_t = np.zeros_like(x) # Setting all the values in the velocity field to zero

    # Looping over all the grid points and setting the velocity field to 1 between 0.1 and 0.3
    for i in range(n_x):
        if 0.1 <= x[i] - c*t <= 0.3:
            u_x_t[i] = 1
    return u_x_t

def numerical_solution(u_old):
    u_new = np.zeros_like(u_old)
    for i in range(1, n_x-1):
        u_new[i] = u_old[i] - c * dt / dx * (u_old[i] - u_old[i-1]) # Using the backward difference scheme
    return u_new

def animate_solution(func, x, type='analytical'):
    images_list = []
    time_list = np.arange(t_initial, t_final+dt, dt).round(1)
    for t in time_list:
        images_list.append(plot_solution(func(x, t), x, t))
    return imageio.mimsave(f'./advection/advection_{type}.gif',images_list , fps=1)

def plot_solution(variable, x, t, save=False, type='analytical'):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(x, variable)
    ax.grid()
    ax.set(xlabel='x', ylabel=f'u(x, {t})', title='Advection equation solution at time t = {}'.format(t))

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image if not save else fig.savefig(f'./advection/advection_{type}_{t}.png')

if __name__ == '__main__':
    u_x_0 = initial_condition(x)
    for t in np.arange(t_initial, t_final, dt).round(1):
        u_x_t = analytical_solution(x, t)
        plot_solution(u_x_t, x, t, save=True)