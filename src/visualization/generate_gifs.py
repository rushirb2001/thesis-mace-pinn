import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load Either of the two equations
type_m = 0
variation = 0

# Load the MATLAB file
if type_m == 0:
    mat_data = sio.loadmat(f'/Users/rushirbhavsar/Downloads/grey_scott_variation{variation}.mat')
else:
    mat_data = sio.loadmat(f'/Users/rushirbhavsar/Downloads/ginzburg_landau_square_{variation}.mat')

# Extract variables
x = mat_data['x'].squeeze()      # spatial grid in x
y = mat_data['y'].squeeze()      # spatial grid in y
t = mat_data['t'].squeeze()      # time vector
usol = mat_data['usol']          # u concentration [time, N, N]
vsol = mat_data['vsol']          # v concentration [time, N, N]

# Get grid dimensions and number of frames
N = len(x)
num_frames = usol.shape[0]

# Create meshgrid for plotting
X, Y = np.meshgrid(x, y)

# Setup the plot for the u concentration simulation
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.imshow(usol[0, :, :], extent=[x.min(), x.max(), y.min(), y.max()],
                origin='lower', cmap='turbo')
ax.set_title('Gray-Scott Reaction-Diffusion: u Concentration')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(cax, ax=ax)

# Text to display current simulation time
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', 
                    fontsize=10, verticalalignment='top')

def init():
    cax.set_data(usol[0, :, :])
    time_text.set_text(f"t = {t[0]:.4f}")
    return cax, time_text

def animate(i):
    cax.set_data(usol[i, :, :])
    time_text.set_text(f"t = {t[i]:.4f}")
    return cax, time_text

# Create animation: interval is in milliseconds, adjust as needed
ani = animation.FuncAnimation(fig, animate, frames=num_frames, init_func=init,
                            interval=10, blit=True)

# To save the animation as an mp4 file, uncomment the next line (requires ffmpeg or similar):
# ani.save('gray_scott_simulation.mp4', writer='ffmpeg', fps=10)

plt.show()
