import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
mplstyle.use('fast')

def create_all_comparison_gifs(h5_file="comparison_dataset.h5"):
    # Create GIFs for Gray-Scott variations
    for variation in range(5):
        create_comparison_gif(
            h5_file=h5_file,
            variation=variation,
            model_type="gray_scott",
            output_name=f"gray_scott_variation_{variation}.gif"
        )
    
    # Create GIFs for Ginzburg-Landau variations
    for variation in range(5):
        create_comparison_gif(
            h5_file=h5_file,
            variation=variation,
            model_type="ginzburg_landau",
            output_name=f"ginzburg_landau_variation_{variation}.gif"
        )

def create_comparison_gif(h5_file, variation, model_type, output_name):
    # Set up LaTeX-compatible formatting
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10
    })

    model_names = {
        "gray_scott": "Gray-Scott",
        "ginzburg_landau": "Ginzburg-Landau"
    }

    with h5py.File(h5_file, "r") as f:
        group = f[f"{model_type}/variation_{variation}"]
        u_truth = group["u_truth"][:]
        u_mlp = group["u_mlp"][:]
        u_custom = group["u_custom"][:]
        
        vmin = min(u_truth.min(), u_mlp.min(), u_custom.min())
        vmax = max(u_truth.max(), u_mlp.max(), u_custom.max())

    # Create figure WITHOUT constrained layout (which conflicts with subplots_adjust)
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))
    
    # Add a small title
    fig.suptitle(
        fr"{model_names[model_type]} (Var. {variation})", 
        fontsize=8
    )

    # Create the visualizations with JET colormap
    im1 = axes[0].imshow(u_truth[0], animated=True, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    im2 = axes[1].imshow(u_mlp[0], animated=True, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)
    im3 = axes[2].imshow(u_custom[0], animated=True, cmap='jet', origin='lower', vmin=vmin, vmax=vmax)

    # Minimal labels, no ticks
    for ax, title in zip(axes, [r"Ground Truth", r"MLP", r"MACE-PINN"]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_frame_on(False)

    # Adjust the layout AFTER creating the plots
    plt.tight_layout()
    
    # After tight_layout, manually adjust to make more space for colorbar
    plt.subplots_adjust(bottom=0.20, wspace=0.05)
    
    # Add a compact colorbar
    cax = fig.add_axes([0.3, 0.07, 0.4, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8, length=3, pad=2)
    
    # Small time indicator in bottom right
    time_text = fig.text(0.98, 0.05, r"$t=0$", ha="right", va="center", fontsize=8)

    def update(frame):
        im1.set_array(u_truth[frame])
        im2.set_array(u_mlp[frame])
        im3.set_array(u_custom[frame])
        time_text.set_text(f"$t={frame}$")
        return im1, im2, im3, time_text

    frames = tqdm(range(len(u_truth)), desc=f"Creating {output_name}", ncols=75)
    ani = FuncAnimation(fig, update, frames=frames, blit=True)

    # Save with extremely high DPI for maximum quality
    ani.save(output_name, writer="pillow", fps=30, dpi=800, progress_callback=lambda i, n: None)
    plt.close()

if __name__ == "__main__":
    create_all_comparison_gifs()