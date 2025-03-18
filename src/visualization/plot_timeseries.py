import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rcParams

# LaTeX configuration ---------------------------------------------------------
# First install required packages in terminal:
# tlmgr install type1cm type1ec cm-super adjustbox collectbox

rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': True,
    'text.latex.preamble': r'''
        \usepackage{lmodern}
        \usepackage{amsmath}
        \usepackage{bm}
    ''',
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# Data loading and processing --------------------------------------------------
data_path = '/Users/rushirbhavsar/Downloads/'
nme = ['grey_scott', 'ginzburg_landau']
types = ['_variation', '_square_']
print(nme[0], nme[1])
for n in range(2):
    for v in range(5):
        name = nme[n]
        ts = types[n]
        filename = f'{data_path}{name}{ts}{v}.mat'

        print(f'Loading {filename}...')
        mat_data = sio.loadmat(file_name=filename)

        x = mat_data['x'].squeeze()
        y = mat_data['y'].squeeze()
        t = mat_data['t'].squeeze()
        usol = mat_data['usol']

        # Plot configuration -----------------------------------------------------------
        idx = [15, 40, 65, 100]
        fig, axs = plt.subplots(1, 4, figsize=(14, 4.5), 
                            gridspec_kw={'wspace': 0.1, 'left': 0.04, 'right': 0.92})

        # Consistent color scaling
        vmin, vmax = usol.min(), usol.max()
        # norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Plotting ---------------------------------------------------------------------
        for i, (ax, time_idx) in enumerate(zip(axs, idx)):
            im = ax.imshow(usol[time_idx], extent=[x.min(), x.max(), y.min(), y.max()],
                        origin='lower', cmap='jet')
            
            # Time annotation
            ax.text(0.05, 0.95, rf'$t = {t[time_idx]:.3f}$', transform=ax.transAxes,
                    color='white', fontsize=18, verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.7, 
                            edgecolor='none', boxstyle='round,pad=0.2'))
            
            # Axis labels
            ax.set_xlabel(r'$x$', fontsize=22, labelpad=5)
            if i > 0:
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        axs[0].set_ylabel(r'$y$', fontsize=22, labelpad=10)

        # Colorbar with proper spacing
        cbar_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'$u(\mathbf{x}, \mathbf{y},t)$', fontsize=22, rotation=270, labelpad=30)


        # Final adjustments
        plt.subplots_adjust(top=0.92, bottom=0.15)
        plt.savefig(f'/Users/rushirbhavsar/Pictures/GIFs-Model-Variation/{name}_plot_{v}.pdf', 
                bbox_inches='tight', pad_inches=0.05)
        plt.close()