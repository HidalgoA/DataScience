import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Ensure we use the TkAgg backend
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# ---- Specified Filters ----
volume_fractions = ["01", "015", "02", "025", "030", "035", "040", "045", "050", "055", "060", "065", "070"]
materials = ["MatB"]
loads = ["1mm", "5mm"]

def match_whole(filename, token):
    """
    Regex-based helper to match a volume fraction as a separate token.
    """
    pattern = r'(?<!\d)' + re.escape(token) + r'(?!\d)'
    return re.search(pattern, filename) is not None

def two_point_correlation_fft(image):
    """
    Compute the 2D autocorrelation (two-point correlation) of a 2D image using FFT.
    Returns a normalized correlation map with maximum value = 1.
    """
    reversed_img = image[::-1, ::-1]
    corr = fftconvolve(image, reversed_img, mode='full')
    corr_norm = corr / np.max(corr)
    return corr_norm

def resize_corr_map(corr_map, target_shape=(200, 200)):
    """
    Resize a 2D correlation map to a fixed target shape using SciPy's zoom.
    """
    zoom_factors = (target_shape[0] / corr_map.shape[0],
                    target_shape[1] / corr_map.shape[1])
    return zoom(corr_map, zoom_factors)

def get_load(filename, load_list):
    """
    Extract the load substring (e.g., "1mm" or "5mm") from a filename.
    """
    for ld in load_list:
        if ld in filename:
            return ld
    return "Unknown"

# -------------------------------------------------------------------------
# Gather and Filter PNG Files
# -------------------------------------------------------------------------
folder_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW4 Deliverable\2 Point Correlation and PCA  Code\sandbox"
file_pattern = os.path.join(folder_path, "*.png")
all_filepaths = sorted(glob.glob(file_pattern))

# Filter by volume fractions, materials, and loads
filepaths = [
    fp for fp in all_filepaths if
    any(match_whole(os.path.basename(fp), vf) for vf in volume_fractions)
    and any(mat in os.path.basename(fp) for mat in materials)
    and any(ld in os.path.basename(fp) for ld in loads)
]

print("Filtered PNG files:")
for fp in filepaths:
    print(fp)

num_images = len(filepaths)
if num_images == 0:
    print("No matching images found. Check your filters.")
    exit()

# Extract load labels for each image
load_labels = [get_load(os.path.basename(fp), loads) for fp in filepaths]

# -------------------------------------------------------------------------
# Compute Two-Point Correlation Maps & Resize
# -------------------------------------------------------------------------
corr_maps = []
target_shape = (200, 200)  # Common shape for correlation maps

for fp in filepaths:
    # Load image and convert to grayscale (float32)
    img = Image.open(fp).convert('L')
    img_array = np.array(img, dtype=np.float32)
    
    # Compute the 2-point correlation map
    corr_map = two_point_correlation_fft(img_array)
    
    # Resize the correlation map
    corr_map_resized = resize_corr_map(corr_map, target_shape)
    corr_maps.append(corr_map_resized)

print("Example correlation map shape:", corr_maps[0].shape)

# -------------------------------------------------------------------------
# Flatten Maps for PCA
# -------------------------------------------------------------------------
data_matrix = np.array([cm.ravel() for cm in corr_maps])
print("Data matrix shape (for PCA):", data_matrix.shape)

# -------------------------------------------------------------------------
# Perform PCA, Print Variance, and Plot Eigenimages
# -------------------------------------------------------------------------
pca = PCA(n_components=3)
pca.fit(data_matrix)

print("\nExplained variance ratio (in %) for each principal component:")
for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"  PC{i}: {ratio*100:.2f}%")

components = pca.components_
eigenimages = [comp.reshape(target_shape) for comp in components]

# Plot the eigenimages (first three PCs)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, eigenimage in enumerate(eigenimages, start=1):
    im = axes[i-1].imshow(eigenimage, cmap='jet', origin='lower', aspect='equal')
    axes[i-1].set_title(f"Principal Component {i}")
    axes[i-1].axis('off')
    fig.colorbar(im, ax=axes[i-1], fraction=0.046, pad=0.04)
plt.suptitle("Eigenimages (First Three Principal Components) on 2-Point Correlation Maps", fontsize=16)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# 3D Scatter Plot of PCA Scores Labeled by Load
# -------------------------------------------------------------------------
scores = pca.transform(data_matrix)

# Color mapping for loads
load_colors = {"1mm": "blue", "5mm": "red", "Unknown": "gray"}

# Group scores by load
scores_by_load = {}
for score, ld in zip(scores, load_labels):
    scores_by_load.setdefault(ld, []).append(score)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for ld, group_scores in scores_by_load.items():
    group_scores = np.array(group_scores)
    ax.scatter(group_scores[:, 0], group_scores[:, 1], group_scores[:, 2],
               color=load_colors.get(ld, "black"), label=ld, s=60)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("3D Scatter Plot of PCA Scores Labeled by Load")
ax.legend(title="Load", fontsize=10)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Single-Figure Approach for Two-Point Correlation Plots (Scroll in Tkinter)
# -------------------------------------------------------------------------
def create_window_with_correlation_maps_single_figure():
    """
    Displays ALL correlation maps in a single matplotlib Figure with a grid of subplots,
    then embeds that figure into a scrollable Tkinter window.
    """
    window = tk.Toplevel()
    window.title("Two-Point Correlation Maps (Single-Figure)")

    ncols = 4
    nrows = (len(corr_maps) + ncols - 1) // ncols  # integer division rounding up

    # Create one big figure with a grid of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = np.array(axes).reshape(nrows, ncols)  # ensure 2D array for indexing

    for idx, (cm, fp) in enumerate(zip(corr_maps, filepaths)):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        im = ax.imshow(cm, cmap='viridis', origin='lower')
        ax.set_title(os.path.basename(fp), fontsize=8)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    total_subplots = nrows * ncols
    for idx in range(len(corr_maps), total_subplots):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')

    plt.tight_layout()

    # Now embed this single figure in a scrollable Tkinter canvas
    canvas = tk.Canvas(window)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    # Create the Tkinter canvas figure
    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    canvas_fig.draw()
    canvas_widget = canvas_fig.get_tk_widget()
    canvas_widget.pack()

    # Update the scrollable region
    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

# Create the Tkinter root and call our function
root = tk.Tk()
root.withdraw()

create_window_with_correlation_maps_single_figure()

# Start the Tkinter event loop
root.mainloop()
