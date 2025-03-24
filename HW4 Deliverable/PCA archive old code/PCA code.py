import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import fftconvolve
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  

# ---- specified filters ----
volume_fractions = ["01", "015", "02", "025", "030", "035", "040", "045", "050", "055", "060", "065", "070"]  #   Specify the volume fractions to include [01 015 02 025 030 035 040 045 050 055 060 065 070]
materials = ["Holes", "MatA", "MatB"]              # Choose from [MatA MatBm Holes]
loads = ["1mm", "5mm"]            # Specify the displacement values [1mm 5mm]

# 
def match_whole(filename, token):

    pattern = r'(?<!\d)' + re.escape(token) + r'(?!\d)'
    return re.search(pattern, filename) is not None

# Functions for two poit correlation and directional profiles 
def two_point_correlation_fft(image):
    
    reversed_img = image[::-1, ::-1]  # Reverse image along both axes.
    corr = fftconvolve(image, reversed_img, mode='full')
    corr_norm = corr / np.max(corr)
    return corr_norm

def radial_average(corr_map):
    """
    Compute the radial average of a 2D correlation map around its center.
    Returns:
        r_vals (1D array): Distances from the center (pixels).
        radial_profile (1D array): Radially averaged correlation values.
    """
    center_y = corr_map.shape[0] // 2
    center_x = corr_map.shape[1] // 2
    y_indices, x_indices = np.indices(corr_map.shape)
    r = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    r_int = r.astype(np.int32)
    tbin = np.bincount(r_int.ravel(), weights=corr_map.ravel())
    nr = np.bincount(r_int.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    r_vals = np.arange(len(radial_profile))
    return r_vals, radial_profile

def vertical_slice(corr_map):
    """
    Extract the vertical slice (center column) from the correlation map.
    """
    center_x = corr_map.shape[1] // 2
    return corr_map[:, center_x]

def horizontal_slice(corr_map):
    """
    Extract the horizontal slice (center row) from the correlation map.
    """
    center_y = corr_map.shape[0] // 2
    return corr_map[center_y, :]

#  Specify the folder containing PNG files 
folder_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW4 Deliverable\2 Point Correlation Code\sandbox"
file_pattern = os.path.join(folder_path, "*.png")
all_filepaths = sorted(glob.glob(file_pattern))

#  Filter images based on user-defined keywords 
filepaths = [
    fp for fp in all_filepaths if 
    any(match_whole(os.path.basename(fp), vf) for vf in volume_fractions) and
    any(mat in os.path.basename(fp) for mat in materials) and
    any(ld in os.path.basename(fp) for ld in loads)
]

print("Filtered PNG files:")
for fp in filepaths:
    print(fp)

num_images = len(filepaths)
if num_images == 0:
    print("No matching images found. Check your filters.")
    exit()

# Lists to store computed data 
all_corr_maps = []   # Numerical two-point correlation arrays
all_labels = []      # Load labels for PCA color-coding
all_radial = []      # Radial profiles (for display)
all_vertical = []    # Vertical slice
all_horizontal = []  # Horizontal slice

#  Create a Tkinter Window with a Scrollable Canvas 
root = tk.Tk()
root.title("Two-Point Correlation & Directional Profiles")

canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas.configure(yscrollcommand=scrollbar.set)
frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor="nw")

def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
frame.bind("<Configure>", on_configure)

#  Process Each Image 
for i, fp in enumerate(filepaths, start=1):
    print(f"\nProcessing image {i}/{num_images}: {fp}")
    # Load image and convert to grayscale (float32)
    img_pil = Image.open(fp).convert('L')
    img_array = np.array(img_pil, dtype=np.float32)
    
    # Compute the 2D two-point correlation map.
    corr_map = two_point_correlation_fft(img_array)
    all_corr_maps.append(corr_map)
    
    # Determine load label from filename (simple check)
    load_label = next((ld for ld in loads if ld in os.path.basename(fp)), "Unknown")
    all_labels.append(load_label)
    
    # Compute directional profiles.
    r_vals, radial_profile = radial_average(corr_map)
    v_slice = vertical_slice(corr_map)
    h_slice = horizontal_slice(corr_map)
    all_radial.append(radial_profile)
    all_vertical.append(v_slice)
    all_horizontal.append(h_slice)
    
    # Create a figure with 3 subplots.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # (A) Original Grayscale Image.
    axes[0].imshow(img_array, cmap='gray', origin='upper')
    axes[0].set_title(f"Original (Image #{i})")
    axes[0].axis("off")
    
    # (B) 2D Two-Point Correlation Map.
    im_corr = axes[1].imshow(corr_map, cmap='jet', origin='lower', aspect='equal')
    axes[1].set_title("2D Two-Point Correlation")
    axes[1].axis("off")
    fig.colorbar(im_corr, ax=axes[1], fraction=0.046, pad=0.04)
    
    # (C) Directional Profiles (Combined).
    axes[2].plot(r_vals, radial_profile, label="Radial Avg", marker='o', linestyle='-')
    axes[2].plot(np.arange(len(v_slice)), v_slice, label="Vertical Slice", marker='s', linestyle='--')
    axes[2].plot(np.arange(len(h_slice)), h_slice, label="Horizontal Slice", marker='^', linestyle='-.')
    axes[2].set_title("Directional Profiles")
    axes[2].set_xlabel("Pixel Offset")
    axes[2].set_ylabel("Correlation")
    axes[2].grid(True)
    axes[2].legend(fontsize=10)
    
    # Add filename as a super-title.
    fig.suptitle(f"File: {os.path.basename(fp)}", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Embed the figure in the Tkinter frame.
    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    widget = canvas_fig.get_tk_widget()
    widget.pack(padx=10, pady=10)
    
    plt.close(fig)  # Free up memory

#  PCA on the Numerical Two-Point Correlation Arrays 
# First, pad all correlation maps to a common shape.
max_rows = max(cm.shape[0] for cm in all_corr_maps)
max_cols = max(cm.shape[1] for cm in all_corr_maps)
target_shape = (max_rows, max_cols)

def pad_to_shape(array, shape):
    rows_diff = shape[0] - array.shape[0]
    cols_diff = shape[1] - array.shape[1]
    pad_top = rows_diff // 2
    pad_bottom = rows_diff - pad_top
    pad_left = cols_diff // 2
    pad_right = cols_diff - pad_left
    return np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)),
                  mode='constant', constant_values=0)

padded_corr_maps = [pad_to_shape(cm, target_shape) for cm in all_corr_maps]

# Flatten each padded correlation map into a vector for PCA.
data_matrix = np.array([cm.ravel() for cm in padded_corr_maps])
print("Data matrix shape for PCA:", data_matrix.shape)

# Perform PCA (3 components).
pca = PCA(n_components=3, svd_solver='full', random_state=10)
pc_scores = pca.fit_transform(data_matrix)
evr = pca.explained_variance_ratio_
print("\nExplained variance ratio per component:", evr)
print("Cumulative explained variance:", np.cumsum(evr))

# Reshape each principal component back into 2D (eigenimages).
eigenimages = [comp.reshape(target_shape) for comp in pca.components_]

# Plot the First Three Principal Components (Eigenimages).
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, eig_img in enumerate(eigenimages, start=1):
    im = axes[i-1].imshow(eig_img, cmap='jet', origin='lower', aspect='equal')
    axes[i-1].set_title(f"PC{i} (Explained Var: {evr[i-1]*100:.2f}%)")
    axes[i-1].axis("off")
    fig.colorbar(im, ax=axes[i-1], fraction=0.046, pad=0.04)
plt.suptitle("First Three Principal Components (Eigenimages)", fontsize=16)
plt.tight_layout()
plt.show()

# 3D Scatter Plot of PCA Scores, Color-coded by Load.
color_map = {"1mm": "red", "5mm": "blue", "Unknown": "gray"}
colors = [color_map[label] for label in all_labels]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(num_images):
    ax.scatter(
        pc_scores[i, 0], pc_scores[i, 1], pc_scores[i, 2],
        color=colors[i],
        s=80,
        label=all_labels[i]  
    )
# Remove duplicate labels in legend.
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D Scatter Plot of PCA Scores")
plt.show()

# Start the Tkinter Main Loop
root.mainloop()
