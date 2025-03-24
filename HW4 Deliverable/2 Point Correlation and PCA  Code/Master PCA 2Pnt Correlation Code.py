import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # use the TkAgg backend
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  

# ---- Specified Filters ----
volume_fractions = ["01", "015", "02", "025", "030", "035", "040", "045", "050", "055", "060", "065", "070"]
materials = ["Holes", "MatA", "MatB"]
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
    return corr / np.max(corr)

def radial_average(corr_map):
    """
    Compute the radial average of a 2D correlation map around its center.
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
    center_x = corr_map.shape[1] // 2
    return corr_map[:, center_x]

def horizontal_slice(corr_map):
    center_y = corr_map.shape[0] // 2
    return corr_map[center_y, :]

def resize_corr_map(corr_map, target_shape=(200, 200)):
    """
    Resize a 2D correlation map to a fixed target shape using SciPy's zoom.
    """
    zoom_factors = (target_shape[0] / corr_map.shape[0],
                    target_shape[1] / corr_map.shape[1])
    return zoom(corr_map, zoom_factors)

# ----Specify folder path and gather filtered PNG files ----
folder_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW4 Deliverable\2 Point Correlation and PCA  Code\sandbox"
file_pattern = os.path.join(folder_path, "*.png")
all_filepaths = sorted(glob.glob(file_pattern))

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

# ----Process images: correlation maps + directional profiles ----
all_radial = []
all_vertical = []
all_horizontal = []
all_labels = []
all_corr_maps = []
target_shape = (200, 200)

for i, fp in enumerate(filepaths, start=1):
    try:
        print(f"Processing image {i}/{num_images}: {fp}")
        img = Image.open(fp).convert('L')
        img_array = np.array(img, dtype=np.float32)
        
        corr_map = two_point_correlation_fft(img_array)
        corr_map_resized = resize_corr_map(corr_map, target_shape)
        
        r_vals, radial_profile = radial_average(corr_map_resized)
        v_slice = vertical_slice(corr_map_resized)
        h_slice = horizontal_slice(corr_map_resized)
        
        all_corr_maps.append(corr_map_resized)
        all_radial.append(radial_profile)
        all_vertical.append(v_slice)
        all_horizontal.append(h_slice)
        all_labels.append(os.path.basename(fp))
    except Exception as e:
        print(f"Error processing {fp}: {e}")

if not all_corr_maps:
    print("No images processed successfully!")
    exit()

print("Final correlation map shape:", all_corr_maps[0].shape)

# ---- Create single-figure approach for correlation maps ----
def create_window_with_correlation_maps_single_figure():
    """
    Displays all correlation maps in a single matplotlib Figure with a grid of subplots.
    Then embeds that single figure into a scrollable  window.
    """
    window = tk.Toplevel()
    window.title("Two-Point Correlation Maps (Single-Figure)")

    ncols = 4
    nrows = (len(all_corr_maps) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, (corr_map, label) in enumerate(zip(all_corr_maps, all_labels)):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        im = ax.imshow(corr_map, cmap='viridis', origin='lower')
        ax.set_title(label)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    total_subplots = nrows * ncols
    for idx in range(len(all_corr_maps), total_subplots):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')

    plt.tight_layout()

    canvas = tk.Canvas(window)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack()

    frame.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))

# ---- Plot compilation functions (grouped by material) ----
highlight_styles = {
    "MatA": {"color": "red", "linewidth": 3, "marker": "o", "linestyle": "-"},
    "MatB": {"color": "blue", "linewidth": 3, "marker": "s", "linestyle": "--"},
    "Holes": {"color": "green", "linewidth": 3, "marker": "^", "linestyle": "-."},
    "Unknown": {"color": "gray", "linewidth": 3, "marker": "x", "linestyle": ":"}
}

def create_compilation_plot(data, title, frame, highlight_material):
    fig, ax = plt.subplots(figsize=(10, 6))
    for profile, label in zip(data, all_labels):
        x_vals = np.arange(len(profile))
        if highlight_material in label:
            style = highlight_styles.get(highlight_material, {})
            ax.plot(x_vals, profile, label=label, **style)
        else:
            ax.plot(x_vals, profile, label=label)
    ax.set_title(title)
    ax.set_xlabel("Pixel Offset")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(padx=10, pady=10)

def create_compilation_plot_combined(data, title, frame):
    fig, ax = plt.subplots(figsize=(10, 6))
    for profile, label in zip(data, all_labels):
        x_vals = np.arange(len(profile))
        style = {}
        for mat in materials:
            if mat in label:
                style = highlight_styles.get(mat, {})
                break
        ax.plot(x_vals, profile, label=label, **style)
    ax.set_title(title)
    ax.set_xlabel("Pixel Offset")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(padx=10, pady=10)

def create_window_with_plots(highlight_material):
    window = tk.Toplevel()
    window.title(f"Compilation Plots - Highlight: {highlight_material}")
    canvas = tk.Canvas(window)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    
    create_compilation_plot(all_radial, "Compilation: Radial Averages", frame, highlight_material)
    create_compilation_plot(all_vertical, "Compilation: Vertical Slices", frame, highlight_material)
    create_compilation_plot(all_horizontal, "Compilation: Horizontal Slices", frame, highlight_material)

def create_window_with_combined_plots():
    window = tk.Toplevel()
    window.title("Compilation Plots - Combined Material Highlights")
    canvas = tk.Canvas(window)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=scrollbar.set)
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    
    create_compilation_plot_combined(all_radial, "Compilation: Radial Averages", frame)
    create_compilation_plot_combined(all_vertical, "Compilation: Vertical Slices", frame)
    create_compilation_plot_combined(all_horizontal, "Compilation: Horizontal Slices", frame)

# ---- PCA on radial profiles (grouped by material) with dimensionality reduction reporting ----
def resample_profile(profile, target_length):
    old_indices = np.linspace(0, 1, len(profile))
    new_indices = np.linspace(0, 1, target_length)
    return np.interp(new_indices, old_indices, profile)

def compute_and_plot_pca_radial(profiles, labels, target_length=100):
    # Resample profiles to a common length (target_length)
    resampled_profiles = [resample_profile(profile, target_length) for profile in profiles]
    data = np.array(resampled_profiles)
    
    # Print original dimension (each profile is now target_length long)
    original_dim = data.shape[1]
    print(f"Original data dimension (per image): {original_dim}")
    
    # Apply PCA to reduce dimensionality to 2 components
    pca = PCA(n_components=2)
    scores = pca.fit_transform(data)
    
    # Print reduced dimension and explained variance
    reduced_dim = 2
    explained_variance = pca.explained_variance_ratio_
    print(f"Reduced dimension: {reduced_dim}")
    print(f"Explained variance by components: {explained_variance}")
    
    scores_by_material = {}
    for score, label in zip(scores, labels):
        material_found = None
        for mat in materials:
            if mat in label:
                material_found = mat
                break
        if material_found is None:
            material_found = "Unknown"
        scores_by_material.setdefault(material_found, []).append(score)
    window = tk.Toplevel()
    window.title("PCA of Radial Profiles (2 Components) - Grouped by Material")
    fig, ax = plt.subplots(figsize=(8, 6))
    for mat, group_scores in scores_by_material.items():
        group_scores = np.array(group_scores)
        color = highlight_styles.get(mat, {}).get("color", "black")
        ax.scatter(group_scores[:, 0], group_scores[:, 1], color=color, label=mat)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of Radial Profiles (2 Components)\nExplained Variance: {:.2f}% and {:.2f}%".format(
        explained_variance[0]*100, explained_variance[1]*100))
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    canvas_fig = FigureCanvasTkAgg(fig, master=window)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(padx=10, pady=10)

def compute_and_plot_pca_radial_3d(profiles, labels, target_length=100):
    resampled_profiles = [resample_profile(profile, target_length) for profile in profiles]
    data = np.array(resampled_profiles)
    original_dim = data.shape[1]
    print(f"Original data dimension (per image): {original_dim}")
    pca = PCA(n_components=3)
    scores = pca.fit_transform(data)
    reduced_dim = 3
    explained_variance = pca.explained_variance_ratio_
    print(f"Reduced dimension: {reduced_dim}")
    print(f"Explained variance by components: {explained_variance}")
    
    scores_by_material = {}
    for score, label in zip(scores, labels):
        material_found = None
        for mat in materials:
            if mat in label:
                material_found = mat
                break
        if material_found is None:
            material_found = "Unknown"
        scores_by_material.setdefault(material_found, []).append(score)
    window = tk.Toplevel()
    window.title("PCA of Radial Profiles (3 Components) - Grouped by Material")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for mat, group_scores in scores_by_material.items():
        group_scores = np.array(group_scores)
        color = highlight_styles.get(mat, {}).get("color", "black")
        ax.scatter(group_scores[:, 0], group_scores[:, 1], group_scores[:, 2],
                   color=color, label=mat)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA of Radial Profiles (3 Components)\nExplained Variance: {:.2f}%, {:.2f}%, {:.2f}%".format(
        explained_variance[0]*100,
        explained_variance[1]*100,
        explained_variance[2]*100))
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    canvas_fig = FigureCanvasTkAgg(fig, master=window)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(padx=10, pady=10)

# ---- PCA eigenimages on entire correlation maps with dimensionality reduction reporting ----
data_matrix = np.array([cm.ravel() for cm in all_corr_maps])
print("Original data matrix shape for PCA:", data_matrix.shape)
original_dimension = data_matrix.shape[1]  # e.g., 200*200 = 40000
print(f"Each correlation map originally has {original_dimension} dimensions.")

# Apply PCA to reduce dimensions to 3
pca_full = PCA(n_components=3)
data_reduced = pca_full.fit_transform(data_matrix)
new_shape = data_reduced.shape
print(f"After PCA, reduced data matrix shape: {new_shape}.") 
reduced_dimension = 3
explained_variance_full = pca_full.explained_variance_ratio_
print(f"Explained variance by the 3 components: {explained_variance_full}")

# Reshape the PCA components back to image dimensions for visualization (eigenimages)
target_shape_full = (all_corr_maps[0].shape[0], all_corr_maps[0].shape[1])
eigenimages = [comp.reshape(target_shape_full) for comp in pca_full.components_]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, eigenimage in enumerate(eigenimages, start=1):
    im = axes[i-1].imshow(eigenimage, cmap='jet', origin='lower', aspect='equal')
    axes[i-1].set_title(f"Principal Component {i} (Eigenimage)")
    axes[i-1].axis('off')
    fig.colorbar(im, ax=axes[i-1], fraction=0.046, pad=0.04)
plt.suptitle("Eigenimages (First Three Principal Components) on 2-Point Correlation Maps", fontsize=16)
plt.tight_layout()
plt.show()

# ---- Launch everything via Tkinter ----
root = tk.Tk()
root.withdraw()

# Windows for each material
for mat in materials:
    create_window_with_plots(mat)

# Combined window
create_window_with_combined_plots()

# Use the single-figure approach for correlation maps
create_window_with_correlation_maps_single_figure()

# PCA on radial profiles grouped by material
compute_and_plot_pca_radial(all_radial, all_labels)
compute_and_plot_pca_radial_3d(all_radial, all_labels)

root.mainloop()
