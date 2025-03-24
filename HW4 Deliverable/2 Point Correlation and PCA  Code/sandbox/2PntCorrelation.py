import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import fftconvolve
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----- Functions for Two-Point Correlation and Directional Profiles -----
def two_point_correlation_fft(image):
    """
    Compute the 2D autocorrelation (two-point correlation) of a 2D image using FFT.
    Returns a normalized correlation map with maximum value = 1.
    """
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

# ----- Specify the folder containing your PNG files -----
folder_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW4 Deliverable\2 Point Correlation Code\sandbox"
file_pattern = os.path.join(folder_path, "*.png")
filepaths = sorted(glob.glob(file_pattern))
print("Found the following PNG files:")
for fp in filepaths:
    print(fp)

num_images = len(filepaths)
if num_images == 0:
    print("No PNG images found. Check your folder path.")
    exit()

# ----- Lists to store directional profiles for compilation plots -----
all_radial = []
all_vertical = []
all_horizontal = []
all_labels = []

# ----- Create a Tkinter Window with a Scrollable Canvas -----
root = tk.Tk()
root.title("Scrollable Two-Point Correlation & Directional Profiles")

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

# ----- Process Each Image and Embed Its Figure in Tkinter -----
for i, fp in enumerate(filepaths, start=1):
    print(f"\nProcessing image {i}/{num_images}: {fp}")
    
    # Load image and convert to grayscale (float32).
    img_pil = Image.open(fp).convert('L')
    img_array = np.array(img_pil, dtype=np.float32)
    
    # Compute the 2D two-point correlation map.
    corr_map = two_point_correlation_fft(img_array)
    
    # Compute directional profiles.
    r_vals, radial_profile = radial_average(corr_map)
    v_slice = vertical_slice(corr_map)
    h_slice = horizontal_slice(corr_map)
    
    # Save directional profiles for compilation plots.
    all_radial.append(radial_profile)
    all_vertical.append(v_slice)
    all_horizontal.append(h_slice)
    label = os.path.basename(fp)
    all_labels.append(label)
    
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
    filename = os.path.basename(fp)
    fig.suptitle(f"File: {filename}", fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Embed the figure in the Tkinter frame.
    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    widget = canvas_fig.get_tk_widget()
    widget.pack(padx=10, pady=10)
    
    plt.close(fig)  # Close the figure to free memory.

# ----- Create Compilation Plots and Embed Them in Tkinter -----
# Combined Radial Average Plot
fig_radial, ax_radial = plt.subplots(figsize=(10, 6))
for r_profile, label in zip(all_radial, all_labels):
    ax_radial.plot(np.arange(len(r_profile)), r_profile, marker='o', linestyle='-', label=label)
ax_radial.set_title("Compilation: Radial Averages")
ax_radial.set_xlabel("Radius (pixels)")
ax_radial.set_ylabel("Correlation")
ax_radial.legend(fontsize=9)
ax_radial.grid(True)
plt.tight_layout()
canvas_fig_radial = FigureCanvasTkAgg(fig_radial, master=frame)
canvas_fig_radial.get_tk_widget().pack(padx=10, pady=10)
plt.close(fig_radial)

# Combined Vertical Slice Plot
fig_vertical, ax_vertical = plt.subplots(figsize=(10, 6))
for v_profile, label in zip(all_vertical, all_labels):
    ax_vertical.plot(np.arange(len(v_profile)), v_profile, marker='s', linestyle='--', label=label)
ax_vertical.set_title("Compilation: Vertical Slices")
ax_vertical.set_xlabel("Pixel Offset")
ax_vertical.set_ylabel("Correlation")
ax_vertical.legend(fontsize=9)
ax_vertical.grid(True)
plt.tight_layout()
canvas_fig_vertical = FigureCanvasTkAgg(fig_vertical, master=frame)
canvas_fig_vertical.get_tk_widget().pack(padx=10, pady=10)
plt.close(fig_vertical)

# Combined Horizontal Slice Plot
fig_horizontal, ax_horizontal = plt.subplots(figsize=(10, 6))
for h_profile, label in zip(all_horizontal, all_labels):
    ax_horizontal.plot(np.arange(len(h_profile)), h_profile, marker='^', linestyle='-.', label=label)
ax_horizontal.set_title("Compilation: Horizontal Slices")
ax_horizontal.set_xlabel("Pixel Offset")
ax_horizontal.set_ylabel("Correlation")
ax_horizontal.legend(fontsize=9)
ax_horizontal.grid(True)
plt.tight_layout()
canvas_fig_horizontal = FigureCanvasTkAgg(fig_horizontal, master=frame)
canvas_fig_horizontal.get_tk_widget().pack(padx=10, pady=10)
plt.close(fig_horizontal)

# Start the Tkinter main loop.
root.mainloop()
