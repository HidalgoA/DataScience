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

# ---- Specified Filters ----
volume_fractions = ["01", "015", "02", "025", "030", "035", "040", "045", "050", "055", "060", "065", "070"]
materials = ["Holes", "MatA", "MatB"]
loads = ["1mm", "5mm"]

# Helper function using regex to match the volume fraction as a whole token.
def match_whole(filename, token):
    pattern = r'(?<!\d)' + re.escape(token) + r'(?!\d)'
    return re.search(pattern, filename) is not None

# ---- Functions for Two-Point Correlation and Directional Profiles ----
def two_point_correlation_fft(image):
    """
    Compute the 2D autocorrelation (two-point correlation) of a 2D image using FFT.
    Returns a normalized correlation map with maximum value = 1.
    """
    reversed_img = image[::-1, ::-1]
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

# ---- Specify the folder containing PNG files ----
folder_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW4 Deliverable\2 Point Correlation and PCA  Code\sandbox"
file_pattern = os.path.join(folder_path, "*.png")
all_filepaths = sorted(glob.glob(file_pattern))

# ---- Filter images based on defined keywords ----
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

# ---- Lists to store directional profiles for compilation plots ----
all_radial = []
all_vertical = []
all_horizontal = []
all_labels = []

# ---- Process Each Image to collect directional profiles ----
for i, fp in enumerate(filepaths, start=1):
    print(f"Processing image {i}/{num_images}: {fp}")
    
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
    all_labels.append(os.path.basename(fp))

# ---- Create a Tkinter Window with a Scrollable Canvas ----
root = tk.Tk()
root.title("Compilation Plots")

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

# ---- Create Compilation Plots and Embed Them in Tkinter ----
def create_compilation_plot(data, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    for profile, label in zip(data, all_labels):
        ax.plot(np.arange(len(profile)), profile, label=label)
    ax.set_title(title)
    ax.set_xlabel("Pixel Offset")
    ax.set_ylabel("Correlation")
    ax.legend(fontsize=9)
    ax.grid(True)
    plt.tight_layout()
    
    canvas_fig = FigureCanvasTkAgg(fig, master=frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack(padx=10, pady=10)
    # Keep the figure open by not calling plt.close(fig)

create_compilation_plot(all_radial, "Compilation: Radial Averages")
create_compilation_plot(all_vertical, "Compilation: Vertical Slices")
create_compilation_plot(all_horizontal, "Compilation: Horizontal Slices")

# Start the Tkinter main loop.
root.mainloop()
