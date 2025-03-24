import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

# ------------------------------------------------------------------------------
# 1) Specify Desired Volume Fractions, Materials, and Loads
# ------------------------------------------------------------------------------
volume_fractions = ["01", "015", "02", "025", "030", "035", "040", 
                    "045", "050", "055", "060", "065", "070"]
materials = ["MatB"]
loads = ["1mm", "5mm"]

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def match_whole(filename, token):
    """
    Regex-based helper to match a volume fraction as a separate token.
    e.g. '01' matches 'someMatA_01.png' but not 'someMatA_011mm.png' unless the pattern fits.
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

def extract_volume_fraction(filename, vol_frac_list):
    """
    Extract the volume fraction substring (e.g., "01", "035") from a filename.
    If none found, return "UnknownVF".
    """
    for vf in vol_frac_list:
        if match_whole(filename, vf):
            return vf
    return "UnknownVF"

# ------------------------------------------------------------------------------
#  Gather and Filter PNG Files
# ------------------------------------------------------------------------------
folder_path = r"C:\Users\hidal\Desktop\Classes\GRAD\Data Sci Mech of Mat\HW4 Deliverable\2 Point Correlation and PCA  Code\sandbox"
all_filepaths = sorted(glob.glob(os.path.join(folder_path, "*.png")))

filepaths = [
    fp for fp in all_filepaths 
    if any(match_whole(os.path.basename(fp), vf) for vf in volume_fractions)
    and any(mat in os.path.basename(fp) for mat in materials)
    and any(ld in os.path.basename(fp) for ld in loads)
]

print("Filtered PNG files:")
for fp in filepaths:
    print(fp)

if not filepaths:
    print("No matching images found! Check your filters.")
    exit()

# Extract volume fraction labels
vf_labels = [extract_volume_fraction(os.path.basename(fp), volume_fractions) for fp in filepaths]

# ------------------------------------------------------------------------------
# Compute Two-Point Correlation Maps & Resize
# ------------------------------------------------------------------------------
corr_maps = []
target_shape = (200, 200)

for fp in filepaths:
    img = Image.open(fp).convert('L')
    img_array = np.array(img, dtype=np.float32)
    corr_map = two_point_correlation_fft(img_array)
    corr_map_resized = resize_corr_map(corr_map, target_shape)
    corr_maps.append(corr_map_resized)

print("Example correlation map shape:", corr_maps[0].shape)
print(f"Number of correlation maps: {len(corr_maps)}")

# ------------------------------------------------------------------------------
# Flatten the Maps and Perform PCA
# ------------------------------------------------------------------------------
data_matrix = np.array([cm.ravel() for cm in corr_maps])
print("Data matrix shape (for PCA):", data_matrix.shape)

pca = PCA(n_components=3)
pca.fit(data_matrix)
scores = pca.transform(data_matrix)

print("\nExplained variance ratio (in %) for each principal component:")
for i, ratio in enumerate(pca.explained_variance_ratio_, start=1):
    print(f"  PC{i}: {ratio*100:.2f}%")

# ------------------------------------------------------------------------------
# 3D Scatter Plot of PCA Scores, Labeled by Volume Fraction
# ------------------------------------------------------------------------------
# Define a color palette or dictionary for each volume fraction
color_map = {
    "01": "red",
    "015": "blue",
    "02": "green",
    "025": "orange",
    "030": "purple",
    "035": "brown",
    "040": "pink",
    "045": "olive",
    "050": "cyan",
    "055": "magenta",
    "060": "lime",
    "065": "gray",
    "070": "teal",
    "UnknownVF": "black"
}

# Group the PCA scores by volume fraction label
scores_by_vf = {}
for score, vf in zip(scores, vf_labels):
    scores_by_vf.setdefault(vf, []).append(score)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for vf, group_scores in scores_by_vf.items():
    group_scores = np.array(group_scores)
    c = color_map.get(vf, "black")
    ax.scatter(group_scores[:, 0], group_scores[:, 1], group_scores[:, 2],
               color=c, label=vf, s=60)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("3D PCA Scatter: Colored by Volume Fraction")
ax.legend(title="VolFrac", fontsize=8)
plt.tight_layout()
plt.show()
