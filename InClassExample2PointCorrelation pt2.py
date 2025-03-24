import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters

def two_point_direct(binary, r_max):
    """
    Computes a 1D radial two-point correlation function for a binary array
    with periodic boundary conditions. This version returns a 1D result.
    
    Parameters
    ----------
    binary : 2D numpy array of 0s and 1s
        The input binary image or grid.
    r_max : int
        The maximum radius (in pixels) to consider.

    Returns
    -------
    S2 : 1D numpy array
        The two-point correlation values as a function of radius r=1..r_max-1.
    """
    height, width = binary.shape
    S2 = np.zeros(r_max, dtype=np.float64)
    counts = np.zeros(r_max, dtype=np.float64)
    phi = np.mean(binary)  # volume fraction (or fraction of 1s)

    for i in range(height):
        for j in range(width):
            if binary[i, j] == 1:
                for di in range(-r_max, r_max + 1):
                    for dj in range(-r_max, r_max + 1):
                        r = int(np.hypot(di, dj))
                        if 0 < r < r_max:
                            # Periodic boundary conditions
                            i2 = (i + di) % height
                            j2 = (j + dj) % width
                            if binary[i2, j2] == 1:
                                S2[r] += 1
                            counts[r] += 1

    # Normalize by the expected number of 1–1 pairs and avoid division by zero
    return S2 / (counts * phi * 2 + 1e-12)


def radial_average(S2_2d):
    """
    Computes the radial average of a 2D correlation function.
    
    NOTE: This function expects a 2D array, but the 'two_point_direct' 
          above returns a 1D array. They are *not* directly compatible 
          without modification.
    """
    L = S2_2d.shape[0]
    # Create a grid of (y, x) coordinates centered at (L/2, L/2)
    y, x = np.indices((L, L)) - L / 2

    # Compute the radius of each coordinate
    r = np.sqrt(x**2 + y**2).astype(int)

    # Flatten for bincount
    S2_flat = S2_2d.ravel()
    r_flat = r.ravel()

    # Set up bins from 0 to the maximum radius
    r_bins = np.arange(0, np.max(r) + 1)

    # Weighted sum in each radial bin
    radial_sum = np.bincount(r_flat, weights=S2_flat, minlength=len(r_bins))
    counts = np.bincount(r_flat, minlength=len(r_bins))

    return radial_sum / (counts + 1e-12)


def main():
    # 1. Load and binarize (grayscale image -> threshold -> binary)
    image = io.imread("microstructure.tif")
    thresh = filters.threshold_otsu(image)
    binary = (image > thresh).astype(np.float32)

    # 2. Compute the two-point correlation (1D radial version)
    r_max = 50
    S2_1d = two_point_direct(binary, r_max=r_max)

    # 3. (Optional) Radial average of a 2D correlation
    #    NOTE: The code below will NOT work as-is with S2_1d 
    #    because radial_average expects a 2D array.
    #
    #    If you had a 2D correlation array called S2_2d, you could do:
    #      S2_radial = radial_average(S2_2d)
    #    For the 1D result, we can simply plot it directly.

    # 4. Plot the 1D two-point correlation
    r_values = np.arange(len(S2_1d))
    plt.plot(r_values, S2_1d, marker='o')
    plt.xlabel('Radial Distance (pixels)')
    plt.ylabel('S2')
    plt.title('Two-Point Correlation Function (1D Radial)')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
