import numpy as np
from skimage import io, filters 
import numpy as np
import matplotlib as plt
from scipy.fft import fft2, ifft2 
import matplotlib.pyplot as plt

image  = io.imread("microstructure.tif")
thresh = filters.threshold_otsu(image)
binary = (image > thresh).astype(np.float32)

def calculate_two_point_correlation(image):
    """
    Calcs the two point correlatio function of a binary image

    args:
        image (numpy.ndarray): a 2d numpy array represetning the binary image

    returns:
        numpy.ndarray: the 2d two point correlation function
    """

    #conver the image to float for calculations
    image = image.astype(float)

    #calcs the fft of the image
    fft_image =fft2(image) 

    #calc the power spectrum
    power_spectrum = np.abs(fft_image)**2

    #calc the inverse fft of the power spectrum
    correlation = ifft2(power_spectrum)

    #normalize the correlation function
    correlation /=np.max(correlation)

    return np.real(correlation)


def plot_correlation(correlation, title ="Two point correlation function"):

    plt.imshow(correlation, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    image = np.random.randint(200, 500, size= (100, 100))

    correlation = calculate_two_point_correlation(image)

    plot_correlation(correlation)





'''
microstructure = np.array([[1, 0, 1], [0, 1, 0], [1, 0,1]])

count = 0
total_pairs =0

for i in range(height):
    for j in range(width):
        #reference pixel
        if microstructure[i, j] != 1:
            continue

        #target position with periodic 

'''
