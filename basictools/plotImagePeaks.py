# TODO Documentation normalization

"""
This function dislpays an atomic resolution image and its corresponding
fitted lattice peaks!
"""

# Libraries
from . import image_filters as imf
import matplotlib.pyplot as plt
import numpy as np


def plotImagePeaks(image, peaks, ax=None):
    # Display the image (using matplotlib)
    # Create new figure window
    image = np.array(image)
    image = imf.normalize(image)
    if ax is None:
        plt.clf()
        plt.figure(1)
        plt.imshow(image, cmap='plasma')
        plt.scatter(peaks[:, 1], peaks[:, 0], c='b', s=5)
        plt.axis('off')
        plt.colorbar()
        plt.show()
        plt.close()
    else:
        fig = ax.figure
        im = ax.imshow(image, cmap='plasma')
        ax.scatter(peaks[:, 1], peaks[:, 0], c='b', s=5)
        ax.axis('off')
        fig.colorbar(im, ax=ax)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
