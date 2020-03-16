#!/usr/bin/python3.7

"""
This function dislpays an atomic resolution image and its corresponding fitted lattice peaks!
"""

# Libraries
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def plotImagePeaks(image, peaks):

    # Display the image (using matplotlib)
    # Create new figure window
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.clf()
    plt.figure(1)
    plt.imshow(image, cmap='plasma')
    plt.scatter(peaks[:, 1], peaks[:, 0], c='b', s=5)
    plt.axis('off')
    plt.colorbar()
    plt.show()
    plt.close()
