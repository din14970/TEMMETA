#!/usr/bin/python3.7

"""
This function applies a Butterworth low pass Fourier filter to a 2D image.
Input: a 2D image, the increment (radial size in reciprocal space in pixels) and order of the filter.
Output: the Butterworth-filtered image
"""

# Libraries
import numpy as np


def bwfilter(image, inc, order):

    # Subtract mean value from image
    (Nx, Ny) = image.shape
    image = image - np.mean(image)

    # Generate Fourier coordinates in 2D
    psize = 1 / inc
    lsize = np.array([Nx, Ny]) * psize

    if Nx % 2 == 0:
        qx = np.roll((np.arange(-Nx / 2, Nx / 2) / lsize[0]), np.rint(-Nx / 2).astype(int, casting='unsafe'),
                     axis=0)
    else:
        qx = np.roll((np.arange(-Nx / 2 + .5, Nx / 2 - .5) / lsize[0]),
                     np.rint(-Nx / 2 + .5).astype(int, casting='unsafe'), axis=0)
    if Ny % 2 == 0:
        qy = np.roll((np.arange(-Ny / 2, Ny / 2) / lsize[0]), np.rint(-Ny / 2).astype(int, casting='unsafe'),
                     axis=0)
    else:
        qy = np.roll((np.arange(-Ny / 2 + .5, Ny / 2 - .5) / lsize[0]),
                     np.rint(-Ny / 2 + .5).astype(int, casting='unsafe'), axis=0)

    # Apply low pass Butterworth filter
    qxa, qya = np.meshgrid(qx, qy, sparse=True)
    q2 = np.square(qxa) + np.square(qya)
    wfilt = 1 - 1 / np.sqrt(1 + np.power(q2, order) / np.power(.5, 16))

    butterfilt = np.real(np.fft.ifft2(np.fft.fft2(image) * wfilt))

    return butterfilt
