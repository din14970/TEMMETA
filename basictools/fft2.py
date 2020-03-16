#!/usr/bin/python3.7

"""
This function computes the Fast Fourier Transform (FFT) of a high resolution image, applies a gaussian filter,
adjusts the intensity threshold and cuts out the central region of interest. This function can be called from any
other script.
Input: a 2d image, the kernel size of the gaussian filter (needs to be odd) and the sigma value. k_size is typically
7 or 9, sigma 4.
Output: the smoothed FFT of the image fft_blur
"""

# Libraries
from numpy import square, abs, fft, log, sqrt, max, arange, rint, meshgrid
import cv2


def fft2_image(image, k_size, sigma):

    # Take FFT of image
    fft2 = fft.fft2(image)
    fft2_shift = fft.fftshift(fft2)
    fft2 = 20 * log(abs(fft2_shift))

    # Smooth FFT with gaussian filter: k_size has to be odd!
    fft2_blur = cv2.GaussianBlur(fft2,(k_size,k_size),sigma)

    # Make mask to boost contrast and only show central part of FFT
    (Nx, Ny) = fft2.shape
    x = arange(-Nx / 2, Nx / 2, 1)
    y = arange(-Ny / 2, Ny / 2, 1)
    xv, yv = meshgrid(x, y, sparse=False)
    mask_radius = sqrt(square(xv) + square(yv))
    mask = mask_radius > 20
    fft2_masked = fft2_blur * mask
    fft2_masked_max = max(fft2_masked)
    fft2_blur = fft2_blur / fft2_masked_max
    fft2_blur[fft2_blur > 1] = 1
    fft2_blur = fft2_blur[rint(Nx * 1 / 3).astype(int, casting='unsafe'):rint(Nx * 2 / 3).astype(int, casting='unsafe'),
                rint(Ny * 1 / 3).astype(int, casting='unsafe'):rint(Ny * 2 / 3).astype(int, casting='unsafe')]

    return fft2_blur

