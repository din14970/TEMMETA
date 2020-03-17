#!/usr/bin/python3.7

"""
This function computes the Fast Fourier Transform (FFT) of a high
resolution image, applies a gaussian filter, adjusts the intensity
threshold and cuts out the central region of interest. This function
can be called from any other script.

Input: a 2d image, the kernel size of the gaussian filter (needs to
be odd) and the sigma value. k_size is typically 7 or 9, sigma 4.

Output: the smoothed FFT of the image fft_blur
"""

# Libraries
import numpy as np
from scipy.ndimage import convolve
# import argparse


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gausfilter(arr, ks=7, sig=4):
    kernel = matlab_style_gauss2D(shape=(ks, ks), sigma=sig)
    return convolve(arr, kernel, mode="constant", cval=0)


def fft2_image(image, k_size=7, sigma=4, cs=20, crop_factor=3):

    image = np.array(image)

    assert k_size % 2 == 1, "kernel size must be odd!"
    assert image.shape[1] == image.shape[0], "image size must be square"
    assert np.log2(image.shape[1]) % 1 == 0, "image must be 2^n sized"
    assert np.log2(image.shape[0]) % 1 == 0, "image must be 2^n sized"
    # Take FFT of image
    fft2 = np.fft.fft2(image)
    fft2_shift = np.fft.fftshift(fft2)
    fft2_db = 20 * np.log10(np.abs(fft2_shift))  # power spectrum? use log10

    # Smooth FFT with gaussian filter: k_size has to be odd!
    fft2_blur = gausfilter(fft2_db, k_size, sigma)

    # Make mask to boost contrast and only show central part of FFT
    (Nx, Ny) = fft2_db.shape
    x = np.arange(-Nx // 2, Nx // 2, 1)
    y = np.arange(-Ny // 2, Ny // 2, 1)
    xv, yv = np.meshgrid(x, y)
    mask_radius = np.sqrt(np.square(xv) + np.square(yv))
    mask = mask_radius > cs
    fft2_masked = fft2_blur * mask  # cut out central spot
    fft2_masked_max = np.max(fft2_masked)  # find max for normalization
    fft2_blur = fft2_blur / fft2_masked_max
    # normalize but don't consider very intense central disk
    fft2_blur[fft2_blur > 1] = 1  # linear cut off
    crop_factor = crop_factor*2
    fft2_blur = fft2_blur[Nx//2-Nx//crop_factor:Nx//2+Nx//crop_factor,
                          Ny//2-Ny//crop_factor:Ny//2+Ny//crop_factor]

    return fft2_blur


def main():
    '''
    Wrapper for main function using argeparsesd
    '''
    # TODO write wrapper


if __name__ == "__main__":
    main()
