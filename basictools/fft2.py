"""
Functions related to calculating FFTs of images.

Functions
---------
fft2_image
    Create a filtered and intensity scaled power spectrum of an image
"""

# Libraries
import numpy as np
from .image_filters import gauss_filter
# import argparse - TODO


def fft2_image(image, k_size=7, sigma=4., cs=20., crop_factor=3):
    '''
    Computer quality FFT of image

    This function computes the Fast Fourier Transform (FFT) of a high
    resolution image, applies a gaussian filter, adjusts the intensity
    threshold (the central spot is always extremely intense) and crops
    out the central region of interest.

    Parameters
    ----------
    image : 2D array like object
        The 2d image
    k_size : int, optional
        The kernel size of the gaussian filter. Needs to
        be odd. Default is 7.
    sigma : float, optional
        Sigma of the Gaussian. Default is 4.0.
    cs: float, optional
        Mask radius for automatic intensity scaling. Default is 20.0.
    crop_factor: float, optional
        1/crop_factor of the image is used centered around the middle

    Returns
    -------
    fft_blur : 2D numpy array
        The filtered Fourrier spectrum image. In fact it is a power spectrum.
    '''
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
    fft2_blur = gauss_filter(fft2_db, k_size, sigma)

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
    xbound = int(round(Nx/crop_factor))
    ybound = int(round(Ny/crop_factor))
    fft2_blur = fft2_blur[Nx//2-xbound:Nx//2+xbound,
                          Ny//2-ybound:Ny//2+ybound]

    return fft2_blur


def main():
    '''
    Wrapper for main function using argeparsesd
    '''
    # TODO write wrapper


if __name__ == "__main__":
    main()
