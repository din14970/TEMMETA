"""
Collection of image filters and modifyers

Functions
---------
bin2
    Rebin an image using PIL.Image.resize
bin2_simple
    Rebin an image by adding together pixels
bw_filter
    Apply a butterworth filter to an image
gauss_filter
    Apply a Gaussian filter to an image
linscale
    Scale the intensity of an image to a new domain linearly
med_filter
    Apply a median filter to an image
normalize
    Linearly scale image intensity to a domain between 0 to 1
scale_percent
    Rescale image intensity to remove intensity outliers
scale_std
    Rescale image intensity to the mean+/- a multiple of the standard
    deviation
suppress_outliers
    Set pixel outliers to the minimum and maximum determined by
    a percentage.
"""

import numpy as np
from scipy.ndimage import median_filter, convolve
from PIL import Image


def normalize(arr):
    """
    Rescale image intensities to 0 and 1

    Parameters
    ----------
    arr : array-like object
        The image to normalize

    Returns
    -------
    result : numpy array
        Image with intensity normalized between 0 and 1

    Notes
    -----
    Wrapper for method :py:func:`linscale`
    """
    return linscale(arr)


def linscale(arr, min=None, max=None, nmin=0, nmax=1, dtype=np.float):
    """
    Rescale image intensities

    Rescale the image intensities to a new scale. The value
    min and everything below gets mapped to nmin, max and everything
    above gets mapped to nmax. By default the minimum and maximum
    get mapped to 0 and 1.

    Parameters
    ----------
    arr : array-like object
        The image to normalize
    min : float, optional
        The intensity to map to the new minimum value. Defaults to
        the minimum of the provided array.
    max : float, optional
        The intensity to map to the new maximum value. Defaults to
        the maximum of the provided array.
    nmin : float, optional
        The new minimum of the image. Defaults to 0.
    nmax : float, optional
        The new maximum of the image. Defaults to 1. For 8-bit images use 255.
        For 16-bit images use 65535.
    dtype : type, optional
        The data type of the output array. Defaults to float. See the possible
        data types here:
        <https://docs.scipy.org/doc/numpy/user/basics.types.html>

    Returns
    -------
    result : array
        Intensity-rescaled image

    Notes
    -----
    The type recasting happens in an 'unsafe' manner. That is, if elements
    have float values like 0.99, recasting to np.uint8 will turn this into 0
    and not 1.

    Examples
    --------
    >>>s=np.array([[1, 2], [3, 4]])
    >>>linscale(s)
    array([[0.        , 0.33333333],
           [0.66666667, 1.        ]])
    >>>linscale(s, min=2, max=3, nmin=1, nmax=2, dtype=np.uint8)
    array([[1, 1],
           [2, 2]], dtype=uint8)
    """
    if min is None:
        min = arr.min()
    else:
        arr[arr < min] = min
    if max is None:
        max = arr.max()
    else:
        arr[arr > max] = max

    a = (nmax-nmin)/(max-min)
    result = (arr-min)*a+nmin
    return result.astype(dtype)


def scale_std(arr, numstd=3):
    """
    Rescales the image intensity to mean +/- n*standard deviation

    Parameters
    ----------
    arr : array-like
        The image to normalize
    numstd : float, optional
        Number of standard deviations to define intensity rescaling boundaries.
        Default is 3.

    Returns
    -------
    result : array with dtype float
        Intensity-rescaled image between 0 and 1

    See also
    --------
    :func:`linscale`
    """
    arr = np.array(arr)
    mean = np.mean(arr)
    std = np.std(arr)

    nstd = numstd*std

    return linscale(arr, mean - nstd, mean + nstd)


def scale_percent(arr, percent=0.07):
    """
    Rescales the image intensity to within an intensity interval

    Calculates the value of intensity above which and below which a certain
    percentage of pixels are more and less intense respectively. The intensity
    is rescaled to these boundaries. The aim is mainly to remove dead and
    hot pixels at the low and high end of the spectrum.

    Parameters
    ----------
    arr : array-like
        The image to normalize
    percent : float, optional
        Percentage of pixels to consider as outlier
        Default is 0.07.

    Returns
    -------
    result : array with dtype float
        Intensity-rescaled image between 0 and 1

    See also
    --------
    :func:`linscale`
    """
    minval, maxval = _find_outliers(arr, percent)
    return linscale(arr, minval, maxval)


def _matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    Returns 2D Gaussian kernel

    Notes
    -----
    Should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])

    See also
    --------
    :py:func:`gausfilter`
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gauss_filter(arr, ks=7, sig=4.):
    """
    Apply Gaussian kernel filter on image

    Blurs an image and fudges noise

    Parameters
    ----------
    arr : array-like object
        The image to filter
    ks : int, optional
        The kernel size. Default is 7.
    sig : float, optional
        The Gaussian standard deviation. Default is 4.

    Returns
    -------
    result : numpy array
        filtered image
    """
    kernel = _matlab_style_gauss2D(shape=(ks, ks), sigma=sig)
    return convolve(arr, kernel, mode="constant", cval=0)


def med_filter(arr, ks=5):
    """
    Apply median kernel filter on image

    Blurs an image and is mainly used for removing outliers.
    Particularly useful for diffraction patterns with hot pixels.

    Parameters
    ----------
    arr : array-like object
        The image to filter
    ks : int, optional
        The kernel size. Default is 5.

    Returns
    -------
    result : numpy array
        filtered image
    """
    return median_filter(arr, size=ks, mode="constant", cval=0)


def cnn_filter(arr):
    # TODO
    pass


def _find_outliers(raw, percent=0.07, bins=(2**16)):
    """
    Find intensity values below and above which we consider outliers

    Calculates the value of intensity above which and below which a certain
    percentage of pixels are more and less intense respectively. The aim is to
    remove mainly dead and hot pixels at the low and high end of the spectrum.

    Parameters
    ----------
    arr : array-like object
        The image to calculate the outlier intensities from
    percent : float, optional
        The percentage to use as cut-off. The first and last *x* percent of
        pixels are considered outliers. Default is 0.07.
    bins : int, optional
        The boundaries are calculated based on a histogram. This determines
        the number of bins this histogram uses. Ideally, the same bittedness
        of the image is used, e.g. for uint8 this is 2**8. Default is 2**16.

    Returns
    -------
    minimum_edge : float
        Intensity value below which pixels are considered outliers
    maximum_edge : float
        Intensity value above which pixels are considered outliers
    """
    raw = np.array(raw)
    cnts, bin_edges = np.histogram(raw, bins=bins)
    stats = np.zeros((2, bins), dtype=np.int)
    # Calculate cumulative intensity distribution
    stats[0] = np.cumsum(cnts)  # low
    stats[1] = np.cumsum(cnts[::-1])  # high
    thresh = stats > percent * raw.shape[0] * raw.shape[1]
    # The minimum and maximum bin
    min = (np.where(thresh[0]))[0][0]
    max = bins - (np.where(thresh[1]))[0][0]

    return bin_edges[min], bin_edges[max+1]


def suppress_outliers(raw, percent=0.07):
    """
    Set pixels with outlier intensities to threshold values

    Calculates the value of intensity above which and below which a certain
    percentage of pixels are more and less intense respectively. The aim is to
    remove mainly dead and hot pixels at the low and high end of the spectrum.
    This function is similar to :func:`scale_percent` but does not rescale.

    Parameters
    ----------
    raw : array-like
        The image to suppress outliers from
    percent : float, optional
        Percentage of pixels to consider as outlier

    Returns
    -------
    result : array with dtype float
        Intensity-rescaled image between 0 and 1
    """
    min, max = _find_outliers(raw, percent)

    raw[raw < min] = min
    raw[raw > max] = max

    return raw


def bw_filter(image, inc=100, order=8):
    """
    This function applies a Butterworth high pass Fourier filter to a 2D image.

    Attenuates the very low frequency components in the image. The effect
    is to even out the intensity of the image.

    Parameters
    ----------
    image : array-like
        The 2D image to be filtered
    inc : float, optional
        The increment or radial size in recyprocal space. In units of pixels.
        Default is 100.
    order : float, optional
        Order of the filter. Default is 8.

    Returns
    -------
    result : numpy array with dtype float
        The Butterworth-filtered image. Normalized between 0 and 1.

    See also
    --------
    <https://en.wikipedia.org/wiki/Butterworth_filter>
    """
    # Make sure the image is an array
    image = np.array(image)
    # Subtract mean value from image
    (Nx, Ny) = image.shape
    image = image - np.mean(image)

    # Generate Fourier coordinates in 2D
    psize = 1 / inc
    lsize = np.array([Nx, Ny]) * psize

    # create array linearly ascending and decending to the middle
    if Nx % 2 == 0:
        qx = np.roll(np.arange(-Nx / 2, Nx / 2) / lsize[0],
                     np.rint(-Nx / 2).astype(int, casting='unsafe'),
                     axis=0)
    else:
        qx = np.roll(np.arange(-Nx / 2 + .5, Nx / 2 - .5) / lsize[0],
                     np.rint(-Nx / 2 + .5).astype(int, casting='unsafe'),
                     axis=0)
    if Ny % 2 == 0:
        qy = np.roll(np.arange(-Ny / 2, Ny / 2) / lsize[1],
                     np.rint(-Ny / 2).astype(int, casting='unsafe'),
                     axis=0)

    else:
        qy = np.roll(np.arange(-Ny / 2 + .5, Ny / 2 - .5) / lsize[1],
                     np.rint(-Ny / 2 + .5).astype(int, casting='unsafe'),
                     axis=0)

    # Apply low pass Butterworth filter
    qxa, qya = np.meshgrid(qx, qy, sparse=True)
    q2 = np.square(qxa) + np.square(qya)
    wfilt = 1 - 1 / np.sqrt(1 + np.power(q2, order) / np.power(.5, 16))

    butterfilt = np.real(np.fft.ifft2(np.fft.fft2(image) * wfilt))

    return normalize(butterfilt)


def bin2(a, factor, resample=Image.NEAREST):
    """
    Rebin an image

    A more advanced rebinning of a 2D image

    Parameters
    ----------
    a : array-like
        The 2D array representing an image
    factor : int
        The image is resized to initial size/factor.
    resample : int
        See <https://pillow.readthedocs.io/en/3.1.x/reference/
        Image.html#PIL.Image.Image.resize>. Default is Image.NEAREST.

    Returns
    -------
    binned : array-like
        The binned image
    """
    assert len(a.shape) == 2
    # binned = imresize(a, (a.shape[0]//factor, a.shape[1]//factor))
    binned = np.array(
        Image.fromarray(a).resize(size=(a.shape[0]//factor,
                                        a.shape[1]//factor),
                                  resample=resample))
    return binned


def bin2_simple(a, factor):
    """
    Rebin an image

    A simple additive rebinning of a 2D array

    Parameters
    ----------
    a : array-like
        The 2D array representing an image
    factor : int
        The image is resized to initial size/factor.

    Returns
    -------
    binned : array-like
        The binned image

    Notes
    -----
    This is based on: http://scipy.org/Cookbook/Rebinning
    It is simplified to the case of a 2D array with the same
    binning factor in both dimensions.
    """
    # TODO: better error handling concerning valid factor values
    assert len(a.shape) == 2
    newshape = int(a.shape[0]/factor), int(a.shape[1]/factor)
    tmpshape = (newshape[0], factor, newshape[1], factor)
    f = factor * factor
    binned = np.sum(np.sum(np.reshape(a, tmpshape), 1), 2) / f
    return binned