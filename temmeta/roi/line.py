import numpy as np
import scipy.ndimage as nd


def _line_profile_coordinates(src, dst, linewidth=1):
    """Return the coordinates of the profile of an image along a scan line.
    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line
    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of
        the profile is the ceil of the computed length of the scan line.
    Notes
    -----
    This is a utility method meant to be used internally by skimage
    functions. The destination point is included in the profile, in
    contrast to standard numpy indexing.
    """
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = np.ceil(np.hypot(d_row, d_col) + 1).astype(int)
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)
    data = np.zeros((2, length, linewidth))
    data[0, :, :] = np.tile(line_col, [linewidth, 1]).T
    data[1, :, :] = np.tile(line_row, [linewidth, 1]).T

    if linewidth != 1:
        # we subtract 1 from linewidth to change from pixel-counting
        # (make this line 3 pixels wide) to point distances (the
        # distance between pixel centers)
        col_width = (linewidth - 1) * np.sin(-theta) / 2
        row_width = (linewidth - 1) * np.cos(theta) / 2
        row_off = np.linspace(-row_width, row_width, linewidth)
        col_off = np.linspace(-col_width, col_width, linewidth)
        data[0, :, :] += np.tile(col_off, [length, 1])
        data[1, :, :] += np.tile(row_off, [length, 1])
    return data


def profile_line(img, src, dst, linewidth=1, averaged=False,
                 order=1, mode='constant', cval=0.0):
    """Return the intensity profile of an image measured along a scan line.
    Parameters
    ----------
    img : numeric array, shape (M, N[, C])
        The image, either grayscale (2D array) or multichannel
        (3D array, where the final axis contains the channel
        information).
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line
    averaged : bool, optional
        When linewidth is >1, whether to average (True) or to
        sum (false) over the linewidth
    order : int in {0, 1, 2, 3, 4, 5}, optional
        The order of the spline interpolation to compute image values at
        non-integer coordinates. 0 means nearest-neighbor interpolation.
    mode : string, one of {'constant', 'nearest', 'reflect', 'wrap'},
            optional
        How to compute any values falling outside of the image.
    cval : float, optional
        If `mode` is 'constant', what constant value to use outside the
        image.
    Returns
    -------
    return_value : array
        The intensity profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.
    Examples
    --------
    >>> x = np.array([[1, 1, 1, 2, 2, 2]])
    >>> img = np.vstack([np.zeros_like(x), x, x, x, np.zeros_like(x)])
    >>> img
    array([[0, 0, 0, 0, 0, 0],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [0, 0, 0, 0, 0, 0]])
    >>> profile_line(img, (2, 1), (2, 4))
    array([ 1.,  1.,  2.,  2.])
    Notes
    -----
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    p0 = (src[0],
          src[1])
    p1 = (dst[0],
          dst[1])
    if linewidth < 0:
        raise ValueError("linewidth must be positive number")
    linewidth_px = linewidth
    linewidth_px = int(round(linewidth_px))
    # Minimum size 1 pixel
    linewidth_px = linewidth_px if linewidth_px >= 1 else 1
    perp_lines = _line_profile_coordinates(p0, p1, linewidth=linewidth_px)
    pixels = nd.map_coordinates(img, perp_lines,
                                order=order, mode=mode, cval=cval)
    if averaged:
        intensities = pixels.mean(axis=1)
    else:
        intensities = pixels.sum(axis=1)
    return intensities
