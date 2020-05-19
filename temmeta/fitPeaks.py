"""
This function reads in an atomic resolution (S)TEM image, or takes it as
input through calling this function in another script. It then applies a
Butterworth and Gaussian filter to smooth out noise and does a simple peak
fitting. The user has then the option to refine the peak positions by
non-linear Gaussian fitting.

Input: atomic resolution (S)TEM image
Output: peaks
        - normal peak fitting: [x, y, peak intensity]
        - Gaussian fitting:    [x, y, peak intensity, sigma, x refined,
                                y refined, background from fit, integrated
                                peak intensity]
"""

# Libraries
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.spatial import distance_matrix

# Own functions
from . import image_filters as imf


def calculate_NN_network(coords, max_r=None, max_nn=None):
    """
    Calculate the nearest neighbor distributuion from a point cloud

    Parameters
    ----------
    xy : array, PxD
        rows represent coordinates of individual points, with P the number
        of points and D the point space dimensionality
    max_r : float, optional
        maximum distance in units of x and y to include in nearest neighbors.
        Default is not defined. All NN with r>max_r are set to nan.
    max_nn : int, optional
        maximum number of nearest neighbors to store. Default is None.
        This determines the output dimension K. If none then max_nn = P.

    Returns
    -------
    nn : array, DxPxK
        The nearest neighbor coordines. Each PxK matrix (nn[i]) represents
        an axis (i.e x, y, z, ...), of which the rows represent the K-1 nearest
        neighbors of each point in the point cloud. The first column is the
        point itself, the other columns the coordinates of its K-1 nearest
        neighbors ordered from nearest to furthest.

    Examples
    --------
    >>>xy = np.array([[0, 0], [4, 0], [0, 3]])
    >>>calculate_NN_network(xy)
    array([[[0., 0., 4.],
            [4., 0., 0.],
            [0., 0., 4.]],
          [[0., 3., 0.],
           [0., 0., 3.],
           [3., 0., 0.]]])
    >>>calculate_NN_network(xy, max_r=4)
    array([[[0., 0., 4.],
            [4., 0., nan],
            [0., 0., nan]],
          [[0., 3., 0.],
           [0., 0., nan],
           [3., 0., nan]]])
    >>>calculate_NN_network(xy, max_nn=2)
    array([[[0., 0.],
            [4., 0.],
            [0., 0.]],
          [[0., 3.],
           [0., 0.],
           [3., 0.]]])
    """
    coords = coords.astype(float)
    d = distance_matrix(coords, coords)
    # sort the columns from smallest to biggest and get the original index
    dsort = np.sort(d, axis=1)
    mask = None
    if max_r is not None:
        # filter out the distances that are too large and set to nan
        dsort[dsort > max_r] = np.nan
        # create a mask
        mask = dsort != dsort
    indx = np.argsort(d, axis=1)
    if max_nn is not None:
        indx = indx[:, :max_nn]
        if mask is not None:
            mask = mask[:, :max_nn]
    # get the coordinate 3D matrix, just do with loop over dims for now
    nn = []
    for i in range(coords.shape[1]):
        coord = coords[:, i]
        coord_matrix = coord[indx]
        if mask is not None:
            coord_matrix[mask] = np.nan
        nn.append(coord_matrix)
    nn = np.array(nn)
    return nn


def _get_repeated_original_coords(nn):
    """
    Return the initial coordinates with the shape of nn for addition
    or subtraction
    """
    initial_coords = nn[:, :, 0].T
    init_coord_mat = np.repeat(initial_coords[:, :, np.newaxis],
                               nn.shape[-1], axis=2)
    init_coord_mat = np.swapaxes(init_coord_mat, 0, 1)
    return init_coord_mat


def get_NN_distance(nn):
    """
    Calculate the euler distances between nearest neighbors

    Parameters
    ----------
    nn : array, DxPxK
        The nearest neighbor coordines. See :method:calculate_NN_network

    Returns
    -------
    dist: array, PxK
        Euler distances between first K nearest neighbors of P points. The
        first column will be 0's as this represents the distance from each
        point to itself
    """
    inco = _get_repeated_original_coords(nn)
    diff = nn-inco
    diff = diff**2
    dist = np.sqrt(diff.sum(axis=0))
    return dist


def get_NN_angle(nn, vector=None, usesense=True, sense=None, units="radians"):
    """
    Calculate angles towards nearest neighbors

    Parameters
    ----------
    nn : array, DxPxK
        Nearest neighbor array with dimensions D (number of dimensions)
        of the original point cloud, P (number of points), K (number of
        nearest neighbors)
    vector : array, D, optional
        Vector relative to which the angles to the nearest neighbors are
        calculated. Must have the same dimensionality as the point cloud.
        By default, a unit vector parrallel to the first dimension of the
        point cloud is chosen (usually x).
    usesense : bool, optional
        Whether to use the full +/-180 degrees rotation range or not. If False
        the result of arccos is returned. If True, the NN position relative
        to the hyperplane is considered to define positive and negative
        angles. Makes the most sense in 2D point clouds where all points
        lie in the plane.
    sense : array, D, optional
        Vector that defines the normal of a hyper-plane for dividing the
        space and defining positive (above the plane) and negative angles
        (below the plane). Must be perpendicular to vector. By default it
        is a unit vector parallel to the second dimension of the point
        cloud (usually y).
    units : str, optional
        Return format. Options: "radians" or "degrees"

    Returns
    -------
    angle : array, PxK
        Angles for each point (rows) to its first K (columns) nearest
        neighbors
    """
    # must be 3d array and at least 2 dimensions to the point cloud
    if nn.ndim != 3:
        raise ValueError("Nearest neighbor coordinate array must be 3D")
    if nn.shape[0] < 2:
        raise ValueError("Point cloud must have at least two coordinates for"
                         "angles")
    if vector is None:
        vector = np.zeros(nn.shape[0])
        vector[0] = 1.
    else:
        vector = np.array(vector)
        if len(vector) != nn.shape[0]:
            raise ValueError("Reference vector has the wrong number of"
                             "dimensions.")
    if sense is None:
        sense = np.zeros(nn.shape[0])
        sense[1] = 1.
    else:
        sense = np.array(sense)
        if len(sense) != nn.shape[0]:
            raise ValueError("Sense vector has the wrong number of"
                             "dimensions.")
    if np.dot(sense, vector) != 0.:
        raise ValueError("Sense and vector must be perpendicular")
    vector = vector/np.linalg.norm(vector)
    sense = sense/np.linalg.norm(sense)
    inco = _get_repeated_original_coords(nn)
    diff = nn-inco
    dist = get_NN_distance(nn)
    # tile the vector
    tv = np.tile(vector, (nn.shape[1], nn.shape[2], 1))
    tv = np.rollaxis(tv, -1)
    # multiply the tiled vector and diff, then sum = dot product
    dp = (tv*diff).sum(axis=0)
    # get the sense if using sense
    if usesense:
        ts = np.tile(sense, (nn.shape[1], nn.shape[2], 1))
        ts = np.rollaxis(ts, -1)
        # multiply the tiled vector and diff, then sum = dot product. Then we
        # just use the sign.
        san = np.sign((ts*diff).sum(axis=0))
        # wherever it is 0 (vector is in the plane), we set it to 1
        san[san == 0] = 1
    else:
        # we don't use sense
        san = 1
    with np.errstate(divide='ignore', invalid='ignore'):
        cosan = np.divide(dp, dist)
    # return arccos of result
    angle = np.arccos(cosan)*san
    if units == "radians":
        return angle
    elif units == "degrees":
        return np.rad2deg(angle)
    else:
        raise ValueError(f"Units not recognized: {units}")


def _detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)
    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask
    # we create the mask of the background
    background = (image == 0)
    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    return detected_peaks*1


def find_peaks(image, do_bw=True, bwf_args=(100, 8), do_gauss=True,
               gauss_args=(7, 2), min_dist=10, min_int=0.2, bound=5):
    """
    Find the (x, y, intensity) of atom column peak positions

    Parameters
    ----------
    image : array, 2D
        The image to find peaks in
    do_bw : bool, optional
        Whether to perform a butterworth filtering on the image
    bwf_args : 2-element tuple, optional
        Arguments passed to butterworth filter if used
    do_gauss : bool, optional
        Whether to perform a gaussian filter on the image
    gauss_args : 2-element tuple, optional
        Arguments passed to gauss_filter
    min_dist : float, optional
        Minimum distance between peaks in pixels
    min_int : float between 0-1, optional
        Minimum intensity at the peak
    bound : float, optional
        Minimum distance of a peak from the edge in pixels

    Returns
    -------
    result : pandas dataframe
        (x, y, intensity) of peaks dataframe
    """
    # Normalize image
    image = np.array(image)
    image = np.squeeze(image)  # remove single dimensional axes
    image = imf.normalize(image)
    # Determine image properties
    # shape no. rows (Nx) x no. columns (Ny) x no. channels (Nz)
    (Nx, Ny) = image.shape
    # Apply Butterworth filter
    if do_bw:
        image = imf.bw_filter(image, bwf_args[0], bwf_args[1])
    # Apply Gaussian filter
    if do_gauss:
        k_size = gauss_args[0]
        sigma = gauss_args[1]
        image = imf.gauss_filter(image, ks=k_size, sig=sigma,
                                 cval=image.mean())
    # Normalize image
    image_filt = imf.normalize(image)
    # Determine lattice peaks with indices x_p, y_p
    peaks = _detect_peaks(image_filt)
    # get x and y coordinates where peaks is nonzero
    (y_p, x_p) = np.nonzero(peaks*1)
    num_peaks = int(np.size(x_p))
    # reshape to column vector, add 1 to account for crop window
    x_p = x_p.reshape(num_peaks, 1)
    y_p = y_p.reshape(num_peaks, 1)
    # find the intensity associated with the peak
    I_p = image[x_p, y_p]
    # make an array [x, y, intensity]
    peaks = np.hstack((x_p, y_p, I_p))
    # sort the peaks by intensity
    peaks = peaks[np.argsort(peaks[:, 2])]
    # Remove peaks too close together
    del_peak = np.ones(num_peaks, dtype=bool)
    for a0 in range(0, num_peaks - 1, 1):
        d2 = (x_p[a0] - x_p[a0 + 1:]) ** 2 + (y_p[a0] - y_p[a0 + 1:]) ** 2
        if np.min(d2) < (min_dist ** 2):
            del_peak[a0] = False

    peaks = peaks[del_peak, :]
    # Remove low intensity peaks
    min_peaks = peaks[:, 2] > min_int
    peaks = peaks[min_peaks, :]
    # Remove peaks too close to image boundaries
    del_bound = np.logical_and.reduce(
        (peaks[:, 0] > bound, peaks[:, 0] < Nx-bound,
         peaks[:, 1] > bound, peaks[:, 1] < Ny-bound))
    peaks = peaks[del_bound, :]
    peaks = pd.DataFrame(peaks, columns=["x", "y", "intensity"])
    return peaks


def find_peaks2(image, do_bw=True, bwf_args=(100, 8), do_gauss=True,
                gauss_args=(7, 2), min_dist=10, min_int=0.2, bound=5):
    # Normalize image
    image = np.array(image)
    image = np.squeeze(image)  # remove single dimensional axes
    image = imf.normalize(image)

    # Determine image properties
    # shape no. rows (Nx) x no. columns (Ny) x no. channels (Nz)
    (Nx, Ny) = image.shape

    # Apply Butterworth filter
    if do_bw:
        image = imf.bw_filter(image, bwf_args[0], bwf_args[1])

    # Apply Gaussian filter
    if do_gauss:
        k_size = gauss_args[0]
        sigma = gauss_args[1]
        image = imf.gauss_filter(image, ks=k_size, sig=sigma,
                                 cval=image.mean())

    # Normalize image
    image_filt = imf.normalize(image)

    # Determine lattice peaks with indices x_p, y_p
    # cut off border of 1 pixel around
    image_mid = image_filt[1:-1, 1:-1]
    # move window around and check in which pixels mid remains highest
    peaks = np.logical_and.reduce((image_mid > image_filt[0:-2, 0:-2],
                                   image_mid > image_filt[1:-1, 0:-2],
                                   image_mid > image_filt[2:, 0:-2],
                                   image_mid > image_filt[0:-2, 1:-1],
                                   image_mid > image_filt[2:, 1:-1],
                                   image_mid > image_filt[0:-2, 2:],
                                   image_mid > image_filt[1:-1, 2:],
                                   image_mid > image_filt[2:, 2:]))

    # get x and y coordinates where peaks is nonzero
    (y_p, x_p) = np.nonzero(peaks*1)
    num_peaks = int(np.size(x_p))

    # reshape to column vector, add 1 to account for crop window
    x_p = x_p.reshape(num_peaks, 1)
    x_p += 1
    y_p = y_p.reshape(num_peaks, 1)
    y_p += 1
    # find the intensity associated with the peak
    I_p = image[x_p, y_p]

    # make an array [x, y, intensity]
    peaks = np.hstack((x_p, y_p, I_p))
    # sort the peaks by intensity
    peaks = peaks[np.argsort(peaks[:, 2])]

    # Remove peaks too close together
    del_peak = np.ones(num_peaks, dtype=bool)
    for a0 in range(0, num_peaks - 1, 1):
        d2 = (x_p[a0] - x_p[a0 + 1:]) ** 2 + (y_p[a0] - y_p[a0 + 1:]) ** 2

        if np.min(d2) < (min_dist ** 2):
            del_peak[a0] = False

    peaks = peaks[del_peak, :]

    # Remove low intensity peaks
    min_peaks = peaks[:, 2] > min_int
    peaks = peaks[min_peaks, :]

    # Remove peaks too close to image boundaries
    del_bound = np.logical_and.reduce(
        (peaks[:, 0] > bound, peaks[:, 0] < Nx-bound,
         peaks[:, 1] > bound, peaks[:, 1] < Ny-bound))
    peaks = peaks[del_bound, :]

    peaks = pd.DataFrame(peaks, columns=["x", "y", "intensity"])
    return peaks


def gaussian_peak_fit(image, peaks, d_xy=0.5, rCut=5, rFit=4, sigma0=5,
                      sigmaMin=2, sigmaMax=9, iterations=2):
    """
    Further refine peaks by fitting a 2D gaussian to the image

    Parameters
    ----------
    image : array, 2D
        The image to find peaks in
    peaks : array with (x, y, intensity)
        Returned from find_peaks
    d_xy : float, optional
        max allowed shift for fitting
    rCut : int, optional
        size of cutting area around inital peak
    rFit : int, optional
        size of fitting radius
    sigma0 : float, optional
        initial guess for the sigma of the gaussian
    sigmaMin : float, optional
        minimum allowed value of sigma
    sigmaMax : float, optional
        maximum allowed value of sigma
    iterations : float, optional
        number of least squares iterations

    Returns
    -------
    peaks : pandas DataFrame
        Refined peaks with columns:
        * Col 1: x coordinate of original peak
        * Col 2: y coordinate of original peak
        * Col 3: peak intensity of original peak
        * Col 4: sigma (standard deviation) of Gaussian function
        * Col 5: refined x coordinate
        * Col 6: refined y coordinate
        * Col 7: background value determined by fit
        * Col 8: integrated peak intensity without background
    """
    # Fitting coordinates
    peaks = peaks.values
    (Nx, Ny) = image.shape
    num_peaks, _ = np.shape(peaks)
    x_coord = np.arange(0, Nx, 1)
    y_coord = np.arange(0, Ny, 1)
    x_a, y_a = np.meshgrid(x_coord, y_coord, sparse=False)

    # Define 2D Gaussian function
    def func(c, x_func, y_func, int_func):
        return (c[0] * np.exp(-1/2 / c[1] ** 2 *
                              ((x_func - c[2]) ** 2 +
                               (y_func - c[3]) ** 2)) + c[4] - int_func)
    # Loop through inital peaks and fit by non-linear Gaussian functions
    peaks_refine = []
    for p0 in range(0, num_peaks, 1):
        # Initial peak positions
        x = peaks[p0, 0]
        xc = int(np.rint(x))
        y = peaks[p0, 1]
        yc = int(np.rint(y))
        # Cut out subsection around the peak
        x_sub_0 = np.max((xc-rCut, 0))
        x_sub_1 = np.min((xc+rCut, Nx)) + 1
        y_sub_0 = np.max((yc-rCut, 0))
        y_sub_1 = np.min((yc+rCut, Ny)) + 1
        # Make indices of subsection
        x_cut = x_a[y_sub_0: y_sub_1, x_sub_0: x_sub_1]
        y_cut = y_a[y_sub_0: y_sub_1, x_sub_0: x_sub_1]
        cut = image[y_sub_0: y_sub_1, x_sub_0: x_sub_1]
        # Inital values for least-squares fitting
        k = np.min(cut)
        int_0 = np.max(cut)-k
        sigma = sigma0
        # Sub-pixel iterations
        for _ in range(0, iterations):
            sub = (x_cut - x) ** 2 + (y_cut - y) ** 2 < rFit ** 2
            # Fitting coordinates
            x_fit = x_cut[sub]
            y_fit = y_cut[sub]
            int_fit = cut[sub]
            # Initial guesses and bounds of fitting function
            c0 = [int_0, sigma, x, y, k]
            lower_bnd = [int_0*.8, max(sigma*0.8, sigmaMin),
                         x-d_xy, y-d_xy, k-int_0*0.5]
            upper_bnd = [int_0*1.2, min(sigma*1.2, sigmaMax),
                         x+d_xy, y+d_xy, k+int_0*0.5]
            # Linear least squares fitting
            peak_fit = least_squares(func, c0, args=(
                x_fit, y_fit, int_fit), bounds=(lower_bnd, upper_bnd))
            # Refined peak positions
            int_0 = peak_fit.x[0]
            sigma = peak_fit.x[1]
            x = peak_fit.x[2]
            y = peak_fit.x[3]
            k = peak_fit.x[4]
        # Write refined peak array: sigma (of Gaussian), x, y, k
        # (fitted background level), peak_int (integrated peak
        # intensity without background)
        peak_int = np.sum(cut * sub)
        # integrated intensity without background
        peaks_refine.append([sigma, x, y, k, peak_int])
    # Reshape refined peak array
    peaks_refine = np.asarray(peaks_refine)
    peaks_refine = peaks_refine.reshape(num_peaks, 5)
    peaks = np.hstack((peaks, peaks_refine))
    peaks = pd.DataFrame(peaks, columns=["x", "y", "intensity",
                                         "sigma", "x refined",
                                         "y refined", "background",
                                         "integrated intensity"])

    return peaks
