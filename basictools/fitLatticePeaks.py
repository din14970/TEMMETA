# TODO Documentation normalization
# TODO Refactor to take out Tk dependence
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
import tkinter as tk
from tkinter import filedialog, Tk, messagebox
import matplotlib
import numpy as np
import cv2
from scipy.optimize import least_squares

# Own functions
from . import image_filters as imf
from . import plotImagePeaks


def fitLatPeaks(image=None, bwf_args=(100, 8),
                gauss_args=(7, 2), min_dist=10, min_int=0.2, bound=5):

    # If *args is empty ask user to open file
    if not image:

        # Ask for user input to open image
        root: Tk = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.update()
        root.destroy()

        # Read an image from file
        image = cv2.imread(file_path)
        image = image[:, :, 0]

    # Normalize image
    image = np.array(image)
    image = np.squeeze(image)  # remove single dimensional axes
    image = cv2.normalize(image, None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Determine image properties
    # shape no. rows (Nx) x no. columns (Ny) x no. channels (Nz)
    (Nx, Ny) = image.shape

    # Apply Butterworth filter
    image_filt = imf.bw_filter(image, bwf_args[0], bwf_args[1])

    # Apply Gaussian filter
    k_size = gauss_args[0]
    sigma = gauss_args[1]
    image_filt = cv2.GaussianBlur(image_filt, (k_size, k_size), sigma)

    # Normalize image
    image_filt = cv2.normalize(image_filt, None, alpha=0, beta=1,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

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
    (x_p, y_p) = np.nonzero(peaks*1)
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
    peaks_Nx, peaks_Ny = np.shape(peaks)
    num_peaks = peaks_Nx

    # Display the image (using matplotlib)
    plotImagePeaks.plotImagePeaks(image, peaks)

    # Ask for user input to refine fit with non-linear double
    # Gaussian functions
    root: Tk = tk.Tk()
    root.withdraw()
    answer = messagebox.askyesno("Non-linear Gaussian peak fitting?",
                                 "Should lattice peaks be fitted with "
                                 "non-linear Gaussian functions? \n \n"
                                 "Be aware, this can "
                                 "take several minutes!")
    if answer is True:
        # Inital parameters for non-linear peak fitting
        # Nsub_pix_iterations = 3     # number of sub-pixel iterations
        d_xy = 0.5                  # max allowed shift for fitting
        rCut = 5                    # size of cutting area around inital peak
        rFit = 4                    # size of fitting radius
        sigma0 = 5
        sigmaMin = 2
        sigmaMax = 9
        # damp = 2/3                  # Damping rate

        # Fitting coordinates
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
            xc = np.rint(x).astype(int, casting='unsafe')
            y = peaks[p0, 1]
            yc = np.rint(y).astype(int, casting='unsafe')

            # Cut out subsection around the peak
            x_sub = np.arange(np.max((xc-rCut, 0)),
                              np.min((xc+rCut, Nx)) + 2, 1)
            y_sub = np.arange(np.max((yc-rCut, 0)),
                              np.min((yc+rCut, Ny)) + 2, 1)

            # Make indices of subsection
            x_cut = x_a[y_sub[0]: y_sub[-1], x_sub[0]: x_sub[-1]]
            y_cut = y_a[y_sub[0]: y_sub[-1], x_sub[0]: x_sub[-1]]
            cut = image[x_cut, y_cut]

            # Inital values for least-squares fitting
            k = np.min(cut)
            int_0 = np.max(cut)-k
            sigma = sigma0

            # Sub-pixel iterations
            for s0 in range(0, 3, 1):
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

            # Display fitting progress
            if p0 % 50 == 0:
                comp = (p0 + 1) / num_peaks
                print('Fitting is {:.2f} % complete!'.format(comp*100))

        # Reshape refined peak array
        peaks_refine = np.asarray(peaks_refine)
        peaks_refine = peaks_refine.reshape(num_peaks, 5)

        # Combine peaks from simple fitting with refined peak positions
        # Column meaning of peaks:
        # Col 1: x coordinate of original peak
        # Col 2: y coordinate of original peak
        # Col 3: peak intensity of original peak
        # Col 4: sigma (standard deviation) of Gaussian function
        # Col 5: refined x coordinate
        # Col 6: refined y coordinate
        # Col 7: background value determined by fit
        # Col 8: integrated peak intensity without background
        peaks = np.hstack((peaks, peaks_refine))

    else:
        messagebox.showinfo(
            "Information",
            "Lattice peaks are fitted without non-linear Gaussian functions!")

    return peaks


if __name__ == "__main__":
    matplotlib.use("TkAgg")
