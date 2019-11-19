#! python3

#Base modules
import sys
import os
#Basic 3rd party packages
import h5py
import numpy as np
import json
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, vstack, spmatrix
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar, SI_LENGTH_RECIPROCAL

def plot_quick_spectrum(data: np.ndarray, plot_channels: bool = False, disp: float = 1.):
    '''Plot a spectrum quickly from a 1D array of intensities. Must provide the dispersion.'''
    fig, ax = plt.subplots(1)
    y =data
    #data.tocsr().sum(axis = 0).getA1()
    x = np.arange(len(y))
    if plot_channels:
        ax.plot(x, y)
        ax.set_xlabel("Channel")
    else:
        ax.plot(x*disp/1000, y)
        ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts (a.u.)")
    plt.show()
    return fig, ax


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def plot_image(imgdata: np.ndarray, pixelsize: float = 1., pixelunit: str = "", scale_bar: bool = True,
               show_fig: bool = False, dpi: int = 100,
               sb_settings: dict = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15},
               imshow_kwargs: dict = {}):
    '''
    Plot a 2D numpy array.

    Args:
    imgdata (np.ndarray) : the image frame
    pixelsize (float) : the scale size of one pixel
    pixelunit (str) : the unit in which pixelsize is expressed
    scale_bar (bool) = True : whether to add a scale bar to the image. Metadata must contain this information.
    show_fig (bool) = False : whether to show the figure
    dpi (int) = 100 : dpi to save the image with
    sb_settings (dict) = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15}: settings for the scale bar
    imshow_kwargs (dict) : optional formating arguments passed to the pyplot.imshow function
    '''
    #initialize the figure and axes objects
    if not show_fig:
        plt.ioff()

    fig = plt.figure(frameon=False, figsize = (imgdata.shape[0]/dpi, imgdata.shape[1]/dpi))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #plot the figure on the axes
    s = ax.imshow(imgdata, **imshow_kwargs)

    if scale_bar:
        #get scale bar info from metadata
        px=pixelsize
        unit=pixelunit
        #check the units and adjust sb accordingly
        if unit=='1/m':
            px=px*10**(-9)
            scalebar = ScaleBar(px, '1/nm', SI_LENGTH_RECIPROCAL, **sb_settings)
        else:
            scalebar = ScaleBar(px, unit, **sb_settings)
        plt.gca().add_artist(scalebar)

    if show_fig:
        plt.show()
    else:
        plt.close()

    return fig, ax
