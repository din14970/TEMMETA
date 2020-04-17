"""
Module for plotting shortcut functions

Functions
---------
plot_quick_spectrum
    Plot a roung spectrum from a 1D array of intensities
plot_spectrum_peaks
    Plot a spectrum as well as peaks calculated on this spectrum
plot_image
    Plot a 2D image and possibly adds scalebar
"""
# Base modules
# Basic 3rd party packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as wdgts
from matplotlib_scalebar.scalebar import ScaleBar, SI_LENGTH_RECIPROCAL
# from .guitools import file_dialog as fdo
from . import image_filters as imf


def plot_profile(profile, ax=None):
    """
    Plot an intensity profile

    Parameters
    ----------
    profile : data_io.Profile
        Profile object
    axis : matplotlib Axis object, optional
        If you want to plot on an existing axis instead of
        creating a new figure.

    Returns
    -------
    ax : matplotlib Axis object
    prof : the line plot itself
    """
    if ax is None:
        _, ax = plt.subplots()
    y = profile.data
    x = profile.x_axis
    ax.set_xlabel(f"x ({profile.pixelunit})")
    prof = ax.plot(x, y)
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(x.min(), x.max())
    return ax, prof


def plot_spectrum(spectrum, ax=None, show_peaks=False, w=5, **kwargs):
    """
    Plot an edx spectrum

    Parameters
    ----------
    spectrum : data_io.Spectrum
        Spectrum object
    axis : matplotlib Axis object, optional
        If you want to plot on an existing axis instead of
        creating a new figure.
    show_peaks : bool, optional
        Whether to also plot the calculated peaks if present
    w : float, optional
        Width of the bands for the peaks in number of channels

    Other parameters
    ----------------
    Kwargs : passed to line plot command

    Returns
    -------
    ax : matplotlib Axis object
    prof : the line plot itself
    """
    if ax is None:
        _, ax = plt.subplots()
    y = spectrum.data
    x = spectrum.energy_axis
    ww = w*spectrum.dispersion
    ax.set_xlabel(f"Energy ({spectrum.energy_unit})")
    prof = ax.plot(x, y, **kwargs)
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_xlim(x.min(), x.max())
    if show_peaks:
        if spectrum.peaks is not None:
            peaks = spectrum.peaks.iloc[:, 1]
            peak_heights = spectrum.peaks.iloc[:, 2]
            ax.scatter(peaks, peak_heights,
                       label="Peaks ({})".format(len(peaks)), color="C1")
            for i in peaks:
                ax.axvspan(i-ww/2, i+ww/2, alpha=0.2, color='red')
    return ax, prof


def plot_histogram(x, y, w, **kwargs):
    """
    Plots a histogram. Only equal sized bins are accepted.

    Parameters
    ----------
    x : array
        The centers of the bins
    y : array
        The values in those bins
    w : float
        plotting width of the bars

    Other parameters
    ----------------
    **kwargs : passed to pyplot.bar, check
    <https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.bar.html> for
    details

    Returns
    -------
    ax : pyplot.Axes object
    bars : pyplot.BarContainer object
    """
    _, ax = plt.subplots()
    bars = ax.bar(x, y, width=w, **kwargs)
    ax.set_xlabel("Pixel value (a.u.)")
    ax.set_ylabel("Pixel count")
    ax.set_ylim(0, None)
    return ax, bars


def plot_mask(mask, **kwargs):
    fig, ax = plt.subplots()
    im = ax.imshow(mask.data, cmap="Greys_r", **kwargs)
    fig.colorbar(im)
    return ax, im


def plot_fft(FFT, k_size=7, sigma=4., cs=20., crop_factor=3, **kwargs):
    '''
    Plot an FFT power spectrum in a nice way

    In order to plot, we plot the power spectrum after applying a gaussian
    filter and adjusting the intensity threshold (the central spot is always
    extremely intense)

    Parameters
    ----------
    FFT : data_io.FFT object
        the object representing the FFT
    k_size : int, optional
        The kernel size of the gaussian filter. Needs to
        be odd. Default is 7.
    sigma : float, optional
        Sigma of the Gaussian. Default is 4.0.
    cs: float, optional
        Mask radius for automatic intensity scaling. Default is 20.0.
    crop_factor: float, optional
        Field of view is limited to (size of the image)/(crop factor),
        centered around the middle. Default is 3, i.e. one third of the image
        is shown.

    Returns
    -------
    fft_blur : 2D numpy array
        The filtered Fourrier spectrum image. In fact it is a power spectrum.
    '''
    assert k_size % 2 == 1, "kernel size must be odd"
    # assert np.log2(image.shape[1]) % 1 == 0, "image must be 2^n sized"
    # assert np.log2(image.shape[0]) % 1 == 0, "image must be 2^n sized"
    fft2_db = FFT.power_spectrum.data  # power spectrum? use log10
    # Smooth PS with gaussian filter: k_size has to be odd!
    fft2_blur = imf.gauss_filter(fft2_db, k_size, sigma)
    # Make mask to boost contrast
    (Ny, Nx) = fft2_db.shape
    x = np.arange(-Nx//2, -Nx//2+Nx, 1)
    y = np.arange(-Ny//2, -Ny//2+Ny, 1)
    xv, yv = np.meshgrid(x, y)
    mask_radius = np.sqrt(np.square(xv) + np.square(yv))
    mask = mask_radius > cs
    fft2_masked = fft2_blur * mask  # cut out central spot
    fft2_masked_max = np.max(fft2_masked)  # find max for normalization
    fft2_blur = fft2_blur / fft2_masked_max
    # normalize but don't consider very intense central disk
    fft2_blur[fft2_blur > 1] = 1  # linear cut off
    ax, im = plot_array(fft2_blur, pixelsize=FFT.pixelsize,
                        pixelunit=FFT.pixelunit, **kwargs)
    # crop to only show central part of FFT
    crop_factor = crop_factor*2
    xbound = int(round(Nx/crop_factor))
    ybound = int(round(Ny/crop_factor))
    ax.set_xlim(Nx//2-xbound, Nx//2+xbound)
    ax.set_ylim(Ny//2+ybound, Ny//2-ybound)
    return ax, im


def plot_quick_spectrum(data, plot_channels=False, disp=1., offset=0):
    '''
    Plot a spectrum from a 1D array of intensities.

    Parameters
    ----------
    data : array-like (1D)
        spectrum intensities
    plot_channels : bool
        whether to use channel index or energy for x-axis
    disp : float
        the dispersion (energy (keV)/channel)
    offset : float
        spectrum offset in energy units (in keV)

    Returns
    -------
    fig : matplotlib.Figure
    ax : matplotlib.Axis
    '''
    fig, ax = plt.subplots(1)
    y = data
    x = np.arange(len(y))
    if plot_channels:
        ax.plot(x, y)
        ax.set_xlabel("Channel")
    else:
        ax.plot((x*disp+offset), y)
        ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts (a.u.)")
    plt.show()
    return fig, ax


def plot_spectrum_peaks(data, peaks, peak_heights, log=True, w=40,
                        disp=1., offset=0):
    '''
    Plot a spectrum with its peaks

    Parameters
    ----------
    data : array-like, 1D
        the spectrum, intensity per channel
    peaks : array-like, 1D
        the peak x-positions in channel index
    peak_heights : array-like, 1D
        the peak y-positions
    log : bool
        make y axis logarithmic. Default is True.
    w : int
        width of peaks in number of channels. Default is 40.
    disp : float
        dispersion, energy per channel (in keV)
    offset : float
        spectrum offset in energy units (in keV)

    Returns
    -------
    figure : matplotlib Figure object
    axis : matplotlib axis object
    '''
    fig, ax = plot_quick_spectrum(data, plot_channels=False, disp=disp,
                                  offset=offset)
    peaks = peaks*disp+offset
    w = w*disp
    if log:
        ax.set_yscale('log')
    ax.scatter(peaks, peak_heights,
               label="peaks ({})".format(len(peaks)), color="C1")
    for i in peaks:
        ax.axvspan(i-w/2, i+w/2, alpha=0.2, color='red')
    return fig, ax


def plot_image(img, **kwargs):
    """
    Wrapper for plot_array using a GeneralImage object directly
    """
    ax, im = plot_array(img.data, pixelsize=img.pixelsize,
                        pixelunit=img.pixelunit, **kwargs)
    return ax, im


def plotImagePeaks(image, peaks, ax=None):
    # Display the image (using matplotlib)
    # Create new figure window
    image = np.array(image)
    image = imf.normalize(image)
    if ax is None:
        _, ax = plt.subplots()
    fig = ax.figure
    im = ax.imshow(image, cmap='plasma')
    pks = ax.scatter(peaks[:, 1], peaks[:, 0], c='b', s=5)
    ax.axis('off')
    fig.colorbar(im, ax=ax)
    return ax, im, pks


def plot_fft_masked(fft, mask, window=0.3, **kwargs):
    """Plot a masked fft"""
    _, ax = plt.subplots()
    fftim = fft.power_spectrum.data
    dt = fftim*mask.data
    msked = ax.imshow(dt, **kwargs)
    d = fft.width*window
    ax.set_xlim(fft.width//2-d//2, fft.width//2+d//2)
    ax.set_ylim(fft.height//2+d//2, fft.height//2-d//2)
    return ax, msked


def plot_fft_and_mask(fft, mask, window=0.3):
    """Plot a masked fft"""
    _, ax = plt.subplots()
    fftim = fft.power_spectrum.data
    ps = ax.imshow(fftim, cmap="jet")
    msk = ax.imshow(mask.data, cmap="Greys_r", alpha=0.5)
    d = fft.width*window
    ax.set_xlim(fft.width//2-d//2, fft.width//2+d//2)
    ax.set_ylim(fft.height//2+d//2, fft.height//2-d//2)
    return ax, ps, msk


def plot_array(imgdata, pixelsize=1., pixelunit="", scale_bar=True,
               show_fig=True, width=15, dpi=None,
               sb_settings={"location": 'lower right',
                            "color": 'k',
                            "length_fraction": 0.15,
                            "font_properties": {"size": 12}},
               imshow_kwargs={"cmap": "Greys_r"}):
    '''
    Plot a 2D numpy array as an image.

    A scale-bar can be included.

    Parameters
    ----------
    imgdata : array-like, 2D
        the image frame
    pixelsize : float, optional
        the scale size of one pixel
    pixelunit : str, optional
        the unit in which pixelsize is expressed
    scale_bar : bool, optional
        whether to add a scale bar to the image. Defaults to True.
    show_fig : bool, optional
        whether to show the figure. Defaults to True.
    width : float, optional
        width (in cm) of the plot. Default is 15 cm
    dpi : int, optional
        alternative to width. dots-per-inch can give an indication of size
        if the image is printed. Overrides width.
    sb_settings : dict, optional
        key word args passed to the scale bar function. Defaults are:
        {"location":'lower right', "color" : 'k', "length_fraction" : 0.15,
         "font_properties": {"size": 40}}
        See: <https://pypi.org/project/matplotlib-scalebar/>
    imshow_kwargs : dict, optional
        optional formating arguments passed to the pyplot.imshow function.
        Defaults are: {"cmap": "Greys_r"}

    Returns
    -------
    ax : matplotlib Axis object
    im : the image plot object
    '''
    # initialize the figure and axes objects
    if not show_fig:
        plt.ioff()
    if dpi is not None:
        fig = plt.figure(frameon=False,
                         figsize=(imgdata.shape[1]/dpi, imgdata.shape[0]/dpi))
    else:
        # change cm units into inches
        width = width*0.3937008
        height = width/imgdata.shape[1]*imgdata.shape[0]
        fig = plt.figure(frameon=False,
                         figsize=(width, height))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot the figure on the axes
    im = ax.imshow(imgdata, **imshow_kwargs)

    if scale_bar:
        # get scale bar info from metadata
        px = pixelsize
        unit = pixelunit
        # check the units and adjust sb accordingly
        scalebar = get_scalebar(px, unit, sb_settings)
        plt.gca().add_artist(scalebar)
    if show_fig:
        plt.show()
    else:
        plt.close()
    return ax, im


def get_scalebar(px, unit, sb_settings):
    if '1/' in unit:
        # px = px*10**(-9)
        scalebar = ScaleBar(px, unit, SI_LENGTH_RECIPROCAL,
                            **sb_settings)
    else:
        scalebar = ScaleBar(px, unit, **sb_settings)
    return scalebar


def plot_image_interactive(imgobj):
    """
    Plot an image object in an interactive way

    Parameters
    ----------
    imgobj : data_io.GeneralImage
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25, left=-0.1)
    img = imgobj.data
    intmax = np.max(img)
    intmin = np.min(img)
    # img = imf.linscale(img, intmin, intmax)
    lim = plt.imshow(img, cmap="Greys_r")
    sb_settings = {"location": 'lower right',
                   "color": 'k',
                   "length_fraction": 0.25,
                   "font_properties": {"size": 12}}
    scalebar = get_scalebar(imgobj.pixelsize, imgobj.pixelunit, sb_settings)
    plt.gca().add_artist(scalebar)
    scalebar.set_visible(False)

    ax.margins(x=0)

    axcolor = 'white'
    # [x, y, width, height]
    axwidth = 0.7
    axheight = 0.03

    axmax = plt.axes([0.5-axwidth/2, 0.10, axwidth, axheight],
                     facecolor=axcolor)
    axmin = plt.axes([0.5-axwidth/2, 0.05, axwidth, axheight],
                     facecolor=axcolor)

    smax = wdgts.Slider(axmax, 'Max', intmin, intmax, valinit=intmax)
    smin = wdgts.Slider(axmin, 'Min', intmin, intmax, valinit=intmin)

    def update_frame(val):
        mx = smax.val
        mn = smin.val
        # arr = imgobj.data
        # arr = imf.linscale(arr, mn, mx)
        lim.set_clim(mn, mx)
        fig.canvas.draw_idle()

    smax.on_changed(update_frame)
    axmax._slider = axmax
    smin.on_changed(update_frame)
    axmin._slider = axmin

    # checkbox for scalebar
    sba = plt.axes([0.65, 0.2, 0.25, 0.25], facecolor=axcolor)
    sba.axis("off")
    sbb = wdgts.CheckButtons(sba, ('scalebar',))

    def seescalebar(label):
        yesorno = sbb.get_status()[0]
        scalebar.set_visible(yesorno)
        fig.canvas.draw_idle()

    sbb.on_clicked(seescalebar)
    sba._checkbox = sbb

    plt.show()


def plot_stack(stack):
    """
    Plot an image stack in an interactive way

    Parameters
    ----------
    stack : data_io.GeneralImageStack
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25, left=-0.1)

    frammax = stack.data.shape[0]-1
    intmax = np.max(stack.data)
    intmin = np.min(stack.data)

    img = stack.data[0]
    # img = imf.linscale(img, intmin, intmax)
    lim = plt.imshow(img, cmap="Greys_r")
    lim.set_clim(intmin, intmax)
    sb_settings = {"location": 'lower right',
                   "color": 'k',
                   "length_fraction": 0.25,
                   "font_properties": {"size": 12}}
    scalebar = get_scalebar(stack.pixelsize, stack.pixelunit, sb_settings)
    plt.gca().add_artist(scalebar)
    scalebar.set_visible(False)

    ax.margins(x=0)

    axcolor = 'white'
    # [x, y, width, height]
    axwidth = 0.7
    axheight = 0.03
    axfram = plt.axes([0.5-axwidth/2, 0.15, axwidth, axheight],
                      facecolor=axcolor)
    axmax = plt.axes([0.5-axwidth/2, 0.10, axwidth, axheight],
                     facecolor=axcolor)
    axmin = plt.axes([0.5-axwidth/2, 0.05, axwidth, axheight],
                     facecolor=axcolor)

    sfam = wdgts.Slider(axfram, 'Frame', 0, frammax, valinit=0, valstep=1,
                        valfmt="%d")
    smax = wdgts.Slider(axmax, 'Max', intmin, intmax, valinit=intmax)
    smin = wdgts.Slider(axmin, 'Min', intmin, intmax, valinit=intmin)

    def update_frame(val):
        mx = smax.val
        mn = smin.val
        fram = int(sfam.val)
        arr = stack.data[fram]
        # arr = imf.linscale(arr, mn, mx)
        lim.set_data(arr)
        lim.set_clim(mn, mx)
        fig.canvas.draw_idle()

    def update_clim(val):
        mx = smax.val
        mn = smin.val
        lim.set_clim(mn, mx)

    sfam.on_changed(update_frame)
    axfram._slider = sfam
    smax.on_changed(update_clim)
    axmax._slider = axmax
    smin.on_changed(update_clim)
    axmin._slider = axmin

    # checkbox for scalebar
    sba = plt.axes([0.65, 0.2, 0.25, 0.25], facecolor=axcolor)
    sba.axis("off")
    sbb = wdgts.CheckButtons(sba, ('scalebar',))

    def seescalebar(label):
        yesorno = sbb.get_status()[0]
        scalebar.set_visible(yesorno)
        fig.canvas.draw_idle()

    sbb.on_clicked(seescalebar)
    sba._checkbox = sbb

    # button for saving frame - does not work in jupyter notebook
    # def dosaveframe(event):
    #     fname = fdo.save()
    #     if fname:
    #         plt.savefig(fname)
    #         mx = smax.val
    #         mn = smin.val
    #         fram = int(sfam.val)
    #         arr = stack.data[fram]
    #         arr = imf.linscale(arr, mn, mx)
    #         plot_image(arr, pixelsize=stack.pixelsize,
    #                    pixelunit=stack.pixelunit,
    #                    scale_bar=sbb.get_status()[0],
    #                    show_fig=False, dpi=100,
    #                    sb_settings=sb_settings,
    #                    imshow_kwargs={"cmap": "Greys_r"})
    #         fig.canvas.draw_idle()

    # savea = plt.axes([0.65, 0.8, 0.15, 0.05], facecolor=axcolor)
    # saveb = wdgts.Button(savea, "save frame", hovercolor="yellow")
    # saveb.on_clicked(dosaveframe)
    # savea._button = saveb

    plt.show()


def plot_line_spectrum(specline):
    """
    Plot an interactive spectrum map
    """
    axwidth = 0.7
    axheight = 0.03

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25, left=0.5-axwidth/2)

    e0 = 0
    w0 = 0.5

    spec = specline.spectrum

    # maximum for sliders
    emin = specline._get_energy_of_channel(0)
    emax = specline._get_energy_of_channel(specline.channels)

    prof = specline.get_profile(e0, w0)
    # img = imf.linscale(img, intmin, intmax)
    _, lpf = prof.plot(axis=ax)
    lpf = lpf[0]

    axcolor = 'white'
    # [x, y, width, height]
    axenergy = plt.axes([0.5-axwidth/2, 0.10, axwidth, axheight],
                        facecolor=axcolor)
    axwidth = plt.axes([0.5-axwidth/2, 0.05, axwidth, axheight],
                       facecolor=axcolor)

    sen = wdgts.Slider(axenergy, f'Energy ({specline.energy_unit})', emin,
                       emax, valinit=e0)
    swi = wdgts.Slider(axwidth, f'Width ({specline.energy_unit})', 0,
                       (emax-emin)/2, valinit=w0)
    axenergy._slider = sen
    axwidth._slider = swi

    def update_im(val):
        en = sen.val
        wi = swi.val
        arr = specline._integrate_to_line(en, wi)
        lpf.set_ydata(arr)
        fig.canvas.draw_idle()

    sen.on_changed(update_im)
    swi.on_changed(update_im)

    # plot the spectrum on the energy axis
    _, lne = spec.plot(ax=axenergy)
    axenergy.set_ylim(0, spec.data.max()/3)
    axenergy.set_xlabel("")
    axenergy.set_ylabel("")
    lne[0].set_color("red")

    plt.show()


def plot_spectrum_map(specmap):
    """
    Plot an interactive spectrum map
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25, left=-0.1)
    ax.set_aspect("equal")

    e0 = 0
    w0 = 0.5

    spec = specmap.spectrum

    # maximum for sliders
    emin = specmap._get_energy_of_channel(0)
    emax = specmap._get_energy_of_channel(specmap.channels)
    # maximum and minimum for specmap color map
    intmax = np.max(specmap.data.sum(axis=0))
    intmin = 0

    img = specmap._get_integrated_image(e0, w0)
    # img = imf.linscale(img, intmin, intmax)
    lim = plt.imshow(img, cmap="hot")
    lim.set_clim(intmin, intmax)
    sb_settings = {"location": 'lower right',
                   "color": 'k',
                   "length_fraction": 0.25,
                   "font_properties": {"size": 12}}
    scalebar = get_scalebar(specmap.pixelsize, specmap.pixelunit, sb_settings)
    plt.gca().add_artist(scalebar)
    scalebar.set_visible(False)

    ax.margins(x=0)

    axcolor = 'white'
    # [x, y, width, height]
    axwidth = 0.7
    axheight = 0.03
    axmax = plt.axes([0.5-axwidth/2, 0.15, axwidth, axheight],
                     facecolor=axcolor)
    axenergy = plt.axes([0.5-axwidth/2, 0.10, axwidth, axheight],
                        facecolor=axcolor)
    axwidth = plt.axes([0.5-axwidth/2, 0.05, axwidth, axheight],
                       facecolor=axcolor)

    smx = wdgts.Slider(axmax, 'Max', intmin, intmax, valinit=intmax//3,
                       valstep=1, valfmt="%d")
    sen = wdgts.Slider(axenergy, f'Energy ({specmap.energy_unit})', emin, emax,
                       valinit=e0)
    swi = wdgts.Slider(axwidth, f'Width ({specmap.energy_unit})', 0,
                       (emax-emin)/2, valinit=w0)

    # plot the spectrum on the energy axis
    _, lne = spec.plot(ax=axenergy)
    axenergy.set_ylim(0, spec.data.max()/3)
    axenergy.set_xlabel("")
    axenergy.set_ylabel("")
    lne[0].set_color("red")

    def update_im(val):
        en = sen.val
        wi = swi.val
        mx = smx.val
        mn = 0
        arr = specmap._get_integrated_image(en, wi)
        # arr = imf.linscale(arr, mn, mx)
        lim.set_data(arr)
        lim.set_clim(mn, mx)
        fig.canvas.draw_idle()

    def update_clim(val):
        mx = smx.val
        mn = 0
        lim.set_clim(mn, mx)

    smx.on_changed(update_clim)
    axmax._slider = smx
    sen.on_changed(update_im)
    axenergy._slider = sen
    swi.on_changed(update_im)
    axwidth._slider = swi

    # checkbox for scalebar
    sba = plt.axes([0.65, 0.2, 0.25, 0.25], facecolor=axcolor)
    sba.axis("off")
    sbb = wdgts.CheckButtons(sba, ('scalebar',))

    def seescalebar(label):
        yesorno = sbb.get_status()[0]
        scalebar.set_visible(yesorno)
        fig.canvas.draw_idle()

    sbb.on_clicked(seescalebar)
    sba._checkbox = sbb

    plt.show()
