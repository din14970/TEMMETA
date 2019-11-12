#! python3

#Base modules
import sys
import os

#Basic 3rd party packages
import h5py
import numpy as np
import json

#GUI elements
import tkinter as Tk
from tkinter import filedialog

#For working with images
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar, SI_LENGTH_RECIPROCAL


def get_file_path_dialog(filetypes: tuple = (("emd files","*.emd"),("all files","*.*"))):
    """
    Open a file dialog and return the path string corresponding to the selected file.

    Args:
        filetypes (tuple): what files the dialog will accept. Default is .emd files.

    Returns:
        str: filename

    """

    root = Tk.Tk()
    root.withdraw()
    fname = filedialog.askopenfilename(initialdir = ".",title = "Choose your file",filetypes = filetypes)
    root.update()
    return fname


def open_emd_gui():
    """Open a file dialog to select an emd file and return as h5py._hl.files.File"""

    emd_filename = get_file_path_dialog(filetypes = (("emd files","*.emd"),))
    try:
        f = h5py.File(emd_filename, 'r')
    except: #pressed cancel
        f = None
    return f


def scan_hdf5_node(hdf5_node, recursive: bool = True, full_path: bool = False,
                   see_info: bool = True, tab_step: int = 4):
    """
    Print the structure of an HDF5 node (can be root).

    Args:
        hdf5_node : The node to be investigated
        recursive (bool): if true it traverses all subgroups. If not it only prints top level of the current node.
        full_path (bool): print the entire group path of groups and datasets or only its name
        see_info (bool): print the array size and datatype next to datasets.
        tab_step (int): how many spaces to indent each level
    """
    #separate the name of the node from the rest of the path in the file
    def get_name(x):
        if not full_path:
            name = x.name.split("/")[-1]
        else:
            name = x.name
        return name
    #traverse the node. Find all datasets and groups in the file. If recersion is on go deep.
    def scan_node(g, tabs=0):
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                name_ds = get_name(v)
                extra=""
                if see_info:
                    extra = " ("+str(v.shape)+", "+str(v.dtype)+")"
                print(' ' * tabs + ' ' * tab_step + ' -', name_ds, extra)

            elif isinstance(v, h5py.Group):
                name_gr = get_name(v)
                print(' ' * tabs + ' ' * tab_step + name_gr)
                if recursive:
                    scan_node(v, tabs=tabs + tab_step)
    #Get the name of the node. If root then name will be / which we must print separately.
    root=get_name(hdf5_node)
    if root == "":
        print("/")
    else:
        print(root)
    scan_node(hdf5_node)


def get_det_uuid(f: h5py._hl.files.File, sig: str, det_no: int):
    """
    Get the UUID key for the detector signal based on a detector index number

    Args:
        f (h5py._hl.files.File): the HDF5 file opened with h5py.File
        sig (str):     Provide signal type (Data subgroup) as a string
                       ["Image", "Line", "Spectrum", "SpectrumImage", "Spectrumstream"]
        det_no (int):  The detector number, since they have UUID names. If not provided, turned into 0.

    Returns:
        str: the uuid that can be used as a key to access data and metadate
    """
    det = None

    #check the input arg types
    if not isinstance(f, h5py._hl.files.File):
        raise ValueError("Invalid type for file argument")

    if not isinstance(sig, str):
        raise ValueError("Invalid type for signal argument")

    if not isinstance(det_no, int):
        raise ValueError("Invalid type for detector number")

    try:
        #get access to a particular subgroup sig in the Data group
        imdat = f["Data"][sig]
        #get a list of the UUID named groups in this group
        dets = list(imdat.keys())
        #get the datagroup corresponding to the detector number
        det = dets[det_no]

    except KeyError as kyrr:
        #the signal key is invalid
        print("The \'Data/{}\' signal is not present in this file".format(sig))

    except IndexError as ixerr:
        #invalid detector number
        print("No detector group associated with key '{}' in the \'{}\' signal "
              "[valid keys: {}]".format(det_no, sig, list(range(len(dets)))))

    return det


def get_meta_dict_unsafe(f, sig, det, frame):
    """
    General importing function of EMD metadata.

    Args:
        f (h5py._hl.files.File): the HDF5 file opened with h5py.File
        sig (str):     Provide signal type (Data subgroup) as a string
                       ["Image", "Line", "Spectrum", "SpectrumImage", "Spectrumstream"]
        det (str):  The detector UUID str.
        frame (int):   The column (integer) corresponding to the frame

    Returns:
        dict
    """

    #get access to a particular subgroup sig in the Data group
    imdat = f["Data"][sig]
    #get the metadata in this group
    meta = imdat[det]['Metadata']
    #turn the metadata array into a list and trim the zeros
    meta_array = list(np.trim_zeros(meta[:,frame]))
    #turn the list of ints into chars
    meta_char = list(map(lambda x: chr(x), meta_array))
    #concatenate the chars into a string
    meta_json = "".join(meta_char)
    #interpret the string with json to create a dictionary
    meta_dict = json.loads(meta_json)
    #return the dictionary
    return meta_dict


def get_meta_dict(f: h5py._hl.files.File, sig: str, det: str, frame: int = 0) -> dict:
    """
    General importing function of EMD metadata.

    Args:
        f (h5py._hl.files.File): the HDF5 file opened with h5py.File
        sig (str):     Provide signal type (Data subgroup) as a string
                       ["Image", "Line", "Spectrum", "SpectrumImage", "Spectrumstream"]
        det (str):     The detector UUID str.
        frame (int):   The column (integer) corresponding to the frame

    Returns:
        dict
    """

    #check the input arg types
    if not isinstance(f, h5py._hl.files.File):
        raise ValueError("Invalid type for file argument")

    if not isinstance(sig, str):
        raise ValueError("Invalid type for signal argument")

    if not isinstance(det, str):
        raise ValueError("Invalid type for detector uuid")

    meta_dict = None

    try:
        #Read the particular image
        imdat = f["Data"][sig]

    except KeyError as kyrr:
        #the signal key is invalid
        print("The \'Data/{}\' signal is not present in this file".format(sig))

    else:
        #No exceptions on file and signal key
        try:
            #does the detector key yield a group?
            assert (isinstance(imdat[det], h5py._hl.group.Group)), ("{} is not "
                "a valid data group in the \'{}\' signal".format(det, sig))
            meta = imdat[det]['Metadata']
            meta_array = list(np.trim_zeros(meta[:,frame]))

        except KeyError as kyrr:
            #no metadata key found
            print("There is no metadata in the {} detector for the \'{}\' signal".format(det, sig))

        except AssertionError as assrt:
            #invalid detector group
            print(assrt)

        else:
            #map the list of characters to a string and interpret it with json. return the dictionary.
            meta_char = list(map(lambda x: chr(x), meta_array))
            meta_json = "".join(meta_char)
            meta_dict = json.loads(meta_json)

    finally:
        #none if we ended in exception, otherwise a dictionary
        return meta_dict


def get_meta_dict_det_no(f: h5py._hl.files.File, sig: str, det_no: int=0, frame: int = 0):
    """
    Wrapper for importing EMD metadata from the a subgroup. The typ subgroup must be specified
    as an element of ["Image", "Line", "Spectrum", "SpectrumImage", "Spectrumstream"]

    Args:
        f (h5py._hl.files.File): the HDF5 file opened with h5py.File
        sig (str):     Provide signal type (Data subgroup) as a string
                       ["Image", "Line", "Spectrum", "SpectrumImage", "Spectrumstream"]
        det_no (int):  The detector number, which is translated to a UUID str. If not provided, turned into 0.
        frame (int):   The column (integer) corresponding to the frame

    Returns:
        dict
    """
    det = get_det_uuid(f, sig, det_no)
    return get_meta_dict(f, sig, det, frame)


def get_pretty_meta_str(meta_dict: dict)->str:
    """
    Get a nicely formatted and indented metadata json string

    Args:
        met_dict (dict): the metadata dictionary

    Returns:
        str
    """
    return json.dumps(meta_dict, indent=4, sort_keys=True)


def print_pretty(meta_dict: dict):
    """
    Print a nicely formatted and indented metadata json string

    Args:
        met_dict (dict): the metadata dictionary
    """
    print(get_pretty_meta_str(meta_dict))


def write_meta_json(filename: str, meta_dict: dict):
    """
    Write metadata out as a json file

    Args:
        filename (str): name of the file to which you will write
        met_dict (dict): the metadata dictionary
    """
    with open(filename, "w") as f:
        f.write(get_pretty_meta_str(meta_dict))


def read_meta_json(filename: str):
    """
    Read metadata from a json string in a file and return a dictionary

    Args:
        filename (str): name of the file from which you will read

    Returns
        met_dict (dict): the metadata dictionary
    """
    with open(filename, "r") as f:
        metadata = json.load(f)
    return metadata


def get_data(f: h5py._hl.files.File, sig: str, det: str):
    """
    Returns EMD data using a signal string and detector uuid

    Args:
        f (h5py._hl.files.File): the HDF5 file opened with h5py.File
        sig (str):     Provide signal type (Data subgroup) as a string
                       ["Image", "Line", "Spectrum", "SpectrumImage", "Spectrumstream"]
        det (int):  The detector index, which is translated to a UUID str.

    Returns:
        h5py._hl.dataset.Dataset
    """
    data=f['Data'][sig][det]['Data']
    return data


def get_image_data_det_no(f: h5py._hl.files.File, det_no: int):
    """
    Wrapper for importing EMD image data using a detector index.

    Args:
        f (h5py._hl.files.File): the HDF5 file opened with h5py.File
        det_no (int):  The detector index, which is translated to a UUID str.

    Returns:
        h5py._hl.dataset.Dataset
    """
    sig = "Image"
    det = get_det_uuid(f, sig, det_no)
    return get_data(f, sig, det)


def save_single_image(imgdata: np.ndarray, filename:str , metadata: dict,
               scale_bar: bool = True, show_fig: bool = False, dpi: int = 100, save_meta: bool = True,
               sb_settings: dict = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15}, imshow_kwargs: dict = {}):
    '''
    Saves a single NxN numpy array representing an image to a TIFF file.

    Args:
    imgdata (np.ndarray) : the image frame
    filename (str) : the filename to which you want to save the image
    metadata (dict) : the metadata dictionary corresponding to the file
    scale_bar (bool) = True : whether to add a scale bar to the image. Metadata must contain this information.
    show_fig (bool) = False : whether to show the figure
    dpi (int) = 100 : dpi to save the image with
    save_meta (bool) = True : save the metadata also as a separate json file with the same filename as the image
    sb_settings (dict) = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15}: settings for the scale bar
    imshow_kwargs (dict) : optional formating arguments passed to the pyplot.imshow function
    '''
    #initialize the figure and axes objects
    fig = plt.figure(frameon=False, figsize = (imgdata.shape[0]/dpi, imgdata.shape[1]/dpi))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    #plot the figure on the axes
    s = ax.imshow(imdat, **imshow_kwargs)

    if scale_bar:
        #get scale bar info from metadata
        px=(float(metadata["BinaryResult"]["PixelSize"]["width"]))
        unit=metadata["BinaryResult"]["PixelUnitX"]
        #check the units and adjust sb accordingly
        if unit=='1/m':
            px=px*10**(-9)
            scalebar = ScaleBar(px, '1/nm', SI_LENGTH_RECIPROCAL, **sb_settings)
        else:
            scalebar = ScaleBar(px, unit, **sb_settings)
        plt.gca().add_artist(scalebar)
    #save the figure
    plt.savefig(filename, dpi = dpi)

    if show_fig:
        plt.show()
    else:
        plt.close()

    if save_meta:
        #if metadata save the metadata to a json file with the same name
        path, ext = os.path.splitext(filename)
        write_meta_json(path+".json", metadata)
