#! python3

#Base modules
import sys
import os
import logging
import concurrent.futures as cf #parrallel tasks

#Basic 3rd party packages
import h5py
import numpy as np
import json
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, vstack, spmatrix
from scipy.sparse import save_npz, load_npz
#For working with images
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar, SI_LENGTH_RECIPROCAL
#GUI elements
import tkinter as Tk
from tkinter import filedialog
#my own modules
#sys.path.append(".")
from . import plottingtools as pl
from .decorators import timeit

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


def read_emd(filepath: str):
    """Open an emd file with a given filepath"""
    try:
        return h5py.File(filepath, 'r')
    except:
        return None

def open_emd_gui():
    """Open a file dialog to select an emd file and return as h5py._hl.files.File"""

    emd_filename = get_file_path_dialog(filetypes = (("emd files","*.emd"),))
    try:
        f = h5py.File(emd_filename, 'r')
    except: #pressed cancel
        f = None
    return f

def scan_node(g, tabs=0, recursive: bool = True,
                   see_info: bool = True, tab_step: int = 4):
    """Traverse the node. Find all datasets and groups in the file. If recersion is on go deep"""
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


#separate the name of the node from the rest of the path in the file
def get_name(x, full_path: bool = False):
    if not full_path:
        name = x.name.split("/")[-1]
    else:
        name = x.name
    return name


def get_emd_tree_view(f):
    """Flattens out the structure of data inside an EMD file"""
    tree = {}
    tree[(0,0)] = ("{}".format(get_name(f)), "EMD")
    counter = 0
    for k, v in f["Data"].items():
        vn = get_name(v)
        for i, j in v.items():
            tree[(1, counter)] = (get_name(j), vn)
            counter += 1
    return tree


def scan_hdf5_node(hdf5_node, full_path: bool = False, **kwargs):
    """
    Print the structure of an HDF5 node (can be root).

    Args:
        hdf5_node : The node to be investigated
        recursive (bool): if true it traverses all subgroups. If not it only prints top level of the current node.
        full_path (bool): print the entire group path of groups and datasets or only its name
        see_info (bool): print the array size and datatype next to datasets.
        tab_step (int): how many spaces to indent each level
    """
    #
    #Get the name of the node. If root then name will be / which we must print separately.
    root=get_name(hdf5_node, full_path)
    if root == "":
        print("/")
    else:
        print(root)
    scan_node(hdf5_node, **kwargs)


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



def get_detector_property(meta: dict, prop: str, exact_match: bool = False):
    '''
    Return the value of a property stored in the detector metadata.
    A detector has a certain name which is stored in the "BinaryResult" dict.
    However, the properties of these detectors are stored in Detectors/Detector-n
    So we loop over these detectors to search for a match with the name, then
    return the desired property.

    Args:
        meta (dict):        Metadata dictionary. Has a reference to the right
                            detector.
        prop (str):         Key to the desired property name
        exact_match (bool): Some detector names may not match exactly between
                            the one in BinaryResult and Detectors. For example
                            the detector for EDX may be SuperX but in Detectors
                            there exists SuperX1, SuperX2, ... Default is False
                            to include fuzzy match.
    '''
    #The detector name to search for
    det_name = meta["BinaryResult"]["Detector"]
    #loop over all the detectors
    for i in meta["Detectors"].keys():
        det_dic = meta["Detectors"][i]
        if exact_match:
            if det_name == det_dic["DetectorName"]:
                return det_dic[prop]
        else:
            if det_name in det_dic["DetectorName"]:
                return det_dic[prop]


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


def get_spectrum_stream_acqset(f: h5py._hl.files.File, det: str):
    '''Read the AcquisitionSettings document in the SpectrumStream data and return as dict'''
    s = f["Data/SpectrumStream"][det]["AcquisitionSettings"][0]
    return json.loads(s)


def get_spectrum_stream_flut(f: h5py._hl.files.File, det: str):
    '''Read the FrameLocationTable document in the SpectrumStream data and return as array'''
    s = f["Data/SpectrumStream"][det]["FrameLocationTable"][:,0]
    return s


def guess_multiframe_dataset(f: h5py._hl.files.File):
    '''
    Guess the detector id of the image dataset that corresponds to the spectrumstream
    Returns the detector number, detector uuid and the number of frames
    '''
    #find the detector number of the dataset containing many frames
    for k, item in enumerate(f["Data"]["Image"].items()):
        i, j = item
        img = j["Data"]
        #if the last dimension is more than 1, we guess this is the good dataset
        if img.shape[-1]>1:
            det_no = k
            det_uuid = i
            num_frames = img.shape[-1]
            break
    else: #no break statement found
        logging.error("No image dataset found with more than 1 frame.")
        return
    return det_no, det_uuid, num_frames


def convert_stream_to_sparse(d1d: np.ndarray, dim: tuple, dv: int = 65535, compress_type: str = 'dok'):
    '''
    Converts a given numpy array formatted like the emd SpectrumStream data into a sparse matrix.
    The rows of this matrix include all the scan positions, the columns represent the number of channels.

    Args:
        d1d (numpy.ndarray):    The array to be formatted
        dim (tuple):            The (x, y, channel, frame) dimension of the hypothetical returned csc_matrix.
                                frame can be none, then it is calculated based on the size of d1d.
        dv (int):               The value in the SpectrumStream that encodes "next pixel"
                                Default = 65535
        compress_type (str):    Type of compression that should be used to store the return data
                                memory. See Scipy.sparse for details.
                                Options: ['dok'(Default), 'none', 'csc', 'csr']

    Returns:
        A Scipy.sparse.spmatrix object or a np.ndarray if compress_type = 'none'
    '''

    #unpack the dimensions
    xs, ys, ch, frm = dim
    #find the indexes where counts are registered (!= the counting number)
    cinx = np.argwhere(d1d!=dv)[:,0]
    #calc the pixel index to which these counts must be mapped
    pixind = cinx - np.arange(len(cinx)) - 1
    #calculate the number of frames based on the size if frm is unknown
    if frm is None:
        pxlcount = len(d1d) - len(cinx)
        frm = pxlcount // xs //ys
    #calc the dimensions of the sparse matrix
    dim_sparse = (xs*ys*frm, ch)
    #Initialize an array with the dimensions: total size * the number of channels
    temp = dok_matrix(dim_sparse, dtype=np.int16)
    #loop over the list of counts and put them in the right bin
    for i, j in zip(cinx, pixind):
        if j<xs*ys*frm: #throw away non complete frames
            chan = d1d[i] #the channel number = the value stored at the index
            temp[j, chan] += 1 #increment the right entry


    #return the right type depending on chosen compression
    compress_type = compress_type.lower()
    if compress_type == 'none':
        return temp.toarray()
    elif compress_type == 'dok':
        return temp
    elif compress_type == 'csc':
        return temp.tocsc()
    elif compress_type == 'csr':
        return temp.tocsr
    else:
        raise ValueError("Not recognized compression type, should be none, dok, csc or csr")


def get_frame_limits(frm: int, flut: np.ndarray):
    '''
    Get the first index ix1 of a SpectrumStream frame and the first index ix2 of the next frame.
    In this way SpectrumStreamData[ix1:ix2] can be efficiently queried.
    '''
    assert isinstance(frm, int), "Must provide valid frame index"
    ix1 = flut[frm] #we will get index error if
    try:
        ix2 = flut[frm+1]
    except IndexError: #the last index
        ix2 = None
    return ix1, ix2


def get_frame_indexes(frm: int, flut: np.ndarray, totln: int = None):
    '''Get all indexes from one SpectrumStram frame'''
    assert isinstance(frm, int), "Must provide valid frame index"
    ix1, ix2 = get_frame_limits(frm, flut)
    if ix2 is None: #in the last frame, the lookuptable doesn't know the final index. must be provided
        assert isinstance(totln, int), "For the last frame, the total length of the stream must be provided"
        ix2 = totln
    return np.arange(ix1, ix2)


def get_frames_indexes(frms: list, flut: np.ndarray, totln: int = None):
    '''
    Get all indexes from multiple Spectrumstream frames. Performs a simple loop and
    performs get_frame_indexes() and adds the arrays together.
    !Note: querying data from SpectrumStream d is much faster with d[ix1:ix2] than
    with d[list of indexes]
    '''
    inxs = np.array([])
    for i in frms:
        inxs = np.append(inxs, get_frame_indexes(i, flut, totln = totln))
    return inxs.astype(int)


def translate_stream_frame(d: h5py._hl.dataset.Dataset, flut: np.ndarray,
                           xs: int, ys: int, cs: int, frm: int,
                           dv: int = 65535, compress_type: str = 'dok'):
    '''
    Return a 2D array or compressed matrix representation of a single frame of a spectrum stream.
    The rows represent a pixel index, the colums represent the channel.
    The values stored represent the counts. For EDX data, only a few thousand counts are registered
    per frame, so the data is very sparse.

    Args:
        d (h5py._hl.dataset.Dataset) : a SpectrumStream dataset read from an emd file
        flut (numpy.ndarray) : a frame lookuptable also read from the emd file
        xs (int): size of the scanning grid in the x-direciton
        ys (int): size of the scanning grid in the y-direction
        cs (int): number of channels
        frm (int): frame number
        dv (int): the number in the data that should be interpreted as a counter. Default = 65535
        compress_type (str): the type of compression Options: ['dok'(Default), 'none', 'csc', 'csr']

    Returns:
        numpy.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix or scipy.sparse.dok_matrix
    '''

    ix1, ix2 = get_frame_limits(frm, flut)
    #query the frame from the long spectrumstream
    d1d = d[ix1:ix2].flatten()
    temp = convert_stream_to_sparse(d1d, (xs, ys, cs, None), dv = dv, compress_type = compress_type)
    return temp


@timeit
def get_spectrum_stream(f: h5py._hl.files.File, det_no: int = 0,
                               frames: list = [], one_matrix = True, compress_type = "dok", re_all: bool = False):
    '''
    Converts the spectrum stream data in .emd files to either one large sparse matrix or a list of sparse matrices,
    one per frame. The first case represents the second case with all frames stacked on top of eachother.

    Args:

        f (h5py._hl.files.File): HDF5 file imported into python with h5py
        det_no (int):            The index for the measurement you want to import if there are
                                 multiple streams in the file. Default = 0.
        frames (list):           the list with indexes of frames you want to import. An empty
                                 list defaults to all frames in the stream.
        compress_type (bool):    Type of compression that should be used to store the return data
                                 memory. See Scipy.sparse for details.
                                 Options: ['dok'(Default), 'none', 'csc', 'csr']
        re_all (bool):           Whether to return only the spectrumstream or additional info

    Returns:
        A SpectrumStream object that contains all the information

    '''
    sig = "SpectrumStream"
    #get the uuid of the dataset based on the index det_no
    det = get_det_uuid(f, sig, det_no)
    #get the data
    d = get_data(f, sig, det)
    #get the corresponding metadata
    #Do this with dictionaries instead
    md1 = get_meta_dict(f, sig, det, frame = 0)
    md2 = get_meta_dict(f, sig, det, frame = 1)
    #get the acquisition parameters
    acq = get_spectrum_stream_acqset(f, det)
    #get the frame table. There is not always one!!! If there is none then it's None
    try:
        flut = get_spectrum_stream_flut(f, det)
    except:
        flut = None
    #get a few commonly used vars
    chan = int(acq["bincount"]) #number of channels

    xs = int(acq['RasterScanDefinition']['Width'])
    ys = int(acq['RasterScanDefinition']['Height'])

    #if not frames: #no frame indexes provided (default) then do the whole array
    d1d = d[:].flatten()
    if flut is not None:
        frame_dimension = len(flut)
    else:
        frame_dimension = None

    if frames:
        if flut is not None: #were indexes of frames provided? is there a lookup table
            frames = np.sort(frames).tolist() #indexing only accepts ascending order
            d1d = np.array([])
            for i in frames:
                ix1, ix2 = get_frame_limits(i, flut)
                d1d = np.append(d1d, d[ix1:ix2])
            #inxs = get_frames_indexes(frames, flut, totln = d.len())
            #d1d = d[:,0][inxs] #this is very slow, better to add incrementally to it
            frame_dimension = len(frames)
        else:
            print("No frame lookup table found, this may take a while...")

    #return one sparse matrix containing the entire stream
    specstr = convert_stream_to_sparse(d1d, (xs, ys, chan, frame_dimension), compress_type = compress_type)

    if flut is None:
        frame_dimension = specstr.shape[0]//xs //ys
        if frames:
            omd = specstr.tocsr()
            dls = [omd[i*xs*ys:(i+1)*xs*ys] for i in frames]
            specstr = vstack(dls)
            frame_dimension = specstr.shape[0]//xs//ys
    #
    # #return a list of sparse matrices, each one for a frame
    # else:
    #     if frames: #there are elements in frames
    #         loopover = frames
    #     else: #there are no elements in frames, loop over all
    #         loopover = range(len(flut))
    #
    #     specstr = []
    #
    #     with cf.ThreadPoolExecutor() as executor: #perform with threading
    #         #create the threads list of translate stream for all frames in loopover
    #         results = [executor.submit(translate_stream_frame, d, flut, xs, ys,
    #                                    cs = chan, frm = i, compress_type = compress_type) for i in loopover]
    #         #When completed, add the output from translate stream to the list
    #         for f in cf.as_completed(results):
    #             specstr.append(f.result())

        #without threading
        #for i in loopover:
            #frmmat = translate_stream_frame(d, flut, xs, ys, cs = chan, frm = i, compress_type = "csc")
            #specstr.append(frmmat)
    disp = float(get_detector_property(md1, "Dispersion"))
    specstr_obj = SpectrumStream(specstr, (md1, md2), xs, ys, frames, chan, disp) #for if we turn the returned thing into some custom class
    if not one_matrix:
        specstr_obj.get_frame_list(convert = True)

    if re_all:
        return specstr_obj, d, md1, md2, flut, frames
    else:
        return specstr_obj


class SpectrumStream(object):
    '''
    Abstraction of SpectrumStream data as an object
    '''
    def __init__(self, data, metadata, xs, ys, frms, cs, disp):
        '''Set all the parameters'''
        self._data = data
        self._metadata = metadata
        if isinstance(self.data, list):
            self._om = False
        else:
            self._om = True
        self.xs = xs
        self.ys = ys
        self.fs = self.xs*self.ys
        self.frames = frms #list of frame indexes
        if frms:
            self.num_frames = len(frms)
        else:
            self.num_frames = int(self.data.shape[0]/self.xs/self.ys)
        self.cs = cs #number of channels
        self.disp = disp #number of eV per channel

        self.px=(float(self.metadata["BinaryResult"]["PixelSize"]["width"]))#x size
        self.py=(float(self.metadata["BinaryResult"]["PixelSize"]["height"]))#x size
        self.units = self.metadata["BinaryResult"]["PixelUnitX"]

        self.tot_spectrum = self.get_own_spectrum_sum()
        self.tot_stemspec_sparse = self.get_frame_sum()
        self.tot_stemspec = self.reshape_own_sparse_matrix(self.tot_stemspec_sparse)


    @property
    def scan_dimensions(self):
        return (self.xs, self.ys)


    @property
    def scan_size(self):
        return (self.xs*self.px, self.ys*self.py)


    @property
    def framecount(self):
        return self.num_frames


    @property
    def channelcount(self):
        return self.cs


    @property
    def is_one_matrix(self):
        return self._om


    @property
    def is_list(self):
        return not self._om


    @property
    def metadata(self):
        try:
            metadata = self._metadata[0]
        except:
            metadata = self._metadata
        return metadata


    @property
    def all_metadata(self):
        return self._metadata


    def get_frame_list(self, convert = False, compress_type = "csr"):
        '''
        Returns the stored data as a list of frames, each frames a sparse matrix.
        If convert = True then the way the data is stored in the instance is changed to list.
        '''
        if self.is_one_matrix:
            dt = self.data.tocsr() #we will do row slicing so it's faster
            toreturn = [change_compress(dt[i*self.fs:(i+1)*self.fs,:], compress_type = compress_type)\
                        for i in range(self.num_frames)]
            if convert:
                self._data = toreturn
                self._om = False
            return toreturn
        else:
            return self.data


    def get_stack_frames(self, convert = False, compress_type = "csr"):
        '''
        Returns the stored data as one sparse matrix, each frame stacked as rows.
        If convert = True then the way the data is stored is to this large matrix.
        '''
        if self.is_one_matrix:
            return self.data
        else:
            tostack = [i.tocsr() for i in self.data] #row stacking, change to csr
            toreturn = vstack(tostack)
            if convert:
                self._data = toreturn
                self._om = False
            return change_compress(toreturn, compress_type = compress_type)


    def reshape_own_sparse_matrix(self, frmdat = None):
        '''Return an intuitive 3D matrix representing one frame that is sliced with indices
        (channel, x, y). If no data is provided, perform on sum of all frames.'''
        if frmdat is None:
            frmdat = self.get_frame_sum(compress_type = "csr")
        return frmdat.T.toarray().reshape(self.cs, self.xs, self.ys)

    @staticmethod
    def reshape_sparse_matrix(frmdat, dim):
        '''Return an intuitive 3D matrix representing one frame that is sliced with indices
        (channel, x, y)'''
        cs, xs, ys = dim
        return frmdat.T.toarray().reshape(cs, xs, ys)


    def save_matrix(self, filename):
        '''Save the full dataset to a file'''
        filename = filename + ".npz"
        data = self.get_stack_frames()
        save_npz(filename, data)


    @property
    def data(self):
        return self._data


    def get_frame_sum(self, comp_type: str = "csr"):
        '''Return a sparse matrix sum of all frames'''
        data = self.get_frame_list()
        temp = 0
        for i in data:
            temp += i
        return change_compress(temp, comp_type)


    def get_image_peak_sum(self, energy: float, width: float, channel_querry: int = 20):
        '''Get a 2D array (x,y) with total counts within a certain energy window'''
        start = int((energy-width/2)*1000/self.disp) #starting channel index
        end = int((energy+width/2)*1000/self.disp) #ending channel index

        if end-start>channel_querry:
            dt_csc = self.get_stack_frames().tocsc() #we intend to filter based on channels: columns
            dt_ar = dt_csc[:, start:end].sum(axis = 1).getA1()  #add all these columns and make into 1D numpy array
            dt_frms = np.reshape(dt_ar, (self.num_frames, self.ys, self.xs)) #reshape
        else:
            dt_frms = self.tot_stemspec[start:end] #quite inefficient for multiple large arrays
        return dt_frms.sum(axis = 0)


    @staticmethod
    def get_spectrum_sum(data: spmatrix):
        '''Return a sum spectrum of the sparse data supplied'''
        data_sm = data.tocsr()
        return data_sm.sum(axis = 0).getA1()


    def get_own_spectrum_sum(self):
        '''Return a sum spectrum of the data stored in the instance _data attribute'''
        data_mat = self.get_stack_frames()
        data_sm = data_mat.tocsr()
        return data_sm.sum(axis = 0).getA1()


    def plot_quick_spectrum(self, plot_channels: bool = False):
        '''Plot a quick total spectrum representation of the data'''
        #toplot = self.get_own_spectrum_sum()
        toplot = self.tot_spectrum
        fig, ax = pl.plot_quick_spectrum(toplot, plot_channels = plot_channels, disp = self.disp)
        return fig, ax


    def write_streamframes(self, path = "./SpectrumStream/", pre = "Frame",
                            counter = None):
        '''Writes out the spectrumstream frame by frame in the .npz format in a separate folder. Metadata is not written out.
        Naming convention is Frame(n).npz'''
        dt = self.get_frame_list()
        if not os.path.exists(path):
            os.makedirs(path)

        #set the number of digits in the counter
        mincounter = get_counter(self.num_frames)
        if counter is None:
            counter = mincounter
        elif counter<mincounter:
            logging.error("Insufficient digits to represent the frames")
            return
        else:
            pass #counter is already set to the right value

        for j,i in enumerate(dt):
            c = str(j).zfill(counter)
            name = "{}_{}".format(pre, c)
            save_npz(path+name, i)


    def plot_energy_image(self, energy: float, width: float, scale_bar: bool = True, show_fig: bool = True, dpi: int = 100,
    save_meta: bool = True, sb_settings: dict = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15},
    imshow_kwargs: dict = {}, savefile: str = ""):
        '''
        Plot and potentially save an image representing the sum of counts over all frames
        within a particular energy window.

        Args:
        energy (float) : the energy to plot in keV
        width (float) : the energy window to integrate over
        scale_bar (bool) = True : whether to add a scale bar to the image. Metadata must contain this information.
        show_fig (bool) = True : whether to show the figure
        dpi (int) = 100 : dpi to save the image with
        save_meta (bool) = True : save the metadata also as a separate json file with the same filename as the image
        sb_settings (dict) = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15}: settings for the scale bar
        imshow_kwargs (dict) : optional formating arguments passed to the pyplot.imshow function
        savefile (str) : file to save the image to. If empty (default) will not save the file.
        '''
        tp = self.get_image_peak_sum(energy, width)
        pixelsize, pixelunit = get_scale(self.metadata)
        fig, ax = pl.plot_image(tp, pixelsize, pixelunit, scale_bar = scale_bar,
                       show_fig = show_fig, dpi = dpi,
                       sb_settings = sb_settings,
                       imshow_kwargs = imshow_kwargs)
        if savefile:
            fig.savefig(savefile, dpi = dpi)

        if save_meta and savefile:
            #if metadata save the metadata to a json file with the same name
            path, ext = os.path.splitext(filename)
            write_meta_json(path+".json", self.metadata)

        return fig, ax


@timeit
def read_streamframes(path = "./", pre = "Frame"):
    '''Read in frames from .npz files'''
    streamdata = []
    for j, i in enumerate(os.listdir(path)):
        name = "{}{}.npz".format(pre, j)
        dt_sparse = load_npz(path+name)
        streamdata.append(dt_sparse)
    return streamdata


def change_compress(temp, compress_type = "none"):
    #return the right type depending on chosen compression
    if compress_type == 'none':
        return temp.toarray()
    elif compress_type == 'dok':
        return temp.todok()
    elif compress_type == 'csc':
        return temp.tocsc()
    elif compress_type == 'csr':
        return temp.tocsr()
    else:
        raise ValueError("Not recognized compression type, should be none, dok, csc or csr")


def get_scale(metadata):
    '''Searches metadata and returns the size of a pixel and the unit as  tuple'''
    pixelsize=(float(metadata["BinaryResult"]["PixelSize"]["width"]))
    pixelunit=metadata["BinaryResult"]["PixelUnitX"]
    return pixelsize, pixelunit

def get_counter(framenumber):
    """Calculate the minimum number of digits necessary to identify all frames"""
    return int(np.log10(framenumber))+1

@timeit
def save_all_image_frames(f, det_no: str, name: str, path:str ,
               scale_bar: bool = True, show_fig: bool = False, dpi: int = 100, save_meta: bool = True,
               sb_settings: dict = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15},
               imshow_kwargs: dict = {"cmap" : "Greys_r"}, counter: int = None):
    '''
    Shortcut to save all images to separate files
    #add threading to this!
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    imgdata = get_image_data_det_no(f, det_no)
    framenum = imgdata.shape[-1]
    toloop = range(framenum) #loop over number of frames

    #set the number of digits in the counter
    mincounter = get_counter(framenum)
    if counter is None:
        counter = mincounter
    elif counter<mincounter:
        logging.error("Insufficient digits to represent the frames")
        return
    else:
        pass #counter is already set to the right value

    for i in toloop:
        frame = imgdata[:,:,i]
        metadata = get_meta_dict_det_no(f, "Image", det_no = det_no, frame = i)
        c = str(i).zfill(counter) #the appropriately formatted counter number
        plot_single_image(imgdata = frame,  metadata = metadata, filename = f"{path}{name}_{c}.tiff",
                       scale_bar = scale_bar, show_fig = show_fig, dpi = dpi, save_meta = save_meta,
                       sb_settings = sb_settings, imshow_kwargs = imshow_kwargs)

    # def todoinloop(i):
    #     frame = imgdata[:,:,i]
    #     metadata = get_meta_dict_det_no(f, "Image", det_no = det_no, frame = i)
    #     plot_single_image(imgdata = frame,  metadata = metadata, filename = f"{path}{name}_{i}.tiff",
    #                    scale_bar = scale_bar, show_fig = show_fig, dpi = dpi, save_meta = save_meta,
    #                    sb_settings = sb_settings, imshow_kwargs = imshow_kwargs)

    # with cf.ThreadPoolExecutor() as executor: #perform with threading
    #     #create the threads list of translate stream for all frames in loopover
    #     [executor.submit(todoinloop, i) for i in toloop]


@timeit
def save_all_spectrum_frames(f, det_no: str = 0, name: str = None,
                            path:str = "./spectrum/", counter:int = None):

    ss = get_spectrum_stream(f, det_no=det_no, one_matrix = True,
                            compress_type = "dok", re_all = False)

    ss.write_streamframes(path = path, pre = name, counter = counter)


def plot_single_image(imgdata: np.ndarray, metadata: dict, filename:str = "",
               scale_bar: bool = True, show_fig: bool = True, dpi: int = 100, save_meta: bool = True,
               sb_settings: dict = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15},
               imshow_kwargs: dict = {}):
    '''
    Plot a single NxN numpy array representing an image. Potentiallys save to a TIFF file.

    Args:
    imgdata (np.ndarray) : the image frame
    metadata (dict) : the metadata dictionary corresponding to the file
    filename (str) : the filename to which you want to save the image. if empty is not saved.
    scale_bar (bool) = True : whether to add a scale bar to the image. Metadata must contain this information.
    show_fig (bool) = True : whether to show the figure
    dpi (int) = 100 : dpi to save the image with
    save_meta (bool) = True : save the metadata also as a separate json file with the same filename as the image
    sb_settings (dict) = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15}: settings for the scale bar
    imshow_kwargs (dict) : optional formating arguments passed to the pyplot.imshow function
    '''

    pixelsize, pixelunit = get_scale(metadata)
    #initialize the figure and axes objects
    fig, ax = pl.plot_image(imgdata, pixelsize, pixelunit, scale_bar = scale_bar,
                   show_fig = show_fig, dpi = dpi,
                   sb_settings = sb_settings,
                   imshow_kwargs = imshow_kwargs)

    if filename:
        fig.savefig(filename, dpi = dpi)
        if save_meta:
            #if metadata save the metadata to a json file with the same name
            path, ext = os.path.splitext(filename)
            write_meta_json(path+".json", metadata)

    return fig, ax


def plot_image_frame(f, det_no, frm = 0, **kwargs):
    '''
    Wrapper for plot_single_image

    Args:
    f: hdf5 file
    det_no: detector number
    **kwargs to be passed to plot_single_image
    '''
    imdat = get_image_data_det_no(f, det_no)[:,:,frm]
    metdat = get_meta_dict_det_no(f, "Image", det_no = det_no, frame = frm)
    fig, ax = plot_single_image(imdat, metdat, **kwargs)
    return fig, ax
