"""
Main module to interact with TEM data
"""

# Base modules
import sys
import os
import re
import logging
from datetime import datetime
from pathlib import Path

# Basic 3rd party packages
import h5py
import numpy as np
import json
import scipy.sparse as spa
import scipy.ndimage as ndi
import pandas as pd
from tqdm import tqdm

# For working with images
from PIL import Image

# My own modules
from . import plottingtools as pl
from . import image_filters as imf
from ._decorators import timeit
from .guitools import file_dialog as fdo
from . import processing as proc
from . import metadata as mda
from . import jsontools as jt
from . import algebrautils as algutil

import concurrent.futures as cf
from multiprocessing import Pool, cpu_count

# Initialize the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

UUIDREGEXF = (r"[0-9a-f]{8}[0-9a-f]{4}[0-9a-f]{4}"
              r"[0-9a-f]{4}[0-9a-f]{12}")
UUIDREGEX = (r"[0-9a-f]{32}")

DEGREE = u"\u00b0"
MU = u"\u03BC"


def read_emd(filepath: str):
    """Open an emd file with a given filepath"""
    try:
        return EMDFile(filepath, 'r')
    except ValueError:
        logger.error("Invalid file path, returning None")
        return None
    except OSError:
        logger.error("Could not open file, returning None")
        return None


def open_emd_gui():
    """
    Open a file dialog to open emd file as EMDFile object
    """
    emd_filename = fdo.open(filters="EMD files (*.emd)")
    f = read_emd(emd_filename)
    return f


def compress_matrix(temp, compress_type="csr"):
    """
    Returns a sparse matrix representation of a 2D numpy array
    """
    # Performance wise, it doesn't seem to matter which type of
    # compression we take. We choose csr.
    if not isinstance(temp, np.ndarray):
        raise TypeError("Expected a numpy array.")
    if not temp.ndim == 2:
        raise ValueError("Array must be 2 dimensional.")
    if np.log10(temp.size) > 8:
        logger.warning("Very large array (>1E8 elements), "
                       "this can take a while...")
    spars = spa.csr_matrix(temp)
    spars = change_compress_type(spars,
                                 compress_type=compress_type)
    return spars


def change_compress_type(temp, compress_type):
    """
    Return a matrix with a different kind of compression

    Parameters
    ----------
    temp : scipy.spmatrix
        Sparse matrix to convert
    compress_type : str
        Type to convert to. Options are 'none', 'dok', 'csc',
        'csr'. If 'none' is chosen, the numpy array is returned.

    Returns
    -------
    temp : scipy.spmatrix or numpy.ndarray
        Converted matrix
    """
    if not isinstance(temp, spa.spmatrix):
        raise TypeError("Expected a sparse array.")
    if compress_type == 'none':
        if np.log10(temp.shape[0]*temp.shape[1]) > 8:
            logger.warning("Very large matrix (>1E8 elements), "
                           "this can take a while...")
        return temp.toarray()
    elif compress_type == 'dok':
        return temp.todok()
    elif compress_type == 'csc':
        return temp.tocsc()
    elif compress_type == 'csr':
        return temp.tocsr()
    else:
        raise ValueError("Not recognized compression type "
                         "should be none, dok, csc or csr")


def _get_counter(number_of_frames):
    """Calculate minimum # of digits to label frames"""
    return int(np.log10(number_of_frames))+1


class EMDFile(h5py.File):
    """
    Represents a Velox EMD file containing multiple datasets

    Methods
    -------
    print_raw_structure
        Print the all the datasets inside the file. Not all are useable.
    print_simple_structure
        Print the directly useable datasets inside the file and
        some info about them
    find_dataset
        Find a dataset based on a uuid, a search word, or a key-value pair
    get_raw_data
        Return the raw dataset contained in the emd file. For images this is
        directly interpretable, for spectrumstreams it is not.
    get_dataset
        Extract a dataset as an appropriate object with its methods

    Hidden methods
    --------------
    _get_simple_im_rep
        get a simple string representation of an image dataset
    _get_simple_ss_rep
        get a simple string representation of a spectrum stream dataset
    _get_simple_structure
        get a simple string representation of the data in the file
    _get_ds_uuid
        get a dataset uuid from an index number
    _get_meta_dict
        get the metadata associated with a dataset
    _get_meta_dict_by_path
        get the metadata associated with a dataset
    _get_meta_dict_ds_no
        get the metadata associated with a dataset
    _get_sig_ds_from_path
        return the signal and uuid from a path
    _for_all_datasets_do
        loop over all datasets and perform a certain function
    _get_spectrum_stream_acqset
        get the acquisition settings dictionary, usually in a spectrumstream
        dataset
    _get_spectrum_stream_flut
        get the frame lookup table, usualy in a spectrumstream dataset
    _get_spectrum_stream_dim
        get the dimensions of a spectrum stream (xs, ys, chan, frame_dim)
    _get_image_data_ds_no
        get the data corresponding to an image with a dataset index
    _guess_multiframe_dataset
        guess the dataset that contains multiple frames
    _get_dataset_frames
        guess the number of frames in a dataset
    _guess_acquisition_type
        guess the type of dataset it is.
    _create_simple_metadata
        create a more intuitive metadata representation of a dataset

    Static methods
    --------------
    _get_ds_info
        of a dataset, return uuid, shape and data type of a hdf5 dataset
    _get_name
        return the name or path of am hdf5 node
    _scan_node
        (recursively) print the structure of an hdf5 node
    _get_scale
        return the pixel size and unit of a dataset. Provide metadata
        dictionary.
    _get_detector property
        return the value of a property of the detector used to acquire
        the dataset. Provide a metadata dictionary.
    _convert_stream_to_sparse
        convert a spectrumstream array into a 2D sparse matrix of dimension
        (xs*ys*frame_dim, chan). Provide array and dimensions.
    _get_frame_limits
        if a flut exists in a spectrumstream dataset, get the index of the
        first and last pixel corresponding to a frame.
    _get_frame_indexes
        get all indexes that belong to a particular frame in a spectrumstream.
        Flut required.
    _get_frames_indexes
        get all indexes that belong to a list of frames in a spectrumstream.
        Flut required.
    _translate_stream_frame
        translate one frame into a spare format (xs*ys, chan)
    _create_dataset_object
        transform the emd dataset into the appropriate format for working
        with it
    """
    valid_signals = ["Image", "SpectrumStream"]
    # to implement: "Line", "Spectrum", "SpectrumImage",

    def __init__(self, filepath, mode="r"):
        super().__init__(filepath, mode)

    @staticmethod
    def _get_ds_info(ds):
        """Return key properties of an h5py dataset: uuid, shape, datatype"""
        uuid = EMDFile._get_name(ds.parent)
        shape = ds.shape
        dt = ds.dtype
        return uuid, shape, dt

    @staticmethod
    def _get_name(x, full_path=False):
        """
        Get name of an hdf5 node

        Returns the short or long group/dataset name.

        Parameters
        ----------
        x : h5py.Dataset or h5py.Group
            The node for which the name is desired
        full_path : bool, optional
            If True returns the full path in file. Default is False.

        Returns
        -------
        name : str
            The name of the node
        """
        if not full_path:
            name = x.name.split("/")[-1]
        else:
            name = x.name
        if not name:
            # for the root
            name = "/"
        return name

    @staticmethod
    def _scan_node(g, tabs=0, recursive=True,
                   see_info=True, tab_step=4):
        """
        Traverse the node. Print datasets and groups on the node.

        Parameters
        ----------
        g : h5py.Dataset of h5py.Group
            The node to scans
        tabs : int, optional
            The number of tabs to start on. Defaults to 0s
        recursive : bool, optional
            Scan subnodes also if they are groups. Defaults to True.
        see_info : bool, optional
            Also print basic info on a Dataset node. Default is True.
        tab_step : int, optional
            When recursive, the number of spaces between steps.
            Default is 4.
        """
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                name_ds = EMDFile._get_name(v)
                extra = ""
                if see_info:
                    extra = " ("+str(v.shape)+", "+str(v.dtype)+")"
                print(' ' * tabs + ' ' * tab_step + ' -', name_ds, extra)

            elif isinstance(v, h5py.Group):
                name_gr = EMDFile._get_name(v)
                print(' ' * tabs + ' ' * tab_step + name_gr)
                if recursive:
                    EMDFile._scan_node(v, tabs=tabs + tab_step)

    def print_raw_structure(self, **kwargs):
        """
        Print the raw structure of the data in the file.

        Other parameters
        ----------------
        **kwargs : :py:meth:`_scan_node`
        """
        print("/Data")
        EMDFile._scan_node(self["Data"], **kwargs)

    def _get_simple_im_rep(self, dspath):
        """Get a str representation of an Image dataset"""
        ds = self[dspath]
        uuid, (x, y, f), dt = EMDFile._get_ds_info(ds)
        img = ds[:, :, 0]
        mn = np.min(img)
        mx = np.max(img)
        imstring = ("UUID: {}, "
                    "Shape: {}x{}, "
                    "Frames: {}, "
                    "Data type: {}, "
                    "Min:{}, "
                    "Max:{}")
        return imstring.format(uuid, x, y, f, dt, mn, mx)

    def _get_simple_ss_rep(self, dspath):
        """Get a str representation of a SpectrumStream dataset"""
        ds = self[dspath]
        uuid, (l, _), dt = EMDFile._get_ds_info(ds)
        ssstring = ("UUID: {}, "
                    "Length: {}, "
                    "Data type: {}, ")
        return ssstring.format(uuid, l, dt)

    def _get_simple_structure(self, func=print):
        """
        Creates a simplified string representation of the file
        """
        def header(name):
            s = 8
            b = s*2+9+len(name)
            hd = ("-"*b+"\n" +
                  " "*s+"{} datasets"+" "*s+"\n" +
                  "-"*b+"\n")
            return hd.format(name)

        represent = ""
        for k, v in self["Data"].items():
            if k == "Image":
                represent += header(k)
                for c, (i, j) in enumerate(v.items()):
                    represent += f"Dataset number: {c}, "
                    path = EMDFile._get_name(j["Data"], True)
                    represent += self._get_simple_im_rep(path)
                    represent += "\n"
            elif k == "SpectrumStream":
                represent += header(k)
                for c, (i, j) in enumerate(v.items()):
                    represent += f"Dataset number: {c}, "
                    path = EMDFile._get_name(j["Data"], True)
                    represent += self._get_simple_ss_rep(path)
                    represent += "\n"
        func(represent)
        return represent

    def print_simple_structure(self):
        """Print simplified string representation of the file"""
        self._get_simple_structure(func=print)

    def _get_ds_uuid(self, sig, ds_no):
        """
        Get the UUID key for the dataset from the index number

        Inside Velox EMD files, all datasets are contained in a
        group identified by a UUID. Accessing the dataset is often
        easier by index number.

        Parameters
        ----------
        sig : str
            Signal type. Options are "Image", "Line", "Spectrum",
            "SpectrumImage", "SpectrumStream"
        ds_no : int
            The dataset index number

        Returns
        -------
        ds_uuid : str
            The UUID of the group containing data and metadata
        """
        if not isinstance(sig, str):
            raise ValueError("Invalid type for signal argument")
            return
        if not isinstance(ds_no, int):
            raise ValueError("Invalid type for dataset number")
            return
        try:
            imdat = self["Data"][sig]
        except KeyError:
            logger.error(f"No {sig} data found in this file")
            return
        try:
            # get a list of the UUID named groups in this group
            dats = list(imdat.keys())
            # get the UUID corresponding to the dataset number
            ds_uuid = dats[ds_no]
            return ds_uuid
        except IndexError:
            logger.error(f"Dataset {ds_no} is out of bounds. "
                         f"Valid keys on the {sig} signal are "
                         f"{list(range(len(dats)))}")
            return

    def _get_meta_dict(self, sig, ds, frame=0):
        """
        Return dictionary representaiton of EMD metadata.

        EMD stores metadata in an array of integers representing
        characters. Combining all characters makes a JSON string.
        This returns the metadata corresponding to a specific
        frame inside a dataset inside a signal.

        Parameters
        ----------
        sig : str
            Signal type. Options are "Image", "Line", "Spectrum",
            "SpectrumImage", "Spectrumstream"
        ds : str
            The dataset UUID
        frame : int
            The column corresponding to the frame. Defaults to 0,
            the first frame

        Returns
        -------
        meta_dict : dict
            Dictionary representing the original metadata
        """
        if not isinstance(sig, str):
            raise TypeError("Invalid type for signal argument")
            return
        if not isinstance(ds, str):
            raise TypeError("Invalid type for dataset UUID")
            return
        if not isinstance(frame, int):
            raise TypeError("Invalid type for frame number")
            return
        try:
            imdat = self["Data"][sig]
        except KeyError:
            raise KeyError(f"{sig} signal not found in dataset")
        try:
            meta = imdat[ds]['Metadata']
        except KeyError:
            raise KeyError(f"Dataset {ds} not found in {sig} signal")
        except ValueError:
            logger.warning(f"No metadata found in {sig}/{ds}, returning None")
            return None
        try:
            mcolumn = meta[:, frame]
        except IndexError:
            raise IndexError(f"{frame} is an invalid index for metadata."
                             f" Valid is from 0-{meta.shape[1]-1}.")
        # We do not account for a case where metadata may be non-
        # two-dimensional
        # turn the metadata array into a list and trim the zeros
        meta_array = list(np.trim_zeros(mcolumn))
        # turn the list of ints into chars
        meta_char = list(map(lambda x: chr(x), meta_array))
        # concatenate the chars into a string
        meta_json = "".join(meta_char)
        # interpret the string with json to create a dictionary
        meta_dict = json.loads(meta_json)
        # return the dictionary
        return meta_dict

    def _get_meta_dict_by_path(self, path, frame=0):
        """
        Wrapper for :py:meth:`_get_meta_dict`

        This returns the metadata corresponding to a specific
        frame inside a dataset inside a signal. Here a path
        is provided
        """
        sig, ds_uuid = self._get_sig_ds_from_path(path)
        return self._get_meta_dict(sig, ds_uuid, frame)

    def _get_meta_dict_ds_no(self, sig, ds_no, frame=0):
        """
        Wrapper for :py:meth:`_get_meta_dict`

        This returns the metadata corresponding to a specific
        frame inside a dataset inside a signal.

        Parameters
        ----------
        sig : str
            Signal type. Options are "Image", "Line", "Spectrum",
            "SpectrumImage", "Spectrumstream"
        ds_no : int
            The dataset index number.
        frame : int, optional
            The column corresponding to the metadata frame.
            Defaults to 0, the first frame.

        Returns
        -------
        meta_dict : dict
            Dictionary representing the original metadata
        """
        dat = self._get_ds_uuid(sig, ds_no)
        return self._get_meta_dict(sig, dat, frame)

    @staticmethod
    def _get_scale(metadata):
        '''
        Returns the size of a pixel and the unit as tuple

        It is assumed that the scale of a pixel in x and y
        is the same. This only works if it is image metadata.
        '''
        pixelsize = (float(metadata["BinaryResult"]["PixelSize"]
                           ["width"]))
        pixelunit = metadata["BinaryResult"]["PixelUnitX"]
        return pixelsize, pixelunit

    @staticmethod
    def _get_detector_property(meta, prop, exact_match=False):
        '''
        Get the value of a detector property

        A detector has a certain name which is stored in the
        `BinaryResult` subdict in the dataset metadata.
        However, the properties of these
        detectors are stored in `Detectors/Detector-n`
        A loop is performed over these detectors to search
        for a match with the name, then return the desired property.

        Parameters
        ----------
        meta : dict
            The metadata dictionary
        prop : str
            Key to the desired property name
        exact_match : bool, optional
            Some detector names may not match exactly between
            the one in BinaryResult and Detectors. For example
            the detector for EDX may be SuperX but in Detectors
            there exists SuperX1, SuperX2, ... Default is False
            to include fuzzy match.

        Returns
        -------
        property : str
            Detector property
        '''
        # The detector name to search for
        det_name = meta["BinaryResult"]["Detector"]
        # loop over all the detectors
        for i in meta["Detectors"].keys():
            det_dic = meta["Detectors"][i]
            if exact_match:
                test = (det_name == det_dic["DetectorName"])
            else:
                test = (det_name in det_dic["DetectorName"])
            if test:
                try:
                    return det_dic[prop]
                except KeyError:
                    logger.error(f"Property {prop} not found on"
                                 f"detector {det_name}")
                    return
        logger.error("No detector info was found")
        return

    def _get_sig_ds_from_path(self, path):
        """
        Returns the signal name and dataset UUID from a path

        Also tests if a path is valid
        """
        try:
            self[path]
        except KeyError:
            logger.error(f"The path {path} can not be resolved.")
            return
        rex = (r"/([a-zA-Z]+)/([0-9a-f]{8}[0-9a-f]{4}[0-9a-f]{4}"
               r"[0-9a-f]{4}[0-9a-f]{12})")
        try:
            sig, ds_uuid = re.findall(rex, path)[0]
            return sig, ds_uuid
        except IndexError:
            logger.error(f"Path {path} does not contain a "
                         f"signal and/or UUID.")
            return None, None

    def _for_all_datasets_do(self, func):
        """
        Helper method to apply a function to all datasets

        The function must accept only the path to the parent.
        Returns a list of return values
        """
        rv = []
        for k, v in self["Data"].items():
            if k in EMDFile.valid_signals:
                for _, (_, j) in enumerate(v.items()):
                    path = j.name
                    oup = func(path)
                    rv.append(oup)
        return rv

    def _get_spectrum_stream_acqset(self, det):
        '''Return the AcquisitionSettings of a dataset as dict'''
        try:
            s = self[f"Data/SpectrumStream/"
                     f"{det}/AcquisitionSettings"][0]
            return json.loads(s)
        except KeyError:
            logger.error(f"No AcquisitionSettings found in {det}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Dataset {det} contains a corrupted"
                         f"AcquisitionSetting")
            return None

    def _get_spectrum_stream_flut(self, det):
        '''Return the FrameLocationTable of a dataset as array'''
        try:
            s = self[f"Data/SpectrumStream/{det}/FrameLocationTable"]
            return s[:, 0]
        except KeyError:
            logger.error(f"No FrameLocationTable found in {det}")
            return None

    @staticmethod
    def _convert_stream_to_sparse(d1d, dim, dv=65535,
                                  compress_type='dok'):
        '''
        Converts an EMD stream to a sparse representation

        Converts a given numpy array formatted like the emd
        SpectrumStream data into a scipy sparse matrix.
        The rows of this matrix include all the scan positions,
        the columns represent the number of channels.

        Parameters
        ----------
        d1d : 1D numpy.ndarray
            The array to be converted
        dim : tuple
            The (x, y, channel, frame) dimension of the
            hypothetical returned csc_matrix. Frame can be none,
            then it is calculated based on the size of d1d.
        dv : int
            The value in the SpectrumStream that encodes "next pixel"
            Default is 65535.
        compress_type : str
            Type of compression that should be used to store the
            return data memory. See Scipy.sparse for details.
            Options: 'dok'(Default), 'none', 'csc', 'csr'.

        Returns
        -------
        temp : scipy.sparse.spmatrix or np.ndarray
            The converted spectrumstream
        '''
        # unpack the dimensions
        xs, ys, ch, frm = dim
        # find the indexes where counts are registered (!= the counting number)
        cinx = np.argwhere(d1d != dv)[:, 0]
        # also query the counts
        chans = d1d[cinx]
        # calculate the number of frames based on the size
        # if frm is unknown
        if frm is None:
            pxlcount = d1d.shape[0] - cinx.shape[0]
            frm = pxlcount // xs // ys
        # calc the dimensions of the sparse matrix
        dim_sparse = (xs*ys*frm, ch)
        # calc the pixel index to which these counts must be mapped
        pixind = cinx - np.arange(cinx.shape[0]) - 1
        # create an array of 1 representing the count
        ns = np.ones(chans.shape[0], dtype=d1d.dtype)
        # create the sparse matrix
        temp = spa.coo_matrix((ns, (pixind, chans)), shape=dim_sparse)
        # return the right type depending on chosen compression
        compress_type = compress_type.lower()
        if compress_type == 'none':
            return temp.toarray()
        elif compress_type == 'dok':
            return temp.todok()
        elif compress_type == 'csc':
            return temp.tocsc()
        elif compress_type == 'csr':
            return temp.tocsr()
        else:
            raise ValueError("Not recognized compression type, "
                             "should be none, dok, csc or csr")
            return temp

    @staticmethod
    def _get_frame_limits(frm, flut):
        '''
        Get the first and last index of a SpectrumStream frame

        In this way SpectrumStreamData[ix1:ix2] can be efficiently
        queried.
        '''
        assert isinstance(frm, int), "Must provide valid frame index"
        ix1 = flut[frm]  # we will get index error if
        try:
            ix2 = flut[frm+1]
        except IndexError:  # the last index
            ix2 = None
        return ix1, ix2

    @staticmethod
    def _get_frame_indexes(frm, flut, totln=None):
        '''
        Get all indexes from one SpectrumStream frame

        :deprecated:

        Notes
        -----
        It is much faster to load the entire spectrum stream and
        perform slicing on this.
        '''
        assert isinstance(frm, int), "Must provide valid frame index"
        ix1, ix2 = EMDFile._get_frame_limits(frm, flut)
        if ix2 is None:
            # In the last frame, the lookuptable doesn't know
            # the final index. must be provided
            assert isinstance(totln, int), (
                "For the last frame, the total length of the"
                " stream must be provided")
            ix2 = totln
        return np.arange(ix1, ix2)

    @staticmethod
    def _get_frames_indexes(frms, flut, totln=None):
        '''
        Get all indexes from multiple Spectrumstream frames.

        Performs a simple loop and performs
        :py:meth:`_get_frame_indexes()` and adds the arrays
        together.

        :depricated:

        Notes
        -----
        Querying data from SpectrumStream d is much faster
        with d[ix1:ix2] than with d[list of indexes].
        '''
        inxs = np.array([])
        for i in frms:
            inxs = np.append(inxs,
                             EMDFile._get_frame_indexes(i,
                                                        flut,
                                                        totln=totln))
        return inxs.astype(int)

    @staticmethod
    def _translate_stream_frame(d, flut, xs, ys, cs, frm,
                                dv=65535, compress_type='dok'):
        '''
        Return an array representation of a spectrum stream frame.

        The rows represent a pixel index, the colums represent
        the channel. The values stored represent the counts. For
        EDX data, only a few thousand counts are registered per
        frame, so the data is very sparse.

        :depricated:

        Parameters
        ----------
        d : h5py.Dataset
            A SpectrumStream frame read from an emd file
        flut : array-like
            A frame lookup-table also read from the emd file
        xs : int
            size of the scanning grid in the x-direction
        ys : int
            size of the scanning grid in the y-direction
        cs : int
            number of channels
        frm : int
            frame number
        dv : int
            the number in the data that should be interpreted as a
            pixel counter. Default = 65535
        compress_type : str
            the type of compression. Options: 'dok'(Default),
            'none', 'csc', 'csr'.

        Returns
        -------
        temp : array-like or sparse matrix
            The sparse matrix representation of the frame
        '''
        ix1, ix2 = EMDFile._get_frame_limits(frm, flut)
        # query the frame from the long spectrumstream
        d1d = d[ix1:ix2].flatten()
        temp = EMDFile._convert_stream_to_sparse(
            d1d, (xs, ys, cs, None), dv=dv,
            compress_type=compress_type)
        return temp

    def _get_spectrum_stream_dim(self, ds_uuid):
        """Return (xs, ys, chan, frame_dim) of SpectrumStream"""
        acq = self._get_spectrum_stream_acqset(ds_uuid)
        flut = self._get_spectrum_stream_flut(ds_uuid)
        chan = int(acq["bincount"])  # number of channels
        xs = int(acq['RasterScanDefinition']['Width'])
        ys = int(acq['RasterScanDefinition']['Height'])
        if flut is not None:
            frame_dimension = flut.shape[0]
        else:
            # if there is no lookup table try to find
            # corresponding image dataset
            dsim = self._guess_multiframe_dataset()
            try:
                frame_dimension = dsim[2]
            except IndexError:
                logger.error("No estimate could be made of "
                             "the number of frames")
                frame_dimension = None
        return (xs, ys, chan, frame_dimension)

    def find_dataset(self, searchterm, value=None):
        """
        Return dataset or datasets based on a searchterm

        Searchterm can be a UUID (recommended) or some metadata
        property. If a value is provided, an exact match will be
        searched between the searchterm as a key and the value

        Parameters
        ----------
        searchterm : object
            UUID or some part of the nested metadata
            dictionary, usually a string
        value : object, optional
            If provided, a key-value pair will be searched in the
            metadata dictionary. Only if a match is found on both
            will the corresponding dataset be returned.

        Returns
        -------
        result : h5py.Dataset or list of h5py.Dataset
            All the datasets where a match is found
        """
        finduuid = re.findall(UUIDREGEX, searchterm)
        if finduuid:
            for sig, v in self["Data"].items():
                for uid, j in v.items():
                    if uid in searchterm:
                        return self.get_raw_data(sig, uid)
        else:
            if value is None:
                logger.warning("No value is provided, will return "
                               "all matches including keys and values")
            matches = []
            for sig, v in self["Data"].items():
                for uid, j in v.items():
                    meta = self._get_meta_dict(sig, uid)
                    if value is None:
                        result = jt.search_json(meta, searchterm)
                    else:
                        result = jt.search_json_kv(meta, searchterm, value)
                    if result:
                        dataset = self.get_raw_data(sig, uid)
                        if dataset:
                            matches.append(dataset)
            return matches

    def get_raw_data(self, sig, det):
        """
        Returns raw data in dataset

        Based on the signal and detector uuid, returns the raw
        dataset

        Parameters
        ----------
        sig : str
            Signal type (Data subgroup). Valid options are
            given in :py:attr:`valid_signals`
        det : str
            The dataset group UUID string

        Returns
        -------
        data : h5py.Dataset
            The actual underlying data
        """
        try:
            data = self['Data'][sig][det]['Data']
            return data
        except KeyError:
            logger.error("Dataset not found")
            return

    def get_dataset(self, sig, uuid):
        """
        Extract the data as an appropriate object with its methods
        """
        original_metadata = self._get_meta_dict(sig, uuid)
        metadata = self._create_simple_metadata(sig, uuid)
        raw_dt = self.get_raw_data(sig, uuid)
        return EMDFile._create_dataset_object(raw_dt,
                                              metadata,
                                              original_metadata)

    @staticmethod
    def _create_dataset_object(rd, meta, ometa):
        extyp = meta.experiment_type
        if extyp == "image" or extyp == "scan_image":
            d = np.array(rd[:, :, 0])  # shape is (y, x, 1)
            return GeneralImage(d, meta, ometa)
        elif extyp == "image_series" or extyp == "scan_image_series":
            d = np.array(rd)  # shape is (y, x, frame)
            d = np.rollaxis(d, -1)  # make shape (frame, y, x)
            return GeneralImageStack(d, meta, ometa)
        elif extyp == "spectrum_stream":
            dims = (meta.data_axes.scan_x.bins,
                    meta.data_axes.scan_y.bins,
                    meta.data_axes.channel.bins,
                    meta.data_axes.frame.bins)
            specstr = EMDFile._convert_stream_to_sparse(
                rd.value.ravel(), dims, compress_type="csr")
            return SpectrumStream(specstr, meta, ometa)

    def _get_image_data_ds_no(self, ds_no):
        """Get an image dataset with an index number"""
        sig = "Image"
        det = self._get_ds_uuid(sig, ds_no)
        return self.get_raw_data(sig, det)

    def _guess_multiframe_dataset(self):
        '''
        Return UUID of the first multi-frame image dataset

        Especially useful for auto-detecting the image dataset
        that corresponds to the SpectrumStream frames.

        Returns
        -------
        ds_no : int
            dataset number
        ds_uuid : str
            dataset uuid
        num_frames : int
            number of image frames
        '''
        # find the detector number of the dataset
        # containing many frames
        for k, (uid, ds) in enumerate(
                    self["Data"]["Image"].items()):
            img = ds["Data"]
            if img.shape[-1] > 1:
                ds_no = k
                det_uuid = uid
                num_frames = img.shape[-1]
                break
        else:  # no break statement found
            logger.error("No image dataset found"
                         "with more than 1 frame.")
            return
        return ds_no, det_uuid, num_frames

    def _get_dataset_frames(self, sig, uuid):
        """Return the number of frames of a dataset"""
        dt = self.get_raw_data(sig, uuid)
        # verify that the number of dimensions is 3
        if sig == "Image":
            try:
                assert dt.ndim == 3
                return dt.shape[-1]
            except AssertionError:
                logger.error("The dataset does not contain the required "
                             "number of dimensions (3).")
                return None
        elif sig == "SpectrumStream":
            _, _, _, f = self._get_spectrum_stream_dim(uuid)
            return f
        else:
            raise ValueError(f"Signal type {sig} does not have frames")

    def _guess_acquisition_type(self, sig, uuid):
        """
        Guess the type of data the dataset contains

        At this point it can only guess:
        # - image
        # - image_series
        # - scan_image
        # - scan_image_series
        # - point_spectrum (No)
        # - line_spectrum (No)
        # - spectrum_map (No)
        # - spectrum_stream
        # - scan_image_map (No)
        """
        s = self._get_meta_dict(sig, uuid)
        if sig == "Image":
            det_typ = self._get_detector_property(s, "DetectorType")
            frames = self._get_dataset_frames(sig, uuid)
            if det_typ == "ImagingDetector":
                if frames == 1:
                    return "image"
                elif frames > 1:
                    return "image_series"
                else:
                    raise ValueError(f"Error in the dataset, "
                                     f"recognized {frames} frames")
            elif det_typ == "ScanningDetector":
                if frames == 1:
                    return "scan_image"
                elif frames > 1:
                    return "scan_image_series"
                else:
                    raise ValueError(f"Error in the dataset, "
                                     f"recognized {frames} frames")
            else:
                raise ValueError(f"Dataset is not a real experimental image. "
                                 f"The detector used was of type: {det_typ}. "
                                 f"Likely it derives from the spectrum info. "
                                 f"Please work with those datasets instead.")
        elif sig == "SpectrumStream":
            return "spectrum_stream"
        elif sig == "Spectrum":
            return "point_spectrum"
        elif sig == "SpectrumImage":
            return "spectrum_map"
        else:
            raise NotImplementedError(f"The datasets with type {sig} "
                                      f"is as of yet unreadable. Please "
                                      f"file a bug report")

    def _create_simple_metadata(self, sig, uuid, save_file=None):
        """
        Creates a unified metadata format of a dataset according to JSON schema

        Parameters
        ----------
        sig : str
            Signal type. Options are "Image", "Line", "Spectrum",
            "SpectrumImage", "Spectrumstream"
        uuid : str
            The dataset UUID
        save_file : str, optional
            Path to a file where the JSON is written to. If None (default),
            the metadata will not be written out.

        Returns
        -------
        simple_meta : Metadata object
        """
        # first get the main metadata
        emdict = self._get_meta_dict(sig, uuid)
        s = mda.Metadata(emdict)

        def gen_optics(m):
            """
            Create the optics array from the original metadata
            We use only the active apertures.
            """
            # create dictionary for apertures
            aps = mda.DotDict()
            for _, v in m.Optics.Apertures.items():
                name = v.Name
                if isinstance(name, str):
                    name = name.lower()
                else:
                    name = "unnamed"
                order = v.Number
                typ = v.Type
                pos = v.PositionOffset
                if pos:
                    x = pos.x
                    y = pos.y
                else:
                    x = None
                    y = None
                if v.Type == "Circular":
                    # slightly more reliable workaround
                    key = f"Aperture[{v.Name}].Name"
                    diam = s.CustomProperties[key].value
                    shape = {"diameter": mda.numerical_value(diam,
                                                             "m", factor=1e-6)}
                else:
                    shape = None
                if v.Enabled != "0" and v.Enabled is not None:
                    ap = mda.aperture(order, typ, x, "m", y, "m", shape)
                    aps[name] = ap
            # create entries for lenses
            lns = mda.DotDict()
            for k, v in m.Optics.items():
                if "LensIntensity" in k:
                    n = k.split("LensIntensity")[0].lower()
                    lns[n] = mda.lens(v, None)
            optcs = mda.DotDict()
            optcs["apertures"] = aps
            optcs["lenses"] = lns
            return optcs

        def gen_beam(m):
            """Generate beam information dictionary"""
            # determine the illumination mode
            bem = mda.DotDict()
            if s.Optics.ProbeMode == "1":
                bem.mode = "convergent"
                bem.convergence_angle = mda.numerical_value(
                    m.Optics.BeamConvergence, "mrad", factor=1e3)
            elif s.Optics.ProbeMode == "2":
                bem.mode == "parallel"
            else:
                logger.error("Could not determine illumination mode")
                return bem
            bem.screen_current = mda.numerical_value(
                                    m.Optics.LastMeasuredScreenCurrent, "A")
            bem.spot_index = int(m.Optics.SpotIndex)
            return bem

        def gen_imaging(s):
            """"Generate imaging information"""
            imdic = mda.DotDict()
            # determine the imaging mode
            if s.Optics.ProjectorMode == "1":
                imdic.mode = "diffraction"
                imdic.camera_length = mda.numerical_value(
                                        s.Optics.CameraLength, "m")
                if s.Optics.ProbeMode == "1":
                    imdic.stem_magnification = int(
                            float(s.CustomProperties.StemMagnification.value))
            elif s.Optics.ProjectorMode == "2":
                imdic.mode == "imaging"
                imdic.magnification == int(
                            float(s.Optics.NominalMagnification))
            else:
                logger.error("Could not determine imaging mode")
                return imdic
            imdic.defocus = mda.numerical_value(s.Optics.Defocus, "m")
            return imdic

        def gen_dax(s, datatype):
            """Generate data_axes information"""
            # autogenerate axes
            if datatype == "image":
                binx = int(s.BinaryResult.ImageSize.width)
                unitx = s.BinaryResult.PixelUnitX
                scalx = s.BinaryResult.PixelSize.width
                xinfo = (binx, unitx, scalx)
                biny = int(s.BinaryResult.ImageSize.height)
                unity = s.BinaryResult.PixelUnitY
                scaly = s.BinaryResult.PixelSize.height
                yinfo = (biny, unity, scaly)
                return mda.gen_image_axes(xinfo, yinfo)
            if datatype == "image_series":
                binx = int(s.BinaryResult.ImageSize.width)
                unitx = s.BinaryResult.PixelUnitX
                scalx = s.BinaryResult.PixelSize.width
                xinfo = (binx, unitx, scalx)
                biny = int(s.BinaryResult.ImageSize.height)
                unity = s.BinaryResult.PixelUnitY
                scaly = s.BinaryResult.PixelSize.height
                yinfo = (biny, unity, scaly)
                framenum = self._get_dataset_frames(sig, uuid)
                exptime = self._get_detector_property(s, "ExposureTime")
                tinfo = (framenum, "s", exptime)
                return mda.gen_image_stack_axes(xinfo, yinfo, tinfo)
            if datatype == "scan_image":
                binx = int(s.Scan.ScanSize.width)
                unitx = s.BinaryResult.PixelUnitX
                scalx = s.BinaryResult.PixelSize.width
                xinfo = (binx, unitx, scalx)
                biny = int(s.Scan.ScanSize.height)
                unity = s.BinaryResult.PixelUnitY
                scaly = s.BinaryResult.PixelSize.height
                yinfo = (biny, unity, scaly)
                taxis = mda.left_to_right_top_to_bottom_scan(binx,
                                                             biny,
                                                             1,
                                                             s.Scan.DwellTime,
                                                             s.Scan.LineTime,
                                                             s.Scan.FrameTime)
                return mda.gen_scan_image_axes(xinfo, yinfo, taxis)
            if datatype == "scan_image_series":
                binx = int(s.Scan.ScanSize.width)
                unitx = s.BinaryResult.PixelUnitX
                scalx = s.BinaryResult.PixelSize.width
                xinfo = (binx, unitx, scalx)
                biny = int(s.Scan.ScanSize.height)
                unity = s.BinaryResult.PixelUnitY
                scaly = s.BinaryResult.PixelSize.height
                yinfo = (biny, unity, scaly)
                framenum = self._get_dataset_frames(sig, uuid)
                taxis = mda.left_to_right_top_to_bottom_scan(binx,
                                                             biny,
                                                             framenum,
                                                             s.Scan.DwellTime,
                                                             s.Scan.LineTime,
                                                             s.Scan.FrameTime)
                return mda.gen_scan_image_stack_axes(xinfo, yinfo,
                                                     framenum, taxis)
            if datatype == "spectrum_point":
                raise NotImplementedError(f"{datatype} can not be converted "
                                          f"directly from EMD files yet. "
                                          f"Please request this feature if "
                                          f"needed.")
            if datatype == "spectrum_line":
                raise NotImplementedError(f"{datatype} can not be converted "
                                          f"directly from EMD files yet. "
                                          f"Please request this feature if "
                                          f"needed.")
            if datatype == "spectrum_map":
                raise NotImplementedError(f"{datatype} can not be converted "
                                          f"directly from EMD files yet. "
                                          f"Please request this feature if "
                                          f"needed.")
                binx = int(s.Scan.ScanSize.width)
                unitx = s.BinaryResult.PixelUnitX
                scalx = s.BinaryResult.PixelSize.width
                xinfo = (binx, unitx, scalx)
                biny = int(s.Scan.ScanSize.height)
                unity = s.BinaryResult.PixelUnitY
                scaly = s.BinaryResult.PixelSize.height
                yinfo = (biny, unity, scaly)
                acq = self._get_spectrum_stream_acqset(uuid)
                channels = int(acq["bincount"])
                specunit = "keV"
                dispersion = float(self._get_detector_property(
                                    s, "Dispersion"))/1000
                eoffset = float(self._get_detector_property(
                                    s, "OffsetEnergy"))/1000
                cinfo = (channels, specunit, dispersion, eoffset)
                return mda.gen_spectrum_map_axes(xinfo, yinfo, cinfo)
            if datatype == "spectrum_stream":
                binx = int(s.Scan.ScanSize.width)
                unitx = s.BinaryResult.PixelUnitX
                scalx = s.BinaryResult.PixelSize.width
                xinfo = (binx, unitx, scalx)
                biny = int(s.Scan.ScanSize.height)
                unity = s.BinaryResult.PixelUnitY
                scaly = s.BinaryResult.PixelSize.height
                yinfo = (biny, unity, scaly)
                framenum = self._get_dataset_frames(sig, uuid)
                acq = self._get_spectrum_stream_acqset(uuid)
                channels = int(acq["bincount"])
                specunit = "keV"
                dispersion = float(self._get_detector_property(
                                    s, "Dispersion"))/1000
                eoffset = float(self._get_detector_property(
                                    s, "OffsetEnergy"))/1000
                cinfo = (channels, specunit, dispersion, eoffset)
                taxis = mda.left_to_right_top_to_bottom_scan(binx,
                                                             biny,
                                                             framenum,
                                                             s.Scan.DwellTime,
                                                             s.Scan.LineTime,
                                                             s.Scan.FrameTime)
                return mda.gen_spectrum_stream_axes(xinfo, yinfo, framenum,
                                                    cinfo, taxis)
            if datatype == "scan_image_map":
                raise NotImplementedError(f"{datatype} can not be converted "
                                          f"directly from EMD files yet. "
                                          f"Please request this feature if "
                                          f"needed.")

        def gen_detx(s):
            """Generate detector information"""
            det = mda.DotDict()
            det.name = s.BinaryResult.Detector
            det.type = self._get_detector_property(s, "DetectorType")
            if det.type == "ScanningDetector":
                det.gain = mda.numerical_value(
                    self._get_detector_property(s, "Gain"), None)
                det.offset = mda.numerical_value(
                    self._get_detector_property(s, "Offset"), None)
                col_angles = self._get_detector_property(
                                        s, "CollectionAngleRange")
                det.collection_angles = mda.DotDict()
                det.collection_angles.begin = mda.numerical_value(
                        col_angles["begin"], "mrad", factor=1e3)
                det.collection_angles.end = mda.numerical_value(
                        col_angles["end"], "mrad", factor=1e3)
                det.pixel_dwell_time = mda.numerical_value(
                        s.Scan.DwellTime, "s")
            elif det.type == "ImagingDetector":
                det.exposure_time = mda.numerical_value(
                    self._get_detector_property(s, "ExposureTime"), "s")
            elif det.type == "AnalyticalDetector":
                pass
            else:
                logger.error("Detector type not recognized.")
            return det

        experiment = self._guess_acquisition_type(sig, uuid)
        # specific to the dataset
        data_axes = gen_dax(s, experiment)
        fname = self.filename.split("/")[-1]
        # general metadata for an acquired dataset
        newdict = {
            "meta": {
                "filename": fname,
            },
            "optics": gen_optics(s),
            "illumination": {
                "source": {
                    "acceleration_voltage": mda.numerical_value(
                                    s.Optics.AccelerationVoltage, "kV",
                                    factor=1e-3),
                    "extraction_voltage": mda.numerical_value(
                                    s.Optics.ExtractorVoltage, "V"),
                },
                "beam": gen_beam(s),
            },
            "imaging": gen_imaging(s),
            "acquisition": {
                "start_date_time": datetime.fromtimestamp(
                    int(s.Acquisition.AcquisitionStartDatetime.DateTime)
                    ).isoformat(),
                "detector": gen_detx(s),
            },
            "stage": {
                "holder": "",
                "position": {
                    "x": mda.numerical_value(s.Stage.Position.x, "m"),
                    "y": mda.numerical_value(s.Stage.Position.y, "m"),
                    "z": mda.numerical_value(s.Stage.Position.z, "m"),
                },
                "tilt": {
                    "alpha": mda.numerical_value(s.Stage.AlphaTilt, DEGREE),
                    "beta": mda.numerical_value(s.Stage.BetaTilt, DEGREE),
                }
            },
            "sample": s.Sample,
            "instrument": {
                "id": s.Instrument.InstrumentId,
            },
            "software": {
                "name": "Velox",
                "version": s.Instrument.ControlSoftwareVersion
            },
            "operator": {},
            "process": f"Extracted dataset {sig}/{uuid} from EMD file {fname}"
        }
        # specific to the dataset
        newdict["experiment_type"] = experiment
        newdict["data_axes"] = data_axes

        if save_file:
            try:
                jt.write_to_json(save_file, newdict)
            except TypeError:
                logger.error(f"Failed to write out the metadata"
                             f"to file {save_file}")

        return mda.Metadata(newdict)


class TEMDataSet(object):
    """
    Represents the parents class common to all TEM datasets
    """
    def __init__(self, data, metadata, original_metadata=None):
        self.data = data
        self.metadata = metadata
        self.original_metadata = original_metadata

    @property
    def history(self):
        self.metadata.print_history()

    @property
    def experiment_type(self):
        return self.metadata.experiment_type

    def _get_axis_prop(self, axname, prop):
        try:
            ax = self.metadata.data_axes[axname]
        except KeyError:
            raise KeyError(f"Axis {axname} not found in dataset")
        try:
            return ax[prop]
        except KeyError:
            raise KeyError(f"Key {prop} not found in axis {axname}")

    def _standard_representation(self):
        """
        Reshape the data into the standard representation

        Standard data representation is a 2D array with the rows representing
        all navigation axes concatenated and the columns all signal axes. In
        general for the different data types this is as follows:
        +--------------------+--------------------------------+
        |Type                | (rows, columns)                |
        +--------------------+--------------------------------+
        |image               | (1, x * y)                     |
        |scan_image          | (scan_x * scan_y, 1)           |
        |image stack         | (frame, x * y)                 |
        |scan_image_stack    | (frame * x * y, 1)             |
        |spectrum_point      | (1, channel)                   |
        |spectrum_line       | (l, channel)                   |
        |spectrum_map        | (x * y, channel)               |
        |spectrum_stream     | (x * y * frames, channel)      |
        |scan_image_map      | (scan_x * scan_y, x * y)       |
        +--------------------+--------------------------------+
        """
        # get the first combination of data axes
        navs = self.metadata.data_axes.combos[0].navigation
        sigs = self.metadata.data_axes.combos[0].signal
        navbins = 0
        sigbins = 0
        for i in navs:
            navbins *= self._get_axis_prop(i, "bins")
        for i in sigs:
            sigbins *= self._get_axis_prop(i, "bins")
        return self.data.reshape((navbins, sigbins))

    def add_custom_axis(self, name, *args, **kwargs):
        """
        Add a custom axis to the dataset. This just adds it to the metadata.

        Parameters
        ----------
        name : str
            The name of the axis, must be a string

        Other parameters
        ----------------
        *args, **kwargs : parameters passed to metadata.axis. These are
            bins : int
                number of bins on the axis
            unit : str, optional
                unit of the axis (default is empty string)
            axistype : string, optional
                type of the axis. Options are "linear", "function" or
                "lookuptable". Default is "linear"
            scale : float, optional
                Only used if linear axis. The scale of 1 pixel in the units
                provided. Default is 1.
            offset : float, optional
                Only used if linear axis. The offset of the axis with respect
                to 0. Default is 0.
            function: str, optional
                Only used if function axis. A function in string form that
                describes how the bin number is related to the axis value.
                Must use "bin" as only variable! Default is None. The function
                '3*bins+1' would be equivalent to a linear axis with scale=3
                and offset=1.
            lookuptable : list of integers, optional
                Only used if lookuptable axis. The value of each bin. Default
                is None.
        """
        assert isinstance(name, str), "Name of the axis must be a string"
        ax = mda.axis(*args, **kwargs)
        if name not in self.metadata.data_axes:
            self.metadata.data_axes[name] = ax
        else:
            logger.error(f"Axis {name} already exists, please use "
                         f"overwrite_axis")

    def overwrite_axis(self, name, *args, **kwargs):
        """
        Overwrite an axis with new parameters.

        Same as add_custom_axis but no check is performed on the existence of
        the axis.
        """
        assert isinstance(name, str), "Name of the axis must be a string"
        ax = mda.axis(*args, **kwargs)
        if name in self.metadata.data_axes:
            self.metadata.data_axes[name] = ax
        else:
            logger.error(f"Axis {name} does not exist, please use update_axis")

    def set_axis_scale(self, name, scale, unit, offset=None):
        try:
            ds = self.metadata.data_axes[name]
            assert ds.axistype == "linear", f"Axis {name} is a non-linear axis"
        except KeyError:
            raise KeyError(f"Axis {name} does not exist in this dataset")
        try:
            scale = float(scale)
        except TypeError:
            raise TypeError("Scale must be a number")
        try:
            unit = str(unit)
        except TypeError:
            raise TypeError("Unit could not be converted to a string")
        # also update the axes in the metadata
        self.update_axis_value(name, "scale", scale)
        self.update_axis_value(name, "unit", unit)
        if offset is not None:
            try:
                offset = float(offset)
            except TypeError:
                raise TypeError("Offset must be a number")
            self.update_axis_value(name, "offset", offset)

    def update_axis_value(self, name, key, value):
        """
        Update one value of an axis

        Type checking is performed.
        No additional checks are performed on function or lookuptable! This is
        a TODO.

        Parameters
        ----------
        name : str
            Name of the axis
        key : str
            Name of the axis property. Options are bins, unit, scale,
            offset, function, lookup_table. You can not update axis_type
            with this function. Use overwrite_axis for this.
        value : object
            Must be the same type as the value already had
        """
        assert name in self.metadata.data_axes, f"Axis {name} not found"
        assert key in self.metadata.data_axes[name], (f"{key} not found in "
                                                      f" axis {name}")
        assert key != "axis_type", ("To change the axis_type, you have to "
                                    "overwrite with overwrite_axis")
        typ = type(self.metadata.data_axes[name][key])
        assert typ == type(value), f"{key} must have data type {typ}"
        self.metadata.data_axes[name][key] = value

    def add_representation(self, nav_axes, sig_axes):
        """
        Add a way to represent the data on different axes
        """
        size = 1
        for i in nav_axes:
            try:
                ax = self.metadata.data_axes[i]
                size *= ax.bins
            except KeyError:
                raise ValueError(f"Axis {i} not found.")
        for j in sig_axes:
            try:
                ax = self.metadata.data_axes[j]
                size *= ax.bins
            except KeyError:
                raise ValueError(f"Axis {j} not found.")
        rsize = 1
        for k in self.data.shape:
            rsize *= k
        assert size == rsize, "The axes don't match the data size"
        newcombo = mda.ax_combo(nav_axes, sig_axes)
        if newcombo not in self.metadata.data_axes.combos:
            self.metadata.data_axes.combos.append(newcombo)
        else:
            logger.error("This combination of axes already exists")

    def get_hs(self):
        """Import and return hyperspy"""
        if "hyperspy.api" not in sys.modules:
            logger.warning("Importing hyperspy, this can take a while...")
            import hyperspy.api as hs
        else:
            hs = sys.modules["hyperspy.api"]
        return hs


def create_profile(arr, pixelsize, pixelunit, parent=None, process=None):
    """
    Create a Profile object from a 1D array, pixelsize and pixelunit

    The "experiment_type" will be set to "profile"

    Parameters
    ----------
    arr : array-like, 1D
        The array representing some intensities
    pixelsize : float
        The size of a pixel in the correct units
    pixelunit : str
        The size unit of the image
    parent : TEMDataSet, optional
        If the image is derived from another image (e.g. a profile from
        an image) then the parent is provided as an argument. The parent
        metadata will be stored in the child metadata under "parent_metadata"
    process : object, optional
        Some description of how the dataset was obtained. Could be a
        dictionary, string, ... and it will be stored under the key "process"

    Returns
    -------
    image : Profile
        The general Profile object with some easy plotting and IO methods
    """
    arr = np.array(arr)
    assert arr.ndim == 1, "The array must have one dimension to be a profile"
    newmeta = mda.Metadata()
    newmeta.experiment_type = "profile"
    if parent:
        newmeta.parent_meta = parent.metadata
    if process:
        newmeta.process = process
    xinfo = (arr.shape[0], pixelunit, pixelsize)
    newmeta["data_axes"] = mda.gen_profile_axes(xinfo)
    newmeta = mda.Metadata(newmeta)
    return Profile(arr, newmeta)


class Profile(TEMDataSet):
    """Represents a simple 1D intensity array"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__x = "x"

    def set_scale(self, pixelsize, pixelunit):
        """Set the size and unit of the x axis"""
        self.set_axis_scale(self.x, pixelsize, pixelunit)

    @property
    def x(self):
        return self.__x

    @property
    def bin_axis(self):
        """Returns an x-axis array in pixel units"""
        return np.arange(self.length)

    @property
    def x_axis(self):
        """Returns an x-axis data array in the right units"""
        return self.bin_axis*self.pixelsize

    @property
    def pixelsize(self):
        return self._get_axis_prop(self.x, "scale")

    @property
    def pixelunit(self):
        return self._get_axis_prop(self.x, "unit")

    @property
    def length(self):
        return self._get_axis_prop(self.x, "bins")

    def plot(self, filename=None, axis=None):
        """
        Wrapper for plot_1D, returns axis and plot object.

        Optionally saves to file
        """
        ax, prof = pl.plot_profile(self, axis)
        if filename:
            ax.figure.savefig(filename)
        return ax, prof

    def to_hspy(self, filepath=None):
        """
        Convert to a hyperspy dataset and potentially save to file

        Metadata not related to linear axes scales and units is lost.
        """
        hs = super().to_hspy()
        hsim = hs.signals.Signal1D(self.data)
        hsim.axes_manager[0].name = self.x
        hsim.axes_manager[self.x].units = self._get_axis_prop(self.x, "unit")
        hsim.axes_manager[self.x].scale = self._get_axis_prop(self.x, "scale")
        # hsim.original_metadata = self.metadata # does not work
        if filepath:
            hsim.save(str(Path(filepath)))
        return hsim

    def to_excel(self, filepath):
        """Write out the profile data to an excel file with two columns"""
        dt = self.data
        x = np.arange(self.length)*self.pixelsize
        adt = np.vstack([x, dt]).T
        gg = pd.DataFrame(adt, columns=[f"x ({self.pixelunit})",
                                        f"Intensity (a.u.)"])
        gg.to_excel(str(Path(filepath)))


def _correct_crop_window(x1, y1, x2, y2, imw, imh, nearest_power=False):
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    if nearest_power:
        w = x2-x1
        h = y2-y1
        toround = max(abs(w), abs(h))
        n1 = np.floor(np.log2(toround))
        n2 = n1+1
        mid = (2**n1+2**n2)//2
        if toround < mid:
            s = 2**n1
        else:
            s = 2**n2
        x2 = int(x1+s*np.sign(w))
        y2 = int(y1+s*np.sign(h))
    # we need to flip indices if order is not correct
    if x2 < x1:
        x2, x1 = x1, x2
    if y2 < y1:
        y2, y1 = y1, y2
    # checkvalidity of indices
    for i in [x1, x2]:
        if i < 0 or i > imw:
            raise ValueError(f"{i} is out of bounds")
    for i in [y1, y2]:
        if i < 0 or i > imh:
            raise ValueError(f"{i} is out of bounds")
    return x1, y1, x2, y2


def create_new_image(arr, pixelsize, pixelunit, parent=None, process=None):
    """
    Create a GeneralImage object from an array, pixelsize and pixelunit

    The "experiment_type" will be set to "modified"

    Parameters
    ----------
    arr : array-like, 2D
        The array representing an image. The first axis is the rows or y
        axis, the second axis is the columns or x axis.
    pixelsize : float
        The size of a pixel in the correct units
    pixelunit : str
        The size unit of the image
    parent : TEMDataSet, optional
        If the image is derived from another image (e.g. a frame from a stack)
        then the parent is provided as an argument. The parent metadata will
        be stored in the child metadata under "parent_metadata"
    process : object, optional
        Some description of how the dataset was obtained. Could be a
        dictionary, string, ... and it will be stored under the key "process"

    Returns
    -------
    image : GeneralImage
        The general image object with some easy image processing tools
    """
    arr = np.array(arr)
    assert arr.ndim == 2, "The array must have two dimensions to be an image"
    newmeta = mda.Metadata()
    newmeta.experiment_type = "modified"
    if parent:
        newmeta.parent_meta = parent.metadata
    if process:
        newmeta.process = process
    xinfo = (arr.shape[1], pixelunit, pixelsize)
    yinfo = (arr.shape[0], pixelunit, pixelsize)
    newmeta["data_axes"] = mda.gen_image_axes(xinfo, yinfo)
    newmeta = mda.Metadata(newmeta)
    return GeneralImage(arr, newmeta)


class GeneralImage(TEMDataSet):
    """
    Represents a single 2D image
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.experiment_type == "image":
            # for TEM images
            self.__x = "x"
            self.__y = "y"
        elif self.experiment_type == "scan_image":
            # for STEM images
            self.__x = "scan_x"
            self.__y = "scan_y"
        elif self.experiment_type == "modified":
            # for images created from an array
            self.__x = "x"
            self.__y = "y"
        else:
            raise TypeError("The dataset is not a recognized image")

    def change_dtype(self, dtype):
        self.data = self.data.astype(dtype)

    def _compatible(self, other):
        """Check whether images are compatible for adding/multiplying"""
        if (self.width == other.width and
           self.height == other.height and
           self.pixelsize == other.pixelsize and
           self.pixelunit == other.pixelunit):
            return True
        else:
            raise ValueError("The images are not compatible")

    def __add__(self, other):
        if self._compatible(other):
            dt = self.data+other.data
            process = "Added images element-wise"
            meta = [self.metadata, other.metadata]
            dummyparent = TEMDataSet(dt, meta)
            return create_new_image(dt, self.pixelsize, self.pixelunit,
                                    parent=dummyparent, process=process)

    def __sub__(self, other):
        if self._compatible(other):
            dt = self.data-other.data
            process = ("Subtracted image 1 (right) from image 0 (left) "
                       "element-wise")
            meta = [self.metadata, other.metadata]
            dummyparent = TEMDataSet(dt, meta)
            return create_new_image(dt, self.pixelsize, self.pixelunit,
                                    parent=dummyparent, process=process)

    def __mul__(self, other):
        if self._compatible(other):
            dt = np.multiply(self.data, other.data)
            process = "Multiplied images element-wise"
            meta = [self.metadata, other.metadata]
            dummyparent = TEMDataSet(dt, meta)
            return create_new_image(dt, self.pixelsize, self.pixelunit,
                                    parent=dummyparent, process=process)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def pixelsize(self):
        return self._get_axis_prop(self.x, "scale")

    @property
    def pixelunit(self):
        return self._get_axis_prop(self.x, "unit")

    @property
    def width(self):
        return self._get_axis_prop(self.x, "bins")

    @property
    def height(self):
        return self._get_axis_prop(self.y, "bins")

    def set_scale(self, scale, unit):
        """Set the image pixel scale and unit"""
        self.set_axis_scale(self.x, scale, unit)
        self.set_axis_scale(self.y, scale, unit)

    @property
    def dimensions(self):
        """
        Returns real size of image
        (width, width unit, height, height unit)
        """
        xs = self._get_axis_prop(self.x, "scale")
        ys = self._get_axis_prop(self.y, "scale")
        xb = self._get_axis_prop(self.x, "bins")
        yb = self._get_axis_prop(self.y, "bins")
        xu = self._get_axis_prop(self.x, "unit")
        yu = self._get_axis_prop(self.y, "unit")
        return (xs*xb, xu, ys*yb, yu)

    def intensity_profile(self, x1, y1, x2, y2, w=1):
        """
        Return an array of summed intensities along a line

        It works by rotating the image so the line vector is horizontal.
        The rotation is calculated with scipy.ndimage.rotate, filling up
        the boundary with the mean of the image (in case of profiles on
        the edge) and 3rd order spline interpolation.
        """
        assert (x1 >= 0 and x1 < self.width and
                x2 >= 0 and x2 < self.width and
                y1 >= 0 and y1 < self.height and
                y2 >= 0 and y2 < self.height), "Some index is out of bounds"
        img = self.data.copy()
        rt = algutil.rotangle(x1, y1, x2, y2)
        cst = img.mean()
        # rotated image
        rim = ndi.rotate(img, rt, cval=cst)
        # get rotated coordinates of the ends of the line
        le = algutil.distance(x1, y1, x2, y2)
        d = (x2-x1)/le
        b = (y2-y1)/le
        midxold = img.shape[1]/2
        midyold = img.shape[0]/2
        midxnew = rim.shape[1]/2
        midynew = rim.shape[0]/2
        x1r = int(round(d*(x1-midxold)+b*(y1-midyold) + midxnew))
        x1r = max(0, x1r)  # take into consideration out of bounds
        y1r = int(round(-b*(x1-midxold)+d*(y1-midxold) + midynew))
        x2r = int(round(d*(x2-midxold)+b*(y2-midxold) + midxnew))
        x2r = min(img.shape[1], x2r)  # take into consideration out of bounds
        # y2r = int(round(-b*(x2-midxold)+d*(y2-midxold) + midynew))
        # select the "window" around the line and sum over the height
        # take into consideration out of bounds
        yb1 = max((y1r-w//2), 0)
        yb2 = min((y1r+w//2+1), img.shape[0])
        lpf = rim[yb1:yb2, x1r:x2r].sum(axis=0)
        # create line_profile object
        process = f"Intensity line profile from ({x1}, {y1}) to ({x2}, {y2})"
        return create_profile(lpf, self.pixelsize, self.pixelunit,
                              parent=self, process=process)

    def _create_child_image(self, inplace, *args, **kwargs):
        """Wrapper for create_new_image which considers inplace"""
        newimg = create_new_image(*args, **kwargs)
        if inplace:
            self.__init__(newimg.data, newimg.metadata)
        else:
            return newimg

    def measure(self, x1, y1, x2, y2):
        """Return the length and angle in real units between two points"""
        rt = algutil.rotangle(x1, y1, x2, y2)
        le = algutil.distance(x1, y1, x2, y2)
        return (le*self.pixelsize, self.pixelunit), (rt, DEGREE)

    def crop(self, x1, y1, x2, y2, inplace=False, nearest_power=False):
        """
        Select a rectangle defined by two points and return a new image

        Parameters
        ----------
        x1 : int
            X-coordinate of first rectangle corner
        y1: int
            Y-coordinate of first rectangle corner
        x2 : int
            X-coordinate of second rectangle corner
        y2: int
            Y-coordinate of second rectangle corner
        inplace: bool, optional
            Whether to replace the image or return a new one.
        nearest_power : bool, optional
            Whether to crop to a square with side length the nearest power
            of 2. (x1, y1) is taken directly and x2, y2 is rounded.
            The largest dimension is taken for the rounding.

        Returns
        -------
        img : GeneralImage or None if inplace
            cropped image
        """
        x1, y1, x2, y2 = _correct_crop_window(x1, y1, x2, y2,
                                              self.width, self.height,
                                              nearest_power)
        data = self.data[y1:y2, x1:x2].copy()
        process = f"Cropped between x={x1}-{x2} and y={y1}-{y2}"
        return self._create_child_image(inplace, data, self.pixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def rebin(self, factor, inplace=False, **kwargs):
        """Rebin an image by a certain factor"""
        data = imf.bin2(self.data, factor, **kwargs)
        process = f"Binned by a factor of {factor}"
        newpixelsize = self.pixelsize*factor
        return self._create_child_image(inplace, data, newpixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def linscale(self, min, max, inplace=False):
        """
        Change the minimum and maximum intensity values

        The dtype of the image remains the same
        """
        data = imf.normalize_convert(self.data, min, max,
                                     dtype=self.data.dtype)
        process = f"Scaled intensity between {min} and {max}"
        return self._create_child_image(inplace, data, self.pixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def calc_strains(self, xx1, yy1, xx2, yy2, r, angle=0, edge_blur=5):
        """
        Calculate strain with geometric phase analysis (gpa) [1].

        Is only supposed to be used on HRTEM images and will produce
        nonsensical output for other images. It is recommended to first
        visualize the FFT to determine the (x,y) coordinates of 2
        reflections. Returns the 4 images with different strain components.

        Parameters
        ----------
        xx1 : int
            x-coordinate of reflection 1 in FFT (pixel units)
        yy1 : int
            y-coordinate of reflection 1 in FFT (pixel units)
        xx2 : int
            x-coordinate of reflection 2 in FFT (pixel units)
        yy2 : int
            y-coordinate of reflection 2 in FFT (pixel units)
        r : float
            Size of the circular mask to place over the reflection
        angle : float, optional
            Angle of the x-axis of the coordinate system in which you want
            the strains expressed. Calculated with respect to the image
            x-axis. Expressed in degrees. Default is 0.
        edge_blur : int, optional
            Amount of blurring of the circular masks. See the method
            add_circle in the Mask class.

        Returns
        -------
        im_exx : GeneralImage
            The epsilon_xx strain component (extension in x)
        im_eyy : GeneralImage
            The epsilon_yy strain component (extension in y)
        im_exy : GeneralImage
            Epsilon_yx = (epsilon_yx), the shear strain
        im_oxy : GeneralImage
            omega_xy = -omega_yx, the rotation component

        Notes
        -----
        See the details in [1] for additional information about the variables
        and way the strain is calculated.

        References
        ----------
        [1] M. Hÿtch, E. Snoeck, R. Kilaas, Ultramicroscopy 74 (1998) 131–146.
        """
        xx1 = int(xx1)
        xx2 = int(xx2)
        yy1 = int(yy1)
        yy2 = int(yy2)

        def _calc_derivative(arr, axis):
            """
            Calculate the derivative of a phase image, see appendix D.
            """
            s1 = np.exp(-1j*arr)
            s2 = np.exp(1j*arr)
            d1 = np.diff(s2, axis=axis)  # will have 1 axis reduced by 1 pix
            nd = np.min(d1.shape)
            dP1x = s1[:nd, :nd]*d1[:nd, :nd]
            return dP1x.imag
        # calculate the fft and mask it only on one side
        fft = self.fft
        m1 = fft.create_mask()
        m2 = fft.create_mask()
        m1.add_circle(xx1, yy1, r, edge_blur=edge_blur, mirrored=False)
        m2.add_circle(xx2, yy2, r, edge_blur=edge_blur, mirrored=False)
        fft_m1 = fft.apply_mask(m1)
        fft_m2 = fft.apply_mask(m2)
        # calculate complex valued ifft
        ifft_m1 = np.fft.ifft2(fft_m1.data)
        ifft_m2 = np.fft.ifft2(fft_m2.data)
        # raw phase images
        phifft_m1 = np.angle(ifft_m1)
        phifft_m2 = np.angle(ifft_m2)
        # calculate g's in 1/pixel units
        w = fft.width
        h = fft.height
        gx1 = (xx1-w//2)/w
        gy1 = (yy1-h//2)/h
        gx2 = (xx2-w//2)/w
        gy2 = (yy2-h//2)/h
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        # calculate term to subtract: -2*pi*(g.r)
        cor1 = -2*np.pi*(gx1*X+gy1*Y)
        cor2 = -2*np.pi*(gx2*X+gy2*Y)
        # corrected phase images. Not normalizing to -pi:pi doesn't matter
        P1 = phifft_m1+cor1
        P2 = phifft_m2+cor2
        # calculate derivatives
        dP1dx = _calc_derivative(P1, 1)
        dP1dy = _calc_derivative(P1, 0)
        dP2dx = _calc_derivative(P2, 1)
        dP2dy = _calc_derivative(P2, 0)
        # calculate strains, first we need lattice points a1 and a2
        G = np.array([[gx1, gx2],
                      [gy1, gy2]])
        [a1x, a2x], [a1y, a2y] = np.linalg.inv(G.T)
        # the strain components
        exx = -1/(2*np.pi)*(a1x*dP1dx+a2x*dP2dx)
        exy = -1/(2*np.pi)*(a1x*dP1dy+a2x*dP2dy)
        eyx = -1/(2*np.pi)*(a1y*dP1dx+a2y*dP2dx)
        eyy = -1/(2*np.pi)*(a1y*dP1dy+a2y*dP2dy)
        # rotate the tensor
        if angle != 0:
            st = np.sin(angle/360*np.pi*2)
            ct = np.cos(angle/360*np.pi*2)
            a = exx
            b = exy
            c = eyx
            d = eyy
            nexx = a*ct**2+b*st*ct+c*ct*st+d*st**2
            nexy = -a*ct*st+b*ct**2-c*st**2+d*ct*st
            neyx = -a*ct*st-b*st**2+c*ct**2+d*st*ct
            neyy = a*st**2-b*st*ct-c*ct*st+d*ct**2
            exx, exy, eyx, eyy = nexx, nexy, neyx, neyy
        # create images of the 4 important components
        process = (f"Calculated a GPA map. Reflections: ({xx1}, {yy1}), "
                   f"({xx2}, {yy2}). Radius: {r}. Edge_blur: {edge_blur}."
                   f" Angle: {angle} {DEGREE}. Epsilon_xx component.")
        im_exx = self._create_child_image(False, exx, self.pixelsize,
                                          self.pixelunit,
                                          parent=self, process=process)
        process = (f"Calculated a GPA map. Reflections: ({xx1}, {yy1}), "
                   f"({xx2}, {yy2}). Radius: {r}. Edge_blur: {edge_blur}."
                   f" Angle: {angle} {DEGREE}. Epsilon_yy component.")
        im_eyy = self._create_child_image(False, eyy, self.pixelsize,
                                          self.pixelunit,
                                          parent=self, process=process)
        process = (f"Calculated a GPA map. Reflections: ({xx1}, {yy1}), "
                   f"({xx2}, {yy2}). Radius: {r}. Edge_blur: {edge_blur}."
                   f" Angle: {angle} {DEGREE}. Epsilon_xy component.")
        eps_xy = (exy+eyx)/2
        im_exy = self._create_child_image(False, eps_xy, self.pixelsize,
                                          self.pixelunit,
                                          parent=self, process=process)
        process = (f"Calculated a GPA map. Reflections: ({xx1}, {yy1}), "
                   f"({xx2}, {yy2}). Radius: {r}. Edge_blur: {edge_blur}."
                   f" Angle: {angle} {DEGREE}. Omega_xy component.")
        omega_xy = (exy-eyx)/2
        im_oxy = self._create_child_image(False, omega_xy, self.pixelsize,
                                          self.pixelunit,
                                          parent=self, process=process)
        return im_exx, im_eyy, im_exy, im_oxy

    def histogram(self, bins=256):
        """
        Get intensity histogram of the image

        Parameters
        ----------
        bins : int, optional
            The number of bins, by default this is 256 corresponding
            to an 8-bit image

        Returns
        -------
        values : 1D array
            The counts in each bin
        bin_edges : 1D array
            The edges of the bins.
        """
        return np.histogram(self.data, bins=256)

    def plot_histogram(self, bins=256, **kwargs):
        """
        Shows the histogram of the image
        """
        bins = int(bins)
        val, bns = self.histogram(bins=bins)
        x = (bns[:-1]+bns[1:])/2
        w = np.diff(bns)[0]
        ax, im = pl.plot_histogram(x, val, w, **kwargs)
        return ax, im

    def plot_interactive(self):
        """Plot the image in an interactive way"""
        pl.plot_image_interactive(self)

    def plot(self, filename=None, **kwargs):
        """
        Plot the image

        Parameters
        ----------
        filename : str, optional
            file to save the image to. Default is None.

        Other parameters
        ----------------
        show_fig : bool, optional
            show the figure. Default is True.
        scale_bar : bool, optional
            Put a scalebar on the image, default is True

        More details in passed to plottingtools.plot_image

        Returns
        -------
        figure : matplotlib Figure object
        axis : matplotlib axis object
        """
        ax, im = pl.plot_image(self, **kwargs)
        if filename:
            pre, _ = os.path.splitext(filename)
            self.metadata.to_file(str(Path(pre+"_meta.json")))
            ax.figure.savefig(filename)
        return ax, im

    def apply_filter(self, filt, *args, inplace=False, **kwargs):
        """
        Applies a filter function to the image and returns new image

        Parameters
        ----------
        filt : callable
            Function to apply to the image
        inplace : bool
            Whether the data in the current object is overwritten. Default is
            False.

        Other parameters
        ----------------
        *args, **kwargs : all additional arguments are passed without check
            to the filt function.
        """
        data = filt(self.data.copy(), *args, **kwargs)
        assert isinstance(data, np.ndarray), ("The output is not an array, "
                                              "the function is probably not "
                                              "a filter.")
        assert data.ndim == 2, ("The output is not a 2D array, the "
                                "function is probably not a filter.")
        if (data.shape[0] != self.data.shape[0] or
           data.shape[1] != self.data.shape[1]):
            logger.warning("The output image is of a different size than the "
                           "original. The scale may no longer be correct.")
        process = {
                    "filter": {
                        "name": filt.__name__,
                        "arguments": list(args),
                        "keyword_arguments": {**kwargs}
                    }
                }
        logger.warning("The pixel size and unit of the original image was "
                       "used. The scale may no longer be correct. Please "
                       "verify and use set_scale.")
        pixelunit = self.pixelunit
        pixelsize = self.pixelsize
        newimage = create_new_image(data, pixelsize, pixelunit,
                                    parent=self, process=process)
        if inplace:
            logger.warning("Data in object will be overwritten")
            self.__init__(newimage.data, newimage.metadata)
        else:
            return newimage

    def get_fft(self):
        return self.fft

    @property
    def fft(self):
        """
        Calculate the FFT of the image and return an FFT object
        """
        assert self.data.shape[1] == self.data.shape[0], ("Image size must "
                                                          "be square")
        data = np.fft.fft2(self.data)
        data_shift = np.fft.fftshift(data)
        process = "Fast-fourrier-transform"
        newmeta = mda.Metadata()
        newmeta.experiment_type = "FFT"
        newmeta.parent_meta = self.metadata
        newmeta.process = process
        fft_pixel = 1/(self.pixelsize*self.width)
        xinfo = (data_shift.shape[1], f"1/{self.pixelunit}", fft_pixel)
        yinfo = (data_shift.shape[0], f"1/{self.pixelunit}", fft_pixel)
        newmeta["data_axes"] = mda.gen_image_axes(xinfo, yinfo)
        newmeta = mda.Metadata(newmeta)
        return FFT(data_shift, newmeta)

    def to_hspy(self, filepath=None):
        """
        Return the image as a hyperspy dataset

        If a file path is provided, it is also saved as a .hspy file.
        Metadata not related to axes is not saved.
        """
        hs = super().get_hs()
        hsim = hs.signals.Signal2D(self.data)
        hsim.axes_manager[0].name = self.x
        hsim.axes_manager[self.x].units = self._get_axis_prop(self.x, "unit")
        hsim.axes_manager[self.x].scale = self._get_axis_prop(self.x, "scale")
        hsim.axes_manager[1].name = self.y
        hsim.axes_manager[self.y].units = self._get_axis_prop(self.y, "unit")
        hsim.axes_manager[self.y].scale = self._get_axis_prop(self.y, "scale")
        # hsim.original_metadata = self.metadata # does not work
        if filepath:
            hsim.save(str(Path(filepath)))
        return hsim

    def save(self, filename, dtype=None, **kwargs):
        """
        Exports the raw image as a png.

        The kwargs are passed to normalize_convert in image_filters

        Parameters
        ----------
        filename : str
            Relative or absolute path to file
        dtype : numpy.dtype, optional
            Output data type. Defaults to data type in image.
            If the image dtype is float, this will not work for most
            data formats! You must then select an integer dataset.

        Other parameters
        ----------------
        min : int, optional
            the value in the img to map to the minimum. Everything
            below is set to minimum. Defaults to the minimum value
            in the array.
        max : int, optional
            the value in the img to map to the maximum. Everything
            above is set to maximum. Defaults to the maximum value
            in the array.
        """
        if dtype is None:
            dtype = self.data.dtype
        frame = imf.normalize_convert(self.data, dtype=dtype, **kwargs)
        img = Image.fromarray(frame)
        img.save(str(Path(filename)))
        pre, _ = os.path.splitext(filename)
        self.metadata.to_file(str(Path(pre+"_meta.json")))


class Mask(object):
    """A class representing a binary mask for FFTs and images"""

    STDEDGEBLUR = 3

    def __init__(self, img, negative=True):
        """Provide an img or FFT object, the mask will adopt size and units"""
        self.parent = img
        w = img.width
        h = img.height
        self.__negative = negative
        if self.__negative:
            arr = np.zeros((h, w))
        else:
            arr = np.ones((h, w))
        self.data = arr
        self.pixelsize = img.pixelsize
        self.pixelunit = img.pixelunit
        if w % 2 == 0:
            self.even = True
        else:
            self.even = False
        self.metadata = mda.Metadata()
        self.metadata.process = (f"Initialized {self._bool_string(negative)} "
                                 f"mask of size {w}x{h}")

    def _record_operation(self, process):
        newmeta = mda.Metadata()
        newmeta.parent_meta = self.metadata
        newmeta.process = str(process)
        self.metadata = newmeta

    def invert(self):
        """Invert the mask"""
        self.data = -1.*self.data+1.
        self.__negative = (not self.negative)
        self._record_operation("Inverted mask")

    @property
    def negative(self):
        """Returns whether it is a negative or positive mask"""
        return self.__negative

    @property
    def width(self):
        return self.data.shape[0]

    @property
    def height(self):
        return self.data.shape[1]

    @property
    def mx(self):
        """Middle x, central spot of the FFT"""
        # for even: w/2, for odd (w-1)/2
        return self.width//2

    @property
    def my(self):
        """Middle y, central spot of the FFT"""
        # for even: w/2, for odd (w-1)/2
        return self.height//2

    @property
    def diameter(self):
        """Length from center to corner"""
        return np.sqrt((self.mx)**2+(self.my)**2)

    @property
    def history(self):
        return self.metadata.print_history()

    def set_mask(self, arr):
        """Set the mask to be a certain numpy array. Fails if the wrong size"""
        h = self.height
        w = self.width
        if arr.shape[0] == h and arr.shape[1] == w and arr.ndim == 2:
            self.data = arr
            self._record_operation("Set an image as the mask")
        else:
            raise ValueError("The mask image is of the wrong size or has "
                             " the wrong number of dimensions")

    def _get_circle_template(self, r, edge_blur, negative):
        """
        Return a binary blurred circle image

        Always odd-shaped so it can be centered on a pixel.

        Parameters
        ----------
        r : float
            Radius of the circle in pixels
        edge_blur : int
            Size of the blurring region on the edge. The blur is achieved
            with a Gaussian filter. The parameter influences both the
            kernel size and the sigma.
        negative : bool
            If True, the inside of the circle is 0 and the outside 1.
            Otherwise it is the other way around.

        Returns
        -------
        arr : numpy array
            The image of the circle
        """
        edge_blur = int(edge_blur)
        s = int(r+edge_blur*2)
        # distance matrix
        x = np.arange(-s, s+1, 1)  # always odd so distance 0 is center
        y = np.arange(-s, s+1, 1)
        xv, yv = np.meshgrid(x, y)
        mask_radius = np.sqrt(np.square(xv) + np.square(yv))
        arr = (mask_radius < r)*1.
        if edge_blur < 0:
            raise ValueError("Expecting a positive value for edge_blur")
        if edge_blur != 0:
            arr = imf.gauss_filter(arr, ks=edge_blur*4+1, sig=edge_blur,
                                   cval=0)
        if negative:
            arr = -1.*arr  # negative, we subtract
        return arr

    def _get_square_template(self, r, edge_blur, negative):
        """
        Return a binary blurred square image

        Parameters
        ----------
        r : int
            Half the sidelength
        edge_blur : int
            Size of the blurring region on the edge. The blur is achieved
            with a Gaussian filter. The parameter influences both the
            kernel size and the sigma.
        negative : bool
            If True, the inside of the square is 0 and the outside 1.
            Otherwise it is the other way around.

        Returns
        -------
        arr : numpy array
            The image of the square
        """
        r = int(r)
        edge_blur = int(edge_blur)
        s = 2*r+edge_blur*2+1  # make it odd
        arr = np.zeros((s, s))
        arr[s//2-r:s//2+r+1, s//2-r:s//2+r+1] = 1
        if edge_blur < 0:
            raise ValueError("Expecting a positive value for edge_blur")
        if edge_blur != 0:
            arr = imf.gauss_filter(arr, ks=edge_blur*4+1, sig=edge_blur,
                                   cval=0)
        if negative:
            arr = -1.*arr
        return arr

    def _get_band_template(self, rin, rout, edge_blur, negative):
        """
        Return a binary blurred bandpass filter

        Parameters
        ----------
        rin : int
            inner diameter
        rout : int
            outer diameter
        edge_blur : int
            Size of the blurring region on the edge. The blur is achieved
            with a Gaussian filter. The parameter influences both the
            kernel size and the sigma.
        negative : bool
            If True, the inside of the square is 0 and the outside 1.
            Otherwise it is the other way around.

        Returns
        -------
        arr : numpy array
            The image of the bandpass filter
        """
        assert rin < rout, "Inner radius must be smaller than outer radius"
        edge_blur = int(edge_blur)
        s = int(rout+edge_blur*2)
        # distance matrix outter matrix
        x = np.arange(-s, s+1, 1)
        y = np.arange(-s, s+1, 1)
        xv, yv = np.meshgrid(x, y)
        mask_radius = np.sqrt(np.square(xv) + np.square(yv))
        arr = np.logical_and(mask_radius < rout, mask_radius > rin)*1.
        if edge_blur < 0:
            raise ValueError("Expecting a positive value for edge_blur")
        if edge_blur != 0:
            arr = imf.gauss_filter(arr, ks=edge_blur*4+1, sig=edge_blur,
                                   cval=0)
        if negative:
            arr = -1.*arr
        return arr

    def _get_feature_props(self, edge_blur=None, negative=None):
        """
        Return default edge_blur and negative values
        """
        if edge_blur is None:
            edge_blur = Mask.STDEDGEBLUR
        if negative is None:
            if self.negative:
                negative = False
            else:
                negative = True
        return edge_blur, negative

    def _add_mask_feature(self, x, y, feature, refine=False, window=15):
        """
        Add a certain template feature onto the mask

        Parameters
        ----------
        x : int
            x position in pixels where center of feature should be placed
        y : int
            y position in pixels where center of feature should be placed
        feature : numpy array
            the mask feature to be added, e.g. a circle or square
        refine : boolean
            place the feature not on the exact pixel but on the maximum within
            a square the size of window
        window : int
            The size of the square around (x,y) to search for the maximum

        Returns
        -------
        x, y : int
            Refined positions

        Notes
        -----
        Only features with odd width and height will be properly centered
        """
        h, w = feature.shape
        xs = x-w//2
        ys = y-h//2
        if refine:
            window = int(window)
            wys = max(y-window//2, 0)
            wye = min(y+window//2+1, self.height)
            wxs = max(x-window//2, 0)
            wxe = min(x+window//2+1, self.width)
            # logger.debug(f"Window between x=[{wxs},{wxe}), y=[{wys},{wye})")
            search_window = np.abs(self.parent.data[wys:wye,
                                                    wxs:wxe])
            dy, dx = np.where(search_window == np.max(search_window))
            dx = dx[0]
            dy = dy[0]
            # logger.debug(f"Maximum in the window at dx={dx}, dy={dy}")
            if (dx == 0 or dy == 0 or dx == 2*window//2 or dy == 2*window//2):
                logger.warning("The maximum was found on the edge of the "
                               "search window, it's possible an incorrect "
                               "coordinate was reached.")
            x = wxs+dx
            y = wys+dy
            logger.warning(f"Refined coordinates: ({x}, {y})")
            xs = x-w//2
            ys = y-h//2
        temp = self.data[ys:ys+h, xs:xs+w]+feature
        temp[temp > 1] = 1
        temp[temp < 0] = 0
        self.data[ys:ys+h, xs:xs+w] = temp
        return x, y

    def _get_opposite_coordinates(self, x, y):
        """
        Return the coordinates opposite of the central pixel

        For even shape the central pixel is (w/2, h/2)
        For odd shape the central pixel is ((w-1)/2, (h-1)/2)
        """
        x2 = 2*self.mx-x
        y2 = 2*self.my-y
        return x2, y2

    def _bool_string(self, b, s1="negative", s2="positive"):
        """Shortcut for returning a string based on a boolean"""
        if b:
            return s1
        else:
            return s2

    def add_circle(self, x, y, r, edge_blur=None, negative=None,
                   mirrored=True, record=True, **kwargs):
        """
        Add a circle feature in the mask

        By default, the circle is repeated on the other side of the center.

        Parameters
        ----------
        x : int
            x position in pixels where center of circle should be placed
        y : int
            y position in pixels where center of circle should be placed
        r : int
            radius in pixels of circle
        edge_blur : int, optional
            Fudge factor of the edge, influencing the Gaussian filter.
            By default this will be 3.
        negative : bool
            If True, center is black and edge is white. By default, if the
            mask is negative, the feature is positive, else the other way
            around
        mirrored : bool
            If True, the feature is repeated on the opposite side of the
            middle. Default is true.
        record : bool
            Whether to record the addition

        Other parameters
        ----------------
        refine : boolean
            place the feature not on the exact pixel but on the maximum within
            a square the size of window. Default is False.
        window : int
            The size of the square to search for the maximum. Default is 15.
        """
        edge_blur, negative = self._get_feature_props(edge_blur=edge_blur,
                                                      negative=negative)
        circ = self._get_circle_template(r, edge_blur, negative)
        x, y = self._add_mask_feature(x, y, circ, **kwargs)
        # add opposite circle also
        if mirrored:
            x2, y2 = self._get_opposite_coordinates(x, y)
            self._add_mask_feature(x2, y2, circ, **kwargs)
        if record:
            operation = (f"Added {self._bool_string(negative)} circle. "
                         f"Radius: {r}. Position: ({x}, {y}). "
                         f"Mirrored: {mirrored}. Edge_blur: {edge_blur}.")
            self._record_operation(operation)

    def add_array1D(self,  x, y, r, edge_blur=None, negative=None, maximum=20,
                    mirrored=True, **kwargs):
        """
        Add a 1D array of circles over the mask

        The center spot is not blanked.

        Parameters
        ----------
        x : int
            x component of basis vector
        y : int
            y component of basis vector
        r : int
            radius of each circle
        edge_blur : int, optional
            Fudge factor of the edge, influencing the Gaussian filter.
            By default this will be 3.
        negative : bool, optional
            If True, between the circles is black and edge is white.
            By default, if the mask is negative, the feature is positive,
            else the other way around
        maximum : int, optional
            By default the minimum size of the array will be calculated but
            this may give a huge number of lattice positions. Maximum puts a
            cap on this: the width of the array can be maximally 2*maximum
            lattice points.
        mirrored : bool, optional
            Whether to repeat array on the other side of the middle as well.
        """
        edge_blur, negative = self._get_feature_props(edge_blur=edge_blur,
                                                      negative=negative)
        # calculate the number of required units
        dx = int(x-self.mx)
        dy = int(y-self.my)
        dl = np.sqrt(dx**2+dy**2)
        numb = int(min(self.diameter//dl, maximum))
        for i in range(numb):
            try:
                xt = x+i*dx
                yt = y+i*dx
                if (xt > 0 and xt < self.width and
                   yt > 0 and yt < self.height):
                    self.add_circle(xt, yt, r, edge_blur=edge_blur,
                                    negative=negative, mirrored=mirrored,
                                    record=False, **kwargs)
            except ValueError:
                # out of bounds, stop
                break
        operation = (f"Added {self._bool_string(negative)} 1D array. "
                     f"Radius: {r}. Start position: ({x}, {y}). "
                     f"Mirrored: {mirrored}. Edge_blur: {edge_blur}.")
        self._record_operation(operation)

    def _get_2D_array(self, x1, y1, x2, y2, r, edge_blur, incl_zero=False,
                      maximum=30):
        # all points within this edge are removed
        edge = r+edge_blur*2
        # for ease of use
        w = self.width
        h = self.height
        mx = self.mx  # middle x
        my = self.my  # middle y
        # check the dimensions the grid needs to have
        dx1 = x1-mx
        dy1 = y1-my
        dl1 = np.sqrt(dx1**2+dy1**2)
        dx2 = x2-mx
        dy2 = y2-my
        dl2 = np.sqrt(dx2**2+dy2**2)
        dl3 = np.sqrt((dx2-dx1)**2+(dy2-dy1)**2)
        dl4 = np.sqrt((dx2+dx1)**2+(dy2+dy1)**2)
        dl = min(dl1, dl2, dl3, dl4)
        # safe estimate of half of the width of the grid
        numb = int(min(self.diameter//dl, maximum))
        # create the coordinates
        x = np.arange(-numb, numb+1)
        y = np.arange(-numb, numb+1)
        X, Y = np.meshgrid(x, y)
        coords = np.vstack([X.ravel(), Y.ravel()])
        if not incl_zero:
            # remove the point 0,0
            no_zero = np.invert((coords == 0).all(axis=0))
            coords = coords[:, no_zero]
        # create the transformation matrix
        transfo = np.array([[x1-mx, x2-mx], [y1-my, y2-my]])
        # transformed coordinates
        co_tr = np.dot(transfo, coords)
        co_tr[0] = co_tr[0]+mx
        co_tr[1] = co_tr[1]+my
        pts = co_tr.T.astype(int)
        # remove out of bounds points
        xb = np.logical_and(pts[:, 0] > edge, pts[:, 0] < w-edge)
        yb = np.logical_and(pts[:, 1] > edge, pts[:, 1] < h-edge)
        ex = np.logical_and(xb, yb)
        pts = pts[ex]
        return pts

    def add_array2D(self, x1, y1, x2, y2, r, edge_blur=None, negative=None,
                    maximum=30):
        """
        Add a 2D array of circles over the mask

        Parameters
        ----------
        x1 : int
            x component of first basis vector
        y1 : int
            y component of first basis vector
        x2 : int
            x component of second basis vector
        y2 : int
            y component of second basis vector
        r : int
            radius of each circle
        edge_blur : int, optional
            Fudge factor of the edge, influencing the Gaussian filter.
            By default this will be 3.
        negative : bool, optional
            If True, between the circles is black and edge is white.
            By default, if the mask is negative, the feature is positive,
            else the other way around
        maximum : int, optional
            By default the minimum size of the array will be calculated but
            this may give a huge number of lattice positions. Maximum puts a
            cap on this: the width of the array can be maximally 2*maximum
            lattice points.
        """
        edge_blur, negative = self._get_feature_props(edge_blur=edge_blur,
                                                      negative=negative)
        pts = self._get_2D_array(x1, y1, x2, y2, r, edge_blur, incl_zero=False,
                                 maximum=maximum)
        # put a circle on all transformed coordinate pairs
        for i in pts:
            try:
                xco = int(i[0])
                yco = int(i[1])
                self.add_circle(xco, yco, r, edge_blur=edge_blur,
                                negative=negative, mirrored=False,
                                record=False)
            except ValueError:
                pass
        operation = (f"Added {self._bool_string(negative)} 2D array. "
                     f"Radius: {r}. Positions: ({x1},{y1}), ({x2},{y2}). "
                     f"Edge_blur: {edge_blur}.")
        self._record_operation(operation)

    def add_band(self, r1, r2, edge_blur=None, negative=None):
        """
        Add a band-pass filter on the mask

        Parameters
        ----------
        r1 : int
            Inner radius of the filter in pixels
        r2 : int
            Outer radius of the filter in pixels
        edge_blur : int, optional
            Fudge factor of the edge, influencing the Gaussian filter.
            By default this will be 3.
        negative : bool
            If True, between the circles is black and edge is white.
            By default, if the mask is negative, the feature is positive,
            else the other way around
        """
        edge_blur, negative = self._get_feature_props(edge_blur=edge_blur,
                                                      negative=negative)
        circ = self._get_band_template(r1, r2, edge_blur, negative)
        self._add_mask_feature(self.mx, self.my, circ)
        operation = (f"Added {self._bool_string(negative)} pass-band. "
                     f"Radii: {r1}, {r2}. "
                     f"Edge_blur: {edge_blur}.")
        self._record_operation(operation)

    def add_square(self, x, y, r, edge_blur=None, negative=None,
                   mirrored=True, **kwargs):
        """
        Add a square feature in the mask

        By default, the square is repeated on the other side of the center.

        Parameters
        ----------
        x : int
            x position in pixels where center of circle should be placed
        y : int
            y position in pixels where center of circle should be placed
        r : int
            half the side length in pixels
        edge_blur : int, optional
            Fudge factor of the edge, influencing the Gaussian filter.
            By default this will be 3.
        negative : bool
            If True, center is black and edge is white. By default, if the
            mask is negative, the feature is positive, else the other way
            around
        mirrored : bool
            If True, the feature is repeated on the opposite side of the
            middle. Default is true.
        """
        edge_blur, negative = self._get_feature_props(edge_blur=edge_blur,
                                                      negative=negative)
        circ = self._get_square_template(r, edge_blur, negative)
        self._add_mask_feature(x, y, circ, **kwargs)
        if mirrored:
            x2, y2 = self._get_opposite_coordinates(x, y)
            self._add_mask_feature(x2, y2, circ, **kwargs)
        operation = (f"Added {self._bool_string(negative)} square. "
                     f"Side-length: {r*2}. Position: ({x}, {y}). "
                     f"Mirrored: {mirrored}. Edge_blur: {edge_blur}.")
        self._record_operation(operation)

    def plot(self, **kwargs):
        """Plot the mask, returns the axis and axisimage object"""
        return pl.plot_mask(self, **kwargs)

    def to_image(self):
        """Returns an image object with the mask as dataset"""
        process = "Mask to image"
        return create_new_image(self.data, self.pixelsize, self.pixelunit,
                                parent=None, process=process)


class FFT(TEMDataSet):
    """
    Represents a centered FFT, a complex valued image with a few special
    functions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__x = "x"
        self.__y = "y"

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def pixelsize(self):
        return self._get_axis_prop(self.x, "scale")

    @property
    def pixelunit(self):
        return self._get_axis_prop(self.x, "unit")

    @property
    def width(self):
        return self._get_axis_prop(self.x, "bins")

    @property
    def height(self):
        return self._get_axis_prop(self.y, "bins")

    def set_scale(self, scale, unit):
        """Set the image pixel scale and unit"""
        self.set_axis_scale(self.x, scale, unit)
        self.set_axis_scale(self.y, scale, unit)

    @property
    def dimensions(self):
        """
        Returns real size of image
        (width, width unit, height, height unit)
        """
        xs = self._get_axis_prop(self.x, "scale")
        ys = self._get_axis_prop(self.y, "scale")
        xb = self._get_axis_prop(self.x, "bins")
        yb = self._get_axis_prop(self.y, "bins")
        xu = self._get_axis_prop(self.x, "unit")
        yu = self._get_axis_prop(self.y, "unit")
        return (xs*xb, xu, ys*yb, yu)

    def measure(self, x1, y1, x2=None, y2=None):
        """
        Return the length and angle in real units between two points

        If no second point is provided then the center is taken to be
        the first point and (x1, y1) the second point. Coordinates in pixels.
        """
        if x2 is None or y2 is None:
            x2, x1 = x1, x2
            y2, y1 = y1, y2
            if x1 is None:
                x1 = self.width/2-0.5
            if y1 is None:
                y1 = self.height/2-0.5
        rt = algutil.rotangle(x1, y1, x2, y2)
        le = algutil.distance(x1, y1, x2, y2)
        return (le*self.pixelsize, self.pixelunit), (rt, DEGREE)

    def _create_derived_image(self, arr, process, pixelsize=None,
                              pixelunit=None):
        if pixelsize is None:
            pixelsize = self.pixelsize
        if pixelunit is None:
            pixelunit = self.pixelunit
        return create_new_image(arr, pixelsize, pixelunit,
                                parent=self, process=process)

    @property
    def real(self):
        process = "Real component"
        return self._create_derived_image(self.data.real, process)

    @property
    def imag(self):
        process = "Imaginary component"
        return self._create_derived_image(self.data.imag, process)

    @property
    def phase(self):
        process = "Phase angle in degrees"
        return self._create_derived_image(np.angle(self.data), process)

    @property
    def arg(self):
        process = "Argument"
        return self._create_derived_image(np.abs(self.data), process)

    @property
    def power_spectrum(self):
        arr = 20 * np.log10(np.abs(self.data))
        process = "Power spectrum (20*log(argument))"
        return self._create_derived_image(arr, process)

    @property
    def ifft(self):
        return self.get_ifft()

    def get_ifft(self):
        """Returns the inverse fourrier transform as a GeneralImage"""
        arr = np.abs(np.fft.ifft2(self.data))
        pixelsize = 1/(self.pixelsize*self.width)
        pixelunit = re.findall(r"^1\/(.+)$", self.pixelunit)[0]
        process = "Inverse Fourrier Transform"
        return create_new_image(arr, pixelsize, pixelunit,
                                parent=self, process=process)

    def plot(self, filename=None, **kwargs):
        logger.warning("An FFT is complex-valued and can not be visualized as "
                       "an image. Actually plot here is a smoothed power "
                       "spectrum, logarithmically scaled. If the FFT in point "
                       "(x, y) is a+bi, then the image intensity there is "
                       "log10(a^2+b^2)")
        ax, im = pl.plot_fft(self, **kwargs)
        if filename:
            ax.figure.savefig(filename)
        return ax, im

    def create_mask(self, negative=True):
        """Initialize and return a mask."""
        return Mask(self, negative)

    def _create_child_fft(self, inplace, data, process=None, parent_2=None):
        """Create a new fft from the parent"""
        newmeta = mda.Metadata()
        newmeta.experiment_type = "FFT"
        if parent_2 is None:
            newmeta.parent_meta = self.metadata
        else:
            newmeta.parent_meta = [self.metadata, parent_2.metadata]
        newmeta.process = process
        xinfo = (data.shape[1], self.pixelunit, self.pixelsize)
        yinfo = (data.shape[0], self.pixelunit, self.pixelsize)
        newmeta["data_axes"] = mda.gen_image_axes(xinfo, yinfo)
        newmeta = mda.Metadata(newmeta)
        if inplace:
            self.__init__(data, newmeta)
        else:
            return FFT(data, newmeta)

    def check_mask(self, mask, **kwargs):
        """
        Plot the power spectrum with the fft over it

        Returns the axis, plot of the power spectrum and the plot of the
        mask as objects for altering the settings of these objects.
        """
        ax, ps, msk = pl.plot_fft_and_mask(self, mask, **kwargs)
        return ax, ps, msk

    def check_masked(self, mask, **kwargs):
        """
        Plot the power spectrum multiplied by the mask
        """
        ax, msked = pl.plot_fft_masked(self, mask, **kwargs)
        return ax, msked

    def apply_mask(self, mask, inplace=False):
        """
        Multiply the fft by a mask. Returns a new masked fft.

        If inplace is on, then the data is replaced.
        """
        data = self.data*mask.data
        process = "Applied mask (multiplied)"
        return self._create_child_fft(inplace, data, process, parent_2=mask)


def create_new_image_stack(arr, pixelsize, pixelunit,
                           parent=None, process=None):
    """
    Create a GeneralImageStack object from an array, pixelsize and pixelunit

    The "experiment_type" will be set to "modified". The time axis is lost.
    A new time axis must be created with add_custom_axis.

    Parameters
    ----------
    arr : array-like, 3D (frames, height, width)
        The array representing an image stack. The first axis is the rows or y
        axis, the second axis is the columns or x axis.
    pixelsize : float
        The size of a pixel in the correct units
    pixelunit : str
        The size unit of the image
    parent : TEMDataSet, optional
        If the image is derived from another image (e.g. a filter)
        then the parent is provided as an argument. The parent metadata will
        be stored in the child metadata under "parent_metadata"
    process : object, optional
        Some description of how the dataset was obtained. Could be a
        dictionary, string, ... and it will be stored under the key "process"

    Returns
    -------
    image : GeneralImageStack
        The general image object with some easy image processing tools
    """
    arr = np.array(arr)
    assert arr.ndim == 3, "The array must have three dimensions to be a stack"
    newmeta = mda.Metadata()
    newmeta.experiment_type = "modified"
    if parent:
        newmeta.parent_meta = parent.metadata
    if process:
        newmeta.process = process
    xinfo = (arr.shape[2], pixelunit, pixelsize)
    yinfo = (arr.shape[1], pixelunit, pixelsize)
    finfo = (arr.shape[0], None, None)
    newmeta["data_axes"] = mda.gen_image_stack_axes(xinfo, yinfo, finfo)
    newmeta = mda.Metadata(newmeta)
    return GeneralImageStack(arr, newmeta)


class GeneralImageStack(TEMDataSet):
    """
    Represents a stack of 2D images
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.experiment_type == "image_series":
            self.__x = "x"
            self.__y = "y"
        elif self.experiment_type == "scan_image_series":
            self.__x = "scan_x"
            self.__y = "scan_y"
        elif self.experiment_type == "modified":
            # for images created from an array
            self.__x = "x"
            self.__y = "y"
        else:
            raise TypeError("The dataset is not an image stack")

    @property
    def frames(self):
        return self._get_axis_prop("frame", "bins")

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def pixelsize(self):
        return self._get_axis_prop(self.x, "scale")

    @property
    def pixelunit(self):
        return self._get_axis_prop(self.x, "unit")

    @property
    def width(self):
        return self._get_axis_prop(self.x, "bins")

    @property
    def height(self):
        return self._get_axis_prop(self.y, "bins")

    @property
    def exposure_time(self):
        if self.experiment_type == "image_series":
            return self._get_axis_prop("time", "scale")
        else:
            return None

    @property
    def time_unit(self):
        if self.experiment_type == "image_series":
            return self._get_axis_prop("time", "unit")
        else:
            return None

    def plot_interactive(self):
        """Plot the stack of images with a slider to go between frames"""
        pl.plot_stack(self)

    def apply_filter(self, filt, *args, inplace=False, multiprocessing=False,
                     **kwargs):
        """
        Apply a filter to all images in the stack

        Parameters
        ----------
        filt : callable
            Filter function to be applied to the images. Must take in the
            image frame as a first argument and return another 2D array.
        inplace : bool, optional
            Whether to update the data inside the object or create a new
            imagestack object. Default is create a new object.
        multiprocessing : bool, optional
            Whether to use multiple cores to perform the operation. If False
            a simple loop is used to go over all the frames. If true a
            tasks are put in a multiprocessing pool. Behavior can be a bit
            shaky and sometimes slower than the simple loop. Default is to
            use only one core.

        Other parameters
        ----------------
        args, kwargs : passed directly to the filt function.
        """
        data = imf.apply_filter_to_stack(self.data, filt, *args,
                                         multiprocessing=multiprocessing,
                                         **kwargs)
        process = {
                    "filter": {
                        "name": filt.__name__,
                        "arguments": list(args),
                        "keyword_arguments": {**kwargs}
                    }
                }
        logger.warning("The pixel size and unit of the original images were "
                       "used. The scale may no longer be correct. Please "
                       "verify and use set_scale.")
        return self._create_child_stack(inplace, data, self.pixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def get_frame(self, frame_index):
        """Return a single frame as an GeneralImage object"""
        if frame_index > self.frames or frame_index < 0:
            raise IndexError(f"The frame number is out of range "
                             f"[0-{self.frames-1}]")
        data = self.data[frame_index, :, :]
        process = f"Frame {frame_index} from stack"
        nimg = create_new_image(data, self.pixelsize, self.pixelunit, self,
                                process)
        return nimg

    def average(self):
        """Average the frames, return GeneralImage object"""
        # Sum becomes a uint64 image
        dt = self.data.dtype
        data = self.data.sum(axis=0)
        # Rescale to turn it back into the same datatype as before
        data = imf.normalize_convert(data, dtype=dt)
        process = f"Averaged all frames in stack"
        nimg = create_new_image(data, self.pixelsize, self.pixelunit, self,
                                process)
        return nimg

    def _create_child_stack(self, inplace, *args, **kwargs):
        """Wrapper for create_new_image_stack which considers inplace"""
        newstack = create_new_image_stack(*args, **kwargs)
        if inplace:
            self.__init__(newstack.data, newstack.metadata)
        else:
            return newstack

    def crop(self, x1, y1, x2, y2, inplace=False, nearest_power=False):
        """
        Select a rectangle defined by two points and return a new image stack
        """
        x1, y1, x2, y2 = _correct_crop_window(x1, y1, x2, y2,
                                              self.width, self.height,
                                              nearest_power)
        data = self.data[:, y1:y2, x1:x2].copy()
        process = f"Cropped between x={x1}-{x2} and y={y1}-{y2}"
        return self._create_child_stack(inplace, data, self.pixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def rebin(self, factor, inplace=False, multiprocessing=False, **kwargs):
        """Rebin the images in a stack by a certain factor"""
        data = imf.apply_filter_to_stack(self.data, imf.bin2, factor,
                                         multiprocessing=multiprocessing,
                                         **kwargs)
        # update factor if non-divisor was used
        factor = (self.data.shape[-1])/(data.shape[-1])
        process = f"Binned by a factor of {factor}"
        newpixelsize = self.pixelsize*factor
        return self._create_child_stack(inplace, data, newpixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def linscale(self, min, max, inplace=False):
        """
        Change the minimum and maximum intensity values

        The dtype of the image remains the same
        """
        data = imf.normalize_convert(self.data, min, max,
                                     dtype=self.data.dtype)
        process = f"Scaled intensity between {min} and {max}"
        return self._create_child_stack(inplace, data, self.pixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def select_frames(self, frames, inplace=False):
        """
        Select a list of frames
        """
        data = self.data[frames, :, :]
        process = f"Selected frames {frames}"
        return self._create_child_stack(inplace, data, self.pixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def exclude_frames(self, frames, inplace=False):
        """
        Exclude a list of frames
        """
        data = np.delete(self.data, frames, axis=0)
        process = f"Excluded frames {frames}"
        return self._create_child_stack(inplace, data, self.pixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def align_frames(self, shifts=None, inplace=False, **kwargs):
        """
        Perform rigid registration on images

        Parameters
        ----------
        shifts : tuple of 1D arrays, optional
            (x_shifts, y_shifts), calculated with calc_misalignment or
            created by the user.
        inplace : bool
            Perform shift in place

        Other parameters
        ----------------
        **kwargs
            If no shifts are provided, they are calculated. The **kwargs
            are passed to calc_misalignment
        """
        if shifts is None:
            logger.warning("No alignment shifts were provided, "
                           "will try to calculate the shifts.")
            sx, sy = self.calc_misalignment(**kwargs)
        else:
            sx, sy = shifts
        # correct the images
        newdat = []
        for j, i in enumerate(self.data):
            newdat.append(ndi.shift(i, (sy[j], sx[j])))
        newdat = np.array(newdat)
        process = "Aligned frames"
        return self._create_child_stack(inplace, newdat, self.pixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    def calc_misalignment(self, cumulative=False, rectangle=None, border=None,
                          fraction=None, reg=0.5, debug=False):
        """
        Calculate the shifts between images based on autocorrelation

        A small rectangular window in the image is cross-correlated with
        a slightly larger rectangle in the previous image of the stack.
        From the maximum, the necessary shift in the second image is calculated
        to have the maximum match between the rectangles.

        Parameters
        ----------
        cumulative : bool, optional
            If False, all frames will be cross-correlated with the first frame.
            If True, each frame will be compared to the previous frame, then
            the absolute shifts of each frame are calculated by cumulating the
            relative shifts. Can be very instable - errors also accumulate.
        rectangle : 4-tuple or list, optional
            Of the form [x1, y1, x2, y2], defining the area that is taken as
            the feature. Must provide either a rectangle or a fraction.
        border : int, optional
            The search area is defined by the rectangle
            [x1-border, y1-border, x2+border, y2+border]. If none specified
            it will take 1/3*min(x2-x1, y2-y1)
        fraction : float, optional
            If no rectangle is specified, a rectangle will be taken in the
            middle of the image that is (width*fraction, height*fraction)
            in size. Must provide either a rectangle or a fraction.
        reg : float, optional
            Regularization constant to penalize large displacements. If >0,
            the cross-correlation is subtracted by reg*D where D the distance
            from the center.
        debug : bool, optional
            If True, an image stack is returned comprised of the
            cross-correlations.

        Returns
        -------
        (rel_shifts_x, rel_shifts_y) : tuple of 1D numpy arrays
            Shifts in x and shifts in y for each frame
        cc_stack : GeneralImageStack, optional
            Stack of cross-correlation frames. Returned only if debug is True.
        """
        if (rectangle is None and fraction is None):
            raise ValueError("You must provide a fraction or rectangle")
        elif (rectangle is not None and fraction is not None):
            raise ValueError("You can't provide both a fraction and rectangle")
        elif (rectangle is not None):
            x1, y1, x2, y2 = rectangle
        else:
            my, mx = (self.height//2, self.width//2)
            y1 = int((1-fraction)*my)
            y2 = int((1+fraction)*my)
            x1 = int((1-fraction)*mx)
            x2 = int((1+fraction)*mx)

        # coordinates for the feature
        x1, y1, x2, y2 = _correct_crop_window(x1, y1, x2, y2, self.width,
                                              self.height)
        if border is None:
            border = min((x2-x1), (y2-y1))//3
        else:
            border = int(abs(border))

        # coordinates for the search window
        x1b, y1b, x2b, y2b = _correct_crop_window(x1-border, y1-border,
                                                  x2+border, y2+border,
                                                  self.width, self.height)
        rel_shifts_x = [0]
        rel_shifts_y = [0]
        # normalize entire stack
        ndata = imf.normalize(self.data)
        if debug:
            cc_d = []
        # determine the first frame
        frm0 = ndata[0, y1b:y2b, x1b:x2b]
        frm0 = frm0 - frm0.mean()
        # a distance from the middle regularization term
        x = np.arange(frm0.shape[1])-frm0.shape[1]//2
        y = np.arange(frm0.shape[0])-frm0.shape[0]//2
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2+Y**2)
        for i in range(1, self.frames):
            if cumulative:
                frm0 = ndata[i-1, y1b:y2b, x1b:x2b]
                frm0 = frm0 - frm0.mean()
            frmcompare = ndata[i, y1:y2, x1:x2]
            frmcompare = frmcompare - frmcompare.mean()
            cc = ndi.correlate(frm0, frmcompare, mode="constant", cval=0)
            # regularize the cross-correlation with distance
            if reg > 0:
                DT = imf.linscale(D, nmin=0, nmax=cc.max())
                cc = cc - reg*DT
            if debug:
                cc_d.append(cc)
            ymx, xmx = np.where(cc == np.max(cc))
            ymx = ymx[0]
            xmx = xmx[0]
            # absolute coordinate of half the compare frame - should match
            hxfc = x1+frmcompare.shape[1]//2
            hyfc = y1+frmcompare.shape[0]//2
            dx = x1b+xmx-hxfc
            dy = y1b+ymx-hyfc
            rel_shifts_x.append(dx)
            rel_shifts_y.append(dy)
        rel_shifts_x = np.array(rel_shifts_x)
        rel_shifts_y = np.array(rel_shifts_y)
        # if cumulative, shifts must be cumulated
        if cumulative:
            rel_shifts_x = np.cumsum(rel_shifts_x)
            rel_shifts_y = np.cumsum(rel_shifts_y)

        if not debug:
            return (rel_shifts_x, rel_shifts_y)
        else:
            cc_d = np.array(cc_d)
            cc_stack = self._create_child_stack(False, cc_d,
                                                self.pixelsize,
                                                self.pixelunit,
                                                parent=self, process=None)
            return ((rel_shifts_x, rel_shifts_y), cc_stack)

    def plot_frame(self, frame_index, **kwargs):
        """
        Plots a single frame from the image stack.

        Wrapper for GeneralImag.plot
        """
        img = self.get_frame(frame_index)
        fig, ax = img.plot(**kwargs)
        return fig, ax

    def save_frame(self, frame_index, filename, **kwargs):
        """
        Export a single image frame
        Wrapper for GeneralImag.save
        """
        frame = self.get_frame(frame_index)
        frame.save(filename)

    def _set_counter(self, counter=None):
        """Set the number of counter digits. If too low will take minimum."""
        # set the number of digits in the counter
        mincounter = _get_counter(self.frames)
        if counter is None:
            counter = mincounter
        elif counter < mincounter:
            logger.error("Insufficient digits to represent"
                         "the frames, will use minimum necessary")
            counter = mincounter
        else:
            pass  # counter is already set to the right value
        return counter

    @timeit
    def export_frames(self, path, name="Frame", counter=None, frames=None,
                      multithreading=False, **kwargs):
        '''
        Exports all the image frames

        No processing or scale bars are applied
        Exports as grayscale PNGS

        Parameters
        ----------
        path : str
            the folder where the images will be written to
        name : str, optional
            name prefix for the images to write out, default is "Frame"
        counter : int, optional
            the number of digits in the image counter
        frames : list, optional
            list of frames to export.
        multithreading: bool, optional
            Use multiple threads for operation for speed up.
            Can sometimes give errors and may not be faster in all cases.

        Other parameters
        ----------------
        **kwargs: optional kwargs to pass to image_filters.normalize_convert:
        min, max, dtype.
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        if frames is None:
            self.metadata.to_file(str(Path(path+"/metadata.json")))
        # set the number of digits in the counter
        counter = self._set_counter(counter)
        _loop_over_stack_thread(self, _save_frame_to_file, path, name, counter,
                                frames=frames, multithreading=multithreading,
                                **kwargs)

    @timeit
    def plot_export_frames(self, path, name="Frame", counter=None, frames=None,
                           multiprocessing=False, **kwargs):
        '''
        Shortcut to save all images to separate files with scale bars

        The images will be saved as RGB 16 bit TIFFS

        Parameters
        ----------
        path : str
            the folder where the images will be written to
        name : str, optional
            name prefix for the images to write out. Defaults to "Frame".
        counter : int, optional
            the number of digits in the image counter. Default is minimum.
        frames : list, optional
            list of frames to export. Default is all.
        multiprocessing: bool, optional
            Use multiple cores for operation for speed up.
            Can sometimes give errors and may not be faster in all cases.

        Other parameters
        ----------------
        **kwargs: optional kwargs to pass to plottingtools.plot_image.

        Notes
        -----
        In the kwargs, show_fig and scale_bar are already set to False and True
        dpi is set to 100.
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        # write out the metadata, only if all frames are written out
        if frames is None:
            self.metadata.to_file(str(Path(path+"/metadata.json")))
        # set the number of digits in the counter
        counter = self._set_counter(counter)
        _loop_over_stack(self, _plot_frame_to_file, self.pixelsize,
                         self.pixelunit, path, name, counter,
                         frames=frames, multiprocessing=multiprocessing,
                         **kwargs)

    def to_hspy(self, filepath=None):
        """
        Return the image stack as a hyperspy dataset

        If a file path is provided, it is also saved as a .hspy file
        """
        hs = super().get_hs()
        hsim = hs.signals.Signal2D(self.data)
        hsim.axes_manager[0].name = "frame"
        hsim.axes_manager[1].name = self.x
        hsim.axes_manager[self.x].units = self._get_axis_prop(self.x, "unit")
        hsim.axes_manager[self.x].scale = self._get_axis_prop(self.x, "scale")
        hsim.axes_manager[2].name = self.y
        hsim.axes_manager[self.y].units = self._get_axis_prop(self.y, "unit")
        hsim.axes_manager[self.y].scale = self._get_axis_prop(self.y, "scale")
        if filepath:
            hsim.save(str(Path(filepath)))
        return hsim


def _loop_over_stack(stack, do_in_loop, *args, frames=None,
                     multiprocessing=True, **kwargs):
    if frames is None:
        toloop = range(stack.frames)
    elif isinstance(frames, list):
        toloop = frames
    else:
        raise TypeError("Argument frames must be a list")
    if multiprocessing:
        try:
            workers = cpu_count()
        except NotImplementedError:
            workers = 1
        with Pool(processes=workers) as pool:
            pool.map(FrameByFrame(do_in_loop, stack.data, *args, **kwargs),
                     toloop)
    else:
        for i in toloop:
            do_in_loop(i, stack.data, *args, **kwargs)


def _loop_over_stack_thread(stack, do_in_loop, *args, frames=None,
                            multithreading=True, **kwargs):
    if frames is None:
        toloop = range(stack.frames)
    elif isinstance(frames, list):
        toloop = frames
    else:
        raise TypeError("Argument frames must be a list")
    if multithreading:
        with cf.ThreadPoolExecutor() as pool:
            pool.map(FrameByFrame(do_in_loop, stack.data, *args, **kwargs),
                     toloop)
    else:
        for i in toloop:
            do_in_loop(i, stack.data, *args, **kwargs)


def _save_frame_to_file(i, data, path, name, counter, **kwargs):
    """Helper function for multiprocessing, saving frame i of stack"""
    c = str(i).zfill(counter)
    fp = str(Path(f"{path}/{name}_{c}.png"))
    frm = data[i]
    frame = imf.normalize_convert(frm, **kwargs)
    img = Image.fromarray(frame)
    img.save(fp)


def _plot_frame_to_file(i, data, pixelsize, pixelunit, path, name,
                        counter, **kwargs):
    """Helper function for multiprocessing, saving plot of frame i of stack"""
    c = str(i).zfill(counter)
    fp = str(Path(f"{path}/{name}_{c}.png"))
    frm = data[i]
    ax, im = pl.plot_array(frm, pixelsize=pixelsize, pixelunit=pixelunit,
                           show_fig=False, **kwargs)
    ax.figure.savefig(fp)


class FrameByFrame(object):
    """A pickle-able wrapper for doing a function on all frames of a stack"""
    def __init__(self, do_in_loop, stack, *args, **kwargs):
        self.func = do_in_loop
        self.stack = stack
        self.args = args
        self.kwargs = kwargs

    def __call__(self, index):
        self.func(index, self.stack, *self.args, **self.kwargs)


def images_to_stack(lst):
    """
    Combines a list of individual GeneralImage objects into a stack
    """
    data = np.array([i.data for i in lst])
    template = lst[0]
    metadata = mda.Metadata()
    metadata.experiment_type = "modified"
    metadata.parent_meta = template.metadata
    metadata.process = (f"Stacked individual images")
    xinfo = (data.shape[2], template.pixelunit, template.pixelsize)
    yinfo = (data.shape[1], template.pixelunit, template.pixelsize)
    finfo = (data.shape[0], None, None)
    metadata["data_axes"] = mda.gen_image_stack_axes(xinfo, yinfo,
                                                     finfo)
    metadata = mda.Metadata(metadata)
    return GeneralImageStack(data, metadata)


def import_file_to_image(path):
    """
    Re-import an image from a file and return a GeneralImage object

    Tries to also import a metadata json file. Only if the json file has the
    same name + _meta as the file will this work! If found it will
    automatically update, otherwise it starts with blank metadata.
    """
    im = Image.open(path)
    data = np.array(im)
    # get name of the file
    fp, _ = os.path.splitext(path)
    metapath = f"{fp}_meta.json"
    if os.path.isfile(metapath):
        meta = jt.read_json(metapath)
        metadata = mda.Metadata(meta)
    else:
        logger.error(f"No metadata json file associated with the image was"
                     f" found. Creating blank metadata and assuming pixel "
                     f"units.")
        metadata = mda.Metadata()
        metadata.experiment_type = "modified"
        metadata.process = (f"Imported from file")
        xinfo = (data.shape[2], "pixels", 1)
        yinfo = (data.shape[1], "pixels", 1)
        metadata["data_axes"] = mda.gen_image_axes(xinfo, yinfo)
        metadata = mda.Metadata(metadata)
    return GeneralImage(data, metadata)


def import_files_to_stack(export_folder):
    """
    Re-import a set of images in a folder and return a stack object

    The images must follow the naming convention prefix_number.extension.
    They must be importable by Image.open. Tries to also import a metadata
    json file. If found it will automatically update, otherwise it starts
    with blank metadata.
    """
    files = os.listdir(export_folder)
    pattern = re.compile(r"(.*)\_([0-9]+)\..+")
    image_names = []
    meta = ""  # name of the metadata file
    prefix = ""  # to be able to find the initial iteration
    for i in files:
        mt = pattern.match(i)
        if mt:
            if not prefix:
                prefix, _ = mt.groups()
            else:
                prefix_compare, _ = mt.groups()
                if prefix_compare != prefix:
                    raise ValueError("Not all images in the folder have the"
                                     " same name, or other files match the "
                                     "pattern .*\\_[0-9]+\\..+ . "
                                     "Could not create stack")
            image_names.append(i)
        else:
            # it might be a JSON file
            pattern_meta = re.compile(r"(.*)\.json")
            match2 = pattern_meta.match(i)
            if match2:
                meta = i

    image_names.sort()

    if image_names:
        images = []
        for i in image_names:
            path = export_folder+"/"+i
            im = Image.open(path)
            im = np.array(im)
            images.append(im)
        data = np.array(images)
        if meta:
            meta = jt.read_json(export_folder+"/"+meta)
            metadata = mda.Metadata(meta)
        else:
            logger.error(f"No metadata json file was found in the folder. "
                         f"Creating blank metadata and assuming pixel units.")
            metadata = mda.Metadata()
            metadata.experiment_type = "modified"
            metadata.process = (f"Imported from individual images in "
                                f"{export_folder}")
            xinfo = (data.shape[2], "pixels", 1)
            yinfo = (data.shape[1], "pixels", 1)
            finfo = (data.shape[0], None, None)
            metadata["data_axes"] = mda.gen_image_stack_axes(xinfo, yinfo,
                                                             finfo)
            metadata = mda.Metadata(metadata)
        return GeneralImageStack(data, metadata)
    else:
        logger.error(f"No images detected in {export_folder}")
        return None


class Spectrum(TEMDataSet):
    """
    Abstraction of a single 1D spectrum
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peaks = None

    @property
    def x(self):
        return "channel"

    @property
    def axis_name(self):
        return "Energy"

    def set_energy_axis(self, dispersion, unit, offset=0):
        """Set the energy scale, unit and offset"""
        self.set_axis_scale("channel", dispersion, unit, offset)

    @property
    def channels(self):
        return self._get_axis_prop(self.x, "bins")

    @property
    def dispersion(self):
        return self._get_axis_prop(self.x, "scale")

    @property
    def energy_unit(self):
        return self._get_axis_prop(self.x, "unit")

    @property
    def spectrum_offset(self):
        return self._get_axis_prop(self.x, "offset")

    @property
    def channel_axis(self):
        return np.arange(self.channels)

    def _get_energy_of_channel(self, channel):
        return channel*self.dispersion+self.spectrum_offset

    def _get_channel_of_energy(self, energy):
        return int((energy-self.spectrum_offset)/self.dispersion)

    @property
    def energy_axis(self):
        return self._get_energy_of_channel(self.channel_axis)

    def calc_peaks(self, pf_props={"height": 300, "width": 10}):
        """
        Calculate the peaks in the spectrum

        Parameters
        ----------
        pf_props : dict
            Properties passed to the peak finding algorithm
            See: processing.get_spectrum_peaks. Default is
            {"height": 300, "width": 10}

        Returns
        -------
        df : pandas DataFrame
            Three column table:
                - channel indexes corresponding to peaks,
                - energies corresponding to the channels
                - Height of the peaks
        props : additional properties of the peaks
        """
        arr = self.data
        peaks, props = proc.get_spectrum_peaks(arr, **pf_props)
        peak_heights = arr[peaks]
        peak_energy = self._get_energy_of_channel(peaks)
        adt = np.vstack([peaks, peak_energy, peak_heights]).T
        df = pd.DataFrame(adt, columns=["Channel",
                                        f"Energy ({self.energy_unit})",
                                        f"Intensity (a.u.)"])
        self.peaks = df
        return df, props

    def plot(self, ax=None, show_peaks=False, **kwargs):
        return pl.plot_spectrum(self, ax=ax, show_peaks=show_peaks, **kwargs)

    def to_hspy(self, filepath=None):
        """
        Convert to a hyperspy dataset and potentially save to file

        Metadata not related to linear axes scales and units is lost.
        """
        hs = super().get_hs()
        hsim = hs.signals.Signal1D(self.data)
        hsim.axes_manager[0].name = self.axis_name
        hsim.axes_manager[self.axis_name].units = self._get_axis_prop(self.x,
                                                                      "unit")
        hsim.axes_manager[self.axis_name].scale = self._get_axis_prop(self.x,
                                                                      "scale")
        if filepath:
            hsim.save(str(Path(filepath)))
        return hsim

    def to_excel(self, filepath):
        """Write out the profile data to an excel file with two columns"""
        dt = self.data
        x = self.energy_axis
        adt = np.vstack([x, dt]).T
        gg = pd.DataFrame(adt, columns=[f"Energy ({self.energy_unit})",
                                        f"Intensity (a.u.)"])
        gg.to_excel(str(Path(filepath)))


class LineSpectrum(TEMDataSet):
    """Abstraction of line spectrum"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__x = "scan_l"

    @property
    def x(self):
        return self.__x

    @property
    def pixelsize(self):
        return self._get_axis_prop(self.x, "scale")

    @property
    def pixelunit(self):
        return self._get_axis_prop(self.x, "unit")

    @property
    def length(self):
        return self._get_axis_prop(self.x, "bins")

    @property
    def channels(self):
        return self._get_axis_prop("channel", "bins")

    @property
    def dispersion(self):
        return self._get_axis_prop("channel", "scale")

    @property
    def energy_unit(self):
        return self._get_axis_prop("channel", "unit")

    @property
    def spectrum_offset(self):
        return self._get_axis_prop("channel", "offset")

    def _get_energy_of_channel(self, channel):
        return channel*self.dispersion+self.spectrum_offset

    def _get_channel_of_energy(self, energy):
        return int((energy-self.spectrum_offset)/self.dispersion)

    def set_scale(self, scale, unit):
        """Set the scan pixel scale and unit"""
        self.set_axis_scale(self.x, scale, unit)

    def set_energy_axis(self, dispersion, unit, offset=0):
        """Set the energy scale, unit and offset"""
        self.set_axis_scale("channel", dispersion, unit, offset)

    @property
    def spectrum(self):
        data = self.data.sum(axis=1)
        process = "Total spectrum"
        return create_new_spectrum(data, self.dispersion, self.energy_unit,
                                   self.spectrum_offset, parent=self,
                                   process=process)

    def _get_start_end_channels(self, energy, width):
        start = self._get_channel_of_energy(energy-width/2)
        if start < 0:
            start = 0
        # ending channel index
        end = self._get_channel_of_energy(energy+width/2)
        if end > self.channels:
            end = self.channels
        return (start, end)

    def _integrate_to_line(self, energy, width):
        start, end = self._get_start_end_channels(energy, width)
        return self.data[start:end, :].sum(axis=0)

    def get_profile(self, energy, width):
        start, end = self._get_start_end_channels(energy, width)
        dt_ar = self.data[start:end, :].sum(axis=0)
        process = (f"Integrated channels {start}-{end} "
                   f"({energy} p.m. {width} {self.energy_unit})")
        nprof = create_profile(dt_ar, self.pixelsize, self.pixelunit, self,
                               process)
        return nprof

    def plot_interactive(self):
        pl.plot_line_spectrum(self)

    def to_hspy(self, filename=None):
        """
        Turn into spectrum map object
        """
        hs = super().get_hs()
        hsim = hs.signals.Signal1D(self.data)
        hsim.axes_manager[0].name = "Energy"
        hsim.axes_manager["Energy"].units = self.energy_unit
        hsim.axes_manager["Energy"].scale = self.dispersion
        hsim.axes_manager["Energy"].offset = self.spectrum_offset
        hsim.axes_manager[1].name = self.x
        hsim.axes_manager[self.x].units = self._get_axis_prop(self.x, "unit")
        hsim.axes_manager[self.x].scale = self._get_axis_prop(self.x, "scale")
        if filename:
            hsim.save(str(Path(filename)))
        return hsim


class SpectrumMap(TEMDataSet):
    '''
    Abstraction of a Spectrum map.

    Can be obtained by adding together all frames.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.experiment_type == "spectrum_map" or
           self.experiment_type == "modified"):
            self.__x = "scan_x"
            self.__y = "scan_y"
        else:
            raise TypeError("The dataset is not a spectrum stream")

    def _create_child_map(self, inplace, data, pixelsize=None,
                          dispersion=None, spectrum_offset=None, process=None):
        if pixelsize is None:
            pixelsize = self.pixelsize
        if dispersion is None:
            dispersion = self.dispersion
        if spectrum_offset is None:
            spectrum_offset = self.spectrum_offset
        newmap = create_new_spectrum_map(data, pixelsize, self.pixelunit,
                                         dispersion, self.energy_unit,
                                         spectrum_offset, parent=self,
                                         process=process)
        if not inplace:
            return newmap
        else:
            self.__init__(data, newmap.metadata)

    def set_scale(self, scale, unit):
        """Set the scan pixel scale and unit"""
        self.set_axis_scale(self.x, scale, unit)
        self.set_axis_scale(self.y, scale, unit)

    def set_energy_axis(self, dispersion, unit, offset=0):
        """Set the energy scale, unit and offset"""
        self.set_axis_scale("channel", dispersion, unit, offset)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def pixelsize(self):
        return self._get_axis_prop(self.x, "scale")

    @property
    def pixelunit(self):
        return self._get_axis_prop(self.x, "unit")

    @property
    def width(self):
        return self._get_axis_prop(self.x, "bins")

    @property
    def height(self):
        return self._get_axis_prop(self.y, "bins")

    @property
    def channels(self):
        return self._get_axis_prop("channel", "bins")

    @property
    def dispersion(self):
        return self._get_axis_prop("channel", "scale")

    @property
    def dimensions(self):
        return (self.channels, self.height, self.width)

    @property
    def energy_unit(self):
        return self._get_axis_prop("channel", "unit")

    @property
    def spectrum_offset(self):
        return self._get_axis_prop("channel", "offset")

    def _get_energy_of_channel(self, channel):
        return channel*self.dispersion+self.spectrum_offset

    def _get_channel_of_energy(self, energy):
        return int((energy-self.spectrum_offset)/self.dispersion)

    def _get_start_end_channels(self, energy, width):
        start = self._get_channel_of_energy(energy-width/2)
        if start < 0:
            start = 0
        # ending channel index
        end = self._get_channel_of_energy(energy+width/2)
        if end > self.channels:
            end = self.channels
        return (start, end)

    def _get_integrated_image(self, energy, width):
        # starting channel index
        start, end = self._get_start_end_channels(energy, width)
        # add all these columns and make into 1D numpy array
        return self.data[start:end, :, :].sum(axis=0)

    def integrate_to_image(self, energy, width):
        '''
        Get 2D array (x,y) with total counts within a certain energy window

        Parameters
        ----------
        energy : float
            Energy to integrate intensities
        width : float
            Width of the energy window to integrate

        Returns
        -------
        nimg : GeneralImage
            New image object
        '''
        start, end = self._get_start_end_channels(energy, width)
        dt_ar = self._get_integrated_image(energy, width)
        process = (f"Integrated channels {start}-{end} "
                   f"({energy} p.m. {width} {self.energy_unit})")
        nimg = create_new_image(dt_ar, self.pixelsize, self.pixelunit, self,
                                process)
        return nimg

    def rebin(self, factor, inplace=False):
        """
        Rebin scan dimensions by an integer factor

        Contributions from neighboring pixels are added together. Only
        the simple integer binning is supported, for more advanced binning
        with interpolation consider using Hyperspy.

        Parameters
        ----------
        factor : integer
            Rebin scaling factor. If 2, then the width and height are divided
            by 2 and nearest 4 pixels summed.
        inplace : bool
            Whether to replace the data inside the object or return a new one.

        Returns
        -------
        nmap : SpectrumMap
            New spectrummap object
        """
        assert factor >= 1, "Factor must be greater than 1"
        assert self.height % factor == 0, ("Factor must divide x "
                                           "and y-dimension")
        assert self.height % factor == 0, ("Factor must divide x "
                                           "and y-dimension")
        scale = np.array([1, factor, factor])
        data = imf.bin2_simple(self.data, scale)
        process = f"Rebinned {self.x} and {self.y} axis by factor {factor}"
        return self._create_child_map(inplace, data,
                                      pixelsize=self.pixelsize*factor,
                                      process=process)

    def rebin_energy(self, factor, inplace=False):
        """
        Rebin energy dimensions by an integer factor

        Contributions from neighboring pixels are added together. Only
        the simple integer binning is supported, for more advanced binning
        with interpolation consider using Hyperspy.

        Parameters
        ----------
        factor : integer
            Rebin scaling factor. If 2, then the number of channels is divided
            by 2 and nearest 2 channels summed.
        inplace : bool
            Whether to replace the data inside the object or return a new one.

        Returns
        -------
        nmap : SpectrumMap
            New spectrummap object
        """
        assert factor >= 1, "Factor must be greater than 1"
        assert self.channels % factor == 0, ("Factor must divide "
                                             "channel-dimension")
        scale = np.array([factor, 1, 1])
        data = imf.bin2_simple(self.data, scale)
        process = f"Rebinned channel axis by factor {factor}"
        return self._create_child_map(inplace, data,
                                      dispersion=self.dispersion*factor,
                                      process=process)

    def plot_interactive(self):
        """Plot so you can browse through the images"""
        pl.plot_spectrum_map(self)

    def crop(self, x1, y1, x2, y2, nearest_power=False, inplace=False):
        """
        Crop the spectrum map and return a new spectrum map
        """
        x1, y1, x2, y2 = _correct_crop_window(x1, y1, x2, y2,
                                              self.width, self.height,
                                              nearest_power)
        data = self.data[:, y1:y2, x1:x2].copy()
        process = f"Cropped between x={x1}-{x2} and y={y1}-{y2}"
        return self._create_child_map(inplace, data, process=process)

    @property
    def spectrum(self):
        '''Return the total spectrum in the map'''
        data = self.data.sum(axis=1).sum(axis=1)
        process = "Total spectrum"
        return create_new_spectrum(data, self.dispersion, self.energy_unit,
                                   self.spectrum_offset, parent=self,
                                   process=process)

    def line_spectrum(self, x1, y1, x2, y2, w=1):
        """
        Get a line spectrum object
        """
        assert (x1 >= 0 and x1 < self.width and
                x2 >= 0 and x2 < self.width and
                y1 >= 0 and y1 < self.height and
                y2 >= 0 and y2 < self.height), "Some index is out of bounds"
        img = self.data.copy()
        rt = algutil.rotangle(x1, y1, x2, y2)
        # rotated matrix - quite slow!
        rim = ndi.rotate(img, rt, (2, 1), cval=0)
        # get rotated coordinates of the ends of the line
        le = algutil.distance(x1, y1, x2, y2)
        d = (x2-x1)/le
        b = (y2-y1)/le
        midxold = img.shape[2]/2
        midyold = img.shape[1]/2
        midxnew = rim.shape[2]/2
        midynew = rim.shape[1]/2
        x1r = int(round(d*(x1-midxold)+b*(y1-midyold) + midxnew))
        x1r = max(0, x1r)  # take into consideration out of bounds
        y1r = int(round(-b*(x1-midxold)+d*(y1-midxold) + midynew))
        x2r = int(round(d*(x2-midxold)+b*(y2-midxold) + midxnew))
        x2r = min(img.shape[2], x2r)  # take into consideration out of bounds
        # y2r = int(round(-b*(x2-midxold)+d*(y2-midxold) + midynew))
        # is of course equal to y1r
        # select the "window" around the line and sum over the height
        # take into consideration out of bounds
        yb1 = max((y1r-w//2), 0)
        yb2 = min((y1r+w//2+1), img.shape[1])
        lpf = rim[:, yb1:yb2, x1r:x2r].sum(axis=1)
        logger.debug(f"Shape of line profile data: {lpf.shape}")
        # create line_profile object
        process = (f"Spectrum line profile from ({x1}, {y1}) to ({x2}, {y2}). "
                   f"Integration width: {w}")
        return create_new_line_spectrum(lpf, self.pixelsize, self.pixelunit,
                                        self.dispersion, self.energy_unit,
                                        self.spectrum_offset,
                                        parent=self, process=process)

    def to_hspy(self, filename=None):
        """
        Turn into spectrum map object
        """
        hs = super().get_hs()
        hsim = hs.signals.Signal2D(self.data)
        hsim.axes_manager[0].name = "Energy"
        hsim.axes_manager["Energy"].units = self.energy_unit
        hsim.axes_manager["Energy"].scale = self.dispersion
        hsim.axes_manager["Energy"].offset = self.spectrum_offset
        hsim.axes_manager[1].name = self.y
        hsim.axes_manager[self.y].units = self._get_axis_prop(self.y, "unit")
        hsim.axes_manager[self.y].scale = self._get_axis_prop(self.y, "scale")
        hsim.axes_manager[2].name = self.x
        hsim.axes_manager[self.x].units = self._get_axis_prop(self.x, "unit")
        hsim.axes_manager[self.x].scale = self._get_axis_prop(self.x, "scale")
        if filename:
            hsim.save(str(Path(filename)))
        return hsim


def create_new_spectrum(arr, dispersion, energy_unit, spectrum_offset,
                        parent=None, process=None):
    arr = np.array(arr)
    assert arr.ndim == 1, "The array must have 1 dimension"
    newmeta = mda.Metadata()
    newmeta.experiment_type = "modified"
    if parent:
        newmeta.parent_meta = parent.metadata
    if process:
        newmeta.process = process
    cinfo = (arr.shape[0], energy_unit, dispersion, spectrum_offset)
    newmeta["data_axes"] = mda.gen_spectrum_axes(cinfo)
    newmeta = mda.Metadata(newmeta)
    return Spectrum(arr, newmeta)


def create_new_line_spectrum(arr, pixelsize, pixelunit, dispersion,
                             energy_unit, spectrum_offset,
                             parent=None, process=None):
    arr = np.array(arr)
    assert arr.ndim == 2, "The array must have 2 dimensions"
    newmeta = mda.Metadata()
    newmeta.experiment_type = "modified"
    if parent:
        newmeta.parent_meta = parent.metadata
    if process:
        newmeta.process = process
    linfo = (arr.shape[1], pixelunit, pixelsize)
    cinfo = (arr.shape[0], energy_unit, dispersion, spectrum_offset)
    newmeta["data_axes"] = mda.gen_spectrum_line_axes(linfo, cinfo)
    newmeta = mda.Metadata(newmeta)
    return LineSpectrum(arr, newmeta)


def create_new_spectrum_map(arr, pixelsize, pixelunit, dispersion, energy_unit,
                            spectrum_offset, parent=None, process=None):
    """
    Create a SpectrumMap object from an array, pixelsize, pixelunit, dispersion
    energy_unit and spectrum offset

    The "experiment_type" will be set to "modified". Any time axis is lost.

    Parameters
    ----------
    arr : array-like, 3D (channels, height, width)
        The array representing the spectrum map.
    pixelsize : float
        The size of a pixel in the correct units
    pixelunit : str
        The size unit of the pixel
    dispersion : float
        Energy per channel
    energy_unit : string
        Energy unit
    spectrum_offset : float
        Energy of channel 0
    parent : TEMDataSet, optional
        If the dataset is derived from another dataset (e.g. a filter)
        then the parent is provided as an argument. The parent metadata will
        be stored in the child metadata under "parent_metadata"
    process : object, optional
        Some description of how the dataset was obtained. Could be a
        dictionary, string, ... and it will be stored under the key "process"

    Returns
    -------
    map : SpectrumMap
        The SpectrumMap object with some easy processing tools
    """
    arr = np.array(arr)
    assert arr.ndim == 3, "The array must have three dimensions to be a map"
    newmeta = mda.Metadata()
    newmeta.experiment_type = "modified"
    if parent:
        newmeta.parent_meta = parent.metadata
    if process:
        newmeta.process = process
    xinfo = (arr.shape[2], pixelunit, pixelsize)
    yinfo = (arr.shape[1], pixelunit, pixelsize)
    cinfo = (arr.shape[0], energy_unit, dispersion, spectrum_offset)
    newmeta["data_axes"] = mda.gen_spectrum_map_axes(xinfo, yinfo, cinfo)
    newmeta = mda.Metadata(newmeta)
    return SpectrumMap(arr, newmeta)


def create_new_spectrum_stream(arr, xbins, ybins, pixelsize, pixelunit,
                               channels, dispersion, energy_unit,
                               spectrum_offset, frames, parent=None,
                               process=None):
    """
    Create a SpectrumStream object from an array, pixelsize, pixelunit,
    dispersion, energy_unit, spectrum offset, and number of frames

    The "experiment_type" will be set to "modified". Any time axis is lost.

    Parameters
    ----------
    arr : scipy sparse matrix
        Stream data in 1 sparse matrix
    xbins : int
        Number of bins in the x-direction
    ybins : int
        Number of bins in the y-direction
    pixelsize : float
        The size of a pixel in the correct units
    pixelunit : str
        The size unit of the pixel
    channels : int
        Number of energy channels
    dispersion : float
        Energy per channel
    energy_unit : string
        Energy unit
    spectrum_offset : float
        Energy of channel 0
    frames : int
        Number of frames
    parent : TEMDataSet, optional
        If the dataset is derived from another dataset (e.g. a filter)
        then the parent is provided as an argument. The parent metadata will
        be stored in the child metadata under "parent_metadata"
    process : object, optional
        Some description of how the dataset was obtained. Could be a
        dictionary, string, ... and it will be stored under the key "process"

    Returns
    -------
    map : SpectrumStream
        The SpectrumStream object with some easy processing tools
    """
    newmeta = mda.Metadata()
    newmeta.experiment_type = "modified"
    if parent:
        newmeta.parent_meta = parent.metadata
    if process:
        newmeta.process = process
    xinfo = (xbins, pixelunit, pixelsize)
    yinfo = (ybins, pixelunit, pixelsize)
    cinfo = (channels, energy_unit, dispersion, spectrum_offset)
    newmeta["data_axes"] = mda.gen_spectrum_stream_axes(xinfo, yinfo,
                                                        int(frames), cinfo)
    newmeta = mda.Metadata(newmeta)
    return SpectrumStream(arr, newmeta)


class SpectrumStream(TEMDataSet):
    '''
    Abstraction of SpectrumStream data as an object
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.experiment_type == "spectrum_stream" or
           self.experiment_type == "modified"):
            self.__x = "scan_x"
            self.__y = "scan_y"
        else:
            raise TypeError("The dataset is not a spectrum stream")

    def set_scale(self, scale, unit):
        """Set the scan pixel scale and unit"""
        self.set_axis_scale(self.x, scale, unit)
        self.set_axis_scale(self.y, scale, unit)

    def set_energy_axis(self, dispersion, unit, offset=0):
        """Set the energy scale, unit and offset"""
        self.set_axis_scale("channel", dispersion, unit, offset)

    def _create_child_stream(self, data, frames, process=None):
        """Create new spectrumstream from child data, frame number differs"""
        return create_new_spectrum_stream(data, self.width, self.height,
                                          self.pixelsize, self.pixelunit,
                                          self.channels, self.dispersion,
                                          self.energy_unit,
                                          self.spectrum_offset, frames,
                                          parent=self,
                                          process=process)

    def _create_child_map(self, data, process=None):
        """Data is in sparse format, turned into full frame"""
        arr = self._reshape_sparse_matrix(data, self.dimensions)
        return create_new_spectrum_map(arr, self.pixelsize, self.pixelunit,
                                       self.dispersion, self.energy_unit,
                                       self.spectrum_offset, parent=self,
                                       process=process)

    def get_frame(self, index):
        frmlst = self._get_frame_list()
        dt = frmlst[index]
        process = f"Selected spectrum stream frame {index}"
        return self._create_child_map(dt, process)

    def select_frames(self, indexes):
        frmlst = self._get_frame_list()
        process = f"Selected frames {indexes}"
        selected = [frmlst[i] for i in indexes]
        data = self._stack_frames(selected)
        return self._create_child_stream(data, len(selected), process=process)

    def exclude_frames(self, indexes):
        frmlst = self._get_frame_list()
        process = f"Excluded frames {indexes}"
        selected = [frmlst[i] for i in range(len(frmlst)) if i not in indexes]
        data = self._stack_frames(selected)
        return self._create_child_stream(data, len(selected), process=process)

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def frames(self):
        return self._get_axis_prop("frame", "bins")

    @property
    def pixelsize(self):
        return self._get_axis_prop(self.x, "scale")

    @property
    def pixelunit(self):
        return self._get_axis_prop(self.x, "unit")

    @property
    def width(self):
        return self._get_axis_prop(self.x, "bins")

    @property
    def height(self):
        return self._get_axis_prop(self.y, "bins")

    @property
    def framesize(self):
        return self.width*self.height

    @property
    def channels(self):
        return self._get_axis_prop("channel", "bins")

    @property
    def dispersion(self):
        return self._get_axis_prop("channel", "scale")

    @property
    def energy_unit(self):
        return self._get_axis_prop("channel", "unit")

    @property
    def spectrum_offset(self):
        return self._get_axis_prop("channel", "offset")

    def align_frames(self, shifts, inplace=False):
        """
        Perform rigid shifting of the individual frames

        These shifts can be calculated based on an image stack.

        Parameters
        ----------
        shifts : array-like, Nx2
            list of (delta_x, delta_y) positions to shift
        inplace : bool
            replace the data in this object. Default is False.

        Returns
        -------
        result : SpectrumStream
            derivative spectrumstream with aligned frames
        """
        sx, sy = shifts
        # correct the images
        newdat = []
        frmlst = self._get_frame_list()
        for j, i in tqdm(enumerate(frmlst)):
            frm = SpectrumStream._reshape_sparse_matrix(i, self.dimensions)
            frm = ndi.shift(frm, (0, sy[j], sx[j]))
            frm = SpectrumStream._to_sparse(frm)
            newdat.append(frm)
        newdat = SpectrumStream._stack_frames(newdat)
        process = "Aligned frames"
        return self._create_child_stack(inplace, newdat, self.pixelsize,
                                        self.pixelunit,
                                        parent=self, process=process)

    @property
    def scan_dimensions(self):
        return (self.width, self.height)

    def _get_frame_list(self, compress_type="csr"):
        '''
        Returns data as a list of frames, each frames a sparse matrix.
        '''
        dt = self.data.tocsr()  # we will do row slicing so it's faster
        toreturn = [change_compress_type(
                    dt[i*self.framesize:(i+1)*self.framesize, :],
                    compress_type=compress_type)
                    for i in range(self.frames)]
        return toreturn

    @staticmethod
    def _stack_frames(lst, compress_type="csr"):
        '''
        Converts a list of sparse matrixes into one by stacking vertically
        '''
        tostack = [i.tocsr() for i in lst]  # row stacking, change to csr
        toreturn = spa.vstack(tostack)
        return change_compress_type(toreturn, compress_type=compress_type)

    @property
    def dimensions(self):
        return (self.channels, self.height, self.width)

    @property
    def spectrum_map(self):
        '''
        Add all frames and return 3D matrix representing spectrum map
        '''
        frmdat = self._get_frame_sum(comp_type="csr")
        process = f"Sum of all frames"
        return self._create_child_map(frmdat, process)

    @staticmethod
    def _reshape_sparse_matrix(frmdat, dim):
        '''
        Return 3D full matrix representation from 2D representation

        If the 2D matrix has the shape (width*height,channels) then the
        returned matrix has shape (channel, y, x)
        '''
        cs, ys, xs = dim
        return frmdat.T.toarray().reshape(cs, ys, xs)

    @staticmethod
    def _to_sparse(frmdat):
        '''
        Return 2D sparse CSR representation from full 3D representation

        If the 3D matrix has the shape (channel, y, x) then the
        returned matrix has shape (width*height,channels)
        '''
        cs, ys, xs = frmdat.shape
        return spa.csr_matrix(frmdat.reshape(cs, ys*xs).T)

    def export_data(self, filename):
        '''Save the full dataset to an .npz file'''
        pre, _ = os.path.splitext(filename)
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path)
        filename = str(Path(pre + ".npz"))
        spa.save_npz(filename, self.data)
        self.metadata.to_file(str(Path(f"{pre}_meta.json")))

    def _get_frame_sum(self, comp_type="csr"):
        '''Return a sparse matrix sum of all frames'''
        data = self._get_frame_list()
        temp = 0
        for i in data:
            temp += i
        return change_compress_type(temp, comp_type)

    @staticmethod
    def _get_spectrum_sum(data):
        '''
        Return a sum spectrum of a sparse matrix

        The data of shape (xs*ys[*frames], channels) is turned into a 1D
        array with shape (channels,) by summing over all spatial coordinates
        '''
        data_sm = data.tocsr()
        return data_sm.sum(axis=0).getA1()

    @property
    def spectrum(self):
        '''Return the total spectrum in the map'''
        data = self._get_spectrum_sum(self.data)
        process = "Total spectrum"
        return create_new_spectrum(data, self.dispersion, self.energy_unit,
                                   self.spectrum_offset, parent=self,
                                   process=process)

    def export_streamframes(self, path, pre="Frame",
                            counter=None):
        '''
        Writes out the spectrumstream frame by frame in the .npz format

        Naming convention is pre_(00n).npz where the number of zeros is
        attempted to be guessed.

        Parameters
        ----------
        path : str
            path to folder where the files will be written
        pre : str, optional
            name prefix. Default is "Frame"
        counter : int, optional
            number of counter digits. Defaults to the minimum necessary.
        '''
        dt = self._get_frame_list()
        if not os.path.exists(path):
            os.makedirs(path)

        self.metadata.to_file(str(Path(f"{path}/{pre}_meta.json")))

        # set the number of digits in the counter
        mincounter = _get_counter(self.frames)
        if counter is None:
            counter = mincounter
        elif counter < mincounter:
            logger.error("Insufficient digits to represent the frames")
            return
        else:
            pass  # counter is already set to the right value

        for j, i in enumerate(dt):
            c = str(j).zfill(counter)
            name = "{}_{}".format(pre, c)
            spa.save_npz(str(Path(f"{path}/{name}")), i)


def import_file_to_spectrumstream(path):
    """
    Read in spectrumstream frames from .npz file

    Sparse matrix representations of the stream is imported
    from an .npz file.

    Parameters
    ----------
    path : str
        The path to the npz file.

    Returns
    -------
    matrix : SpectrumStream
        The reconstructed spectrumstream object
    """
    data = spa.load_npz(path)
    # get name of the file
    fp, _ = os.path.splitext(path)
    metapath = f"{fp}_meta.json"
    if os.path.isfile(metapath):
        meta = jt.read_json(metapath)
        metadata = mda.Metadata(meta)
        return SpectrumStream(data, metadata)
    else:
        logger.error(f"No metadata json file associated with the stream was"
                     f" found. Unfortunately the stream can't be recreated.")
        return


def import_files_to_spectrumstream(path):
    """
    Read in spectrumstream frames from .npz files

    Sparse matrix representations of stream frames are imported
    from .npz files and concatenated to one large data matrix.
    The expected file format is .*\\_[0-9]+\\.npz. If metadata
    is available in the folder it will also be imported.

    Parameters
    ----------
    path : str
        The path to the folder containing the streamframe files.

    Returns
    -------
    matrix : SpectrumStream
        The concatenated csr sparse matrix
    """
    files = os.listdir(path)
    pattern = re.compile(r"(.*)\_([0-9]+)\.npz")
    frame_names = []
    meta = ""  # name of the metadata file
    prefix = ""  # to be able to find the initial iteration
    for i in files:
        mt = pattern.match(i)
        if mt:
            if not prefix:
                prefix, _ = mt.groups()
            else:
                prefix_compare, _ = mt.groups()
                if prefix_compare != prefix:
                    raise ValueError("Not all frames in the folder have the"
                                     " same prefix, or other files match the "
                                     "pattern .*\\_[0-9]+\\.npz . "
                                     "Could not create Spectrumstream")
            frame_names.append(i)
        else:
            # it might be a JSON file
            pattern_meta = re.compile(r"(.*)\.json")
            match2 = pattern_meta.match(i)
            if match2:
                meta = i

    frame_names.sort()

    if frame_names:
        frames = []
        for i in frame_names:
            path_c = path+"/"+i
            im = spa.load_npz(path_c)
            frames.append(im)
        data = SpectrumStream._stack_frames(frames)
        if meta:
            meta = jt.read_json(path+"/"+meta)
            metadata = mda.Metadata(meta)
        else:
            logger.error(f"No metadata json file was found in the folder. "
                         f"Creating blank metadata and guessing axes "
                         f"dimensions...")
            metadata = mda.Metadata()
            metadata.experiment_type = "modified"
            metadata.process = (f"Imported from individual images in "
                                f"{path}")
            xbins = int(np.sqrt(frames[0].shape[0]))
            assert xbins**2 == frames[0].shape[0], "Shape of "
            channels = frames[0].shape[1]
            xinfo = (xbins, "pixels", 1)
            yinfo = (xbins, "pixels", 1)
            cinfo = (channels, "channels", 1, 0)
            metadata["data_axes"] = mda.gen_spectrum_stream_axes(xinfo, yinfo,
                                                                 len(frames),
                                                                 cinfo)
            metadata = mda.Metadata(metadata)
        return SpectrumStream(data, metadata)
    else:
        logger.error(f"No npz frames in {path}")
        return None
