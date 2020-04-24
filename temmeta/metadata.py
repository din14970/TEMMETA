"""
Module for creating standardized metadata
"""
from functools import reduce
import operator
import logging
import numpy as np

from . import jsontools as jt

# Initialize the Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DEGREE = u"\u00b0"
MU = u"\u03BC"


class DotDict(dict):
    '''
    Simple extension of dict to allow dot-access
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args):
        dict.__init__(self, *args)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def to_file(self, filepath):
        """Write out the data to a JSON file"""
        jt.write_to_json(filepath, self)

    def _get_by_path(self, path):
        '''
        Access and return part of the tree with list of keys/indices
        '''
        if isinstance(path, str):
            lst = path.strip().split("/")
        elif isinstance(path, list):
            lst = path
        else:
            raise TypeError("Invalid key type")
        return reduce(operator.getitem, lst, self)

    def _set_by_path(self, path, value, force=False):
        '''
        Set part of the tree using list of keys/indices

        If the path doesn't exist, it can be forcefully created.
        '''
        # does the path exist
        if isinstance(path, str):
            lst = path.strip().split("/")
        elif isinstance(path, list):
            lst = path
        else:
            raise TypeError("Invalid key type")
        # try to turn all integers into integers
        for j, i in enumerate(lst):
            try:
                lst[j] = int(i)
            except ValueError:
                pass
        # does the node exist?
        try:
            self._get_by_path(lst[:-1])[lst[-1]] = value
        except (KeyError, IndexError, TypeError):
            if force:
                # test if all elements are valid
                if [i for i in lst
                   if not isinstance(i, str) and not isinstance(i, int)]:
                    # some element is a non-integer or non-string
                    raise ValueError("There are invalid elements in this path")
                logger.debug("Finding where the path fails")
                for j, i in enumerate(lst):
                    # can be either an integer or a string for a valid
                    # path in json
                    try:
                        self._get_by_path(lst[:j+1])
                    except (KeyError, IndexError, TypeError):
                        exist = lst[:j]
                        new = lst[j:]
                        break
                logger.debug(f"{exist} are working keys")
                logger.debug(f"{new} path must be created")
                for j, i in enumerate(new):
                    try:
                        nextnode = new[j+1]
                    except IndexError:
                        # we reached the end of the path
                        nextnode = None

                    if isinstance(nextnode, int):
                        toadd = []
                    elif isinstance(nextnode, str):
                        toadd = DotDict()
                    else:
                        toadd = value

                    if isinstance(i, int):
                        # check if what we need to add to is a list
                        try:
                            self._get_by_path(exist+new[:j]).append(toadd)
                            new[j] = len(self._get_by_path(exist+new[:j]))-1
                            # because when we loop again the index will
                            # be wrong
                        except TypeError:
                            # some element is a non-integer or non-string
                            raise IndexError(f"{i} is an invalid index, "
                                             f"expected a string.")
                    if isinstance(i, str):
                        # check if what we need to add to is a dict
                        try:
                            self._get_by_path(exist+new[:j])[i] = toadd
                        except TypeError:
                            # some element is a non-integer or non-string
                            raise KeyError(f"{i} is an invalid key, "
                                           f"expected an integer.")
            else:
                raise KeyError(f"Can't locate the path {lst} inside "
                               f"dictionary. Aborting.")


class Metadata(DotDict):
    """
    Represents a JSON metadata dictionary
    """
    def __init__(self, *args):
        dict.__init__(self, *args)
        self._makedotdict(self)

    def _makedotdict(self, dic):
        if isinstance(dic, dict):
            for key, value in dic.items():
                dic[key] = self._makedotdict(value)
            return DotDict(dic)
        elif isinstance(dic, list):
            for j, i in enumerate(dic):
                dic[j] = self._makedotdict(i)
            return dic
        else:
            return dic

    def __repr__(self):
        return jt.get_pretty_dic_str(self)

    @staticmethod
    def _get_process_history(dic, pre=""):
        process = dic.process
        print(pre+"* "+str(process))
        parent = dic.parent_meta
        if parent is not None:
            if isinstance(parent, list):
                if len(parent) == 2:
                    # we expect a maximum of 2 parents
                    print(pre+"|\\")
                    Metadata._get_process_history(parent[1], "|"+pre)
                    Metadata._get_process_history(parent[0], pre)
                else:
                    raise TypeError(f"Only 2 parents are supported "
                                    f"(found {len(parent)})")
            elif isinstance(parent, dict):
                # we have one parent
                print(pre+"|")
                Metadata._get_process_history(parent, pre)
            else:
                raise TypeError(f"Unrecognized parent_meta type "
                                f"{type(parent)}")
        else:
            print(pre)

    def print_history(self):
        Metadata._get_process_history(self)


def numerical_value(v, u, integer=False, factor=1):
    try:
        v = float(v)*factor
        if integer:
            v = int(v)
    except TypeError:
        v = None
    return [v, u]


def aperture(order, typ, x, xu, y, yu, shape, **kwargs):
    if x is not None:
        xwu = numerical_value(x, xu)
    else:
        xwu = None
    if y is not None:
        ywu = numerical_value(y, yu)
    else:
        ywu = None
    ap = {
        "order": order,
        "type": typ,
        "position": {
            "x": xwu,
            "y": ywu,
        },
        "shape": shape,
        **kwargs
    }
    return DotDict(ap)


def lens(current, currentunit, **kwargs):
    le = {
        "current": numerical_value(current, currentunit),
        **kwargs
    }
    return DotDict(le)


def axis(bins, unit="", axistype="linear", scale=1, offset=0,
         function=None, lookuptable=None):
    assert isinstance(bins, int), "An integer number of bins must be provided."
    ax = DotDict()
    ax.unit = unit
    ax.bins = int(bins)
    ax.axistype = axistype
    if axistype == "linear":
        ax.scale = float(scale)
        ax.offset = float(offset)
    elif axistype == "function":
        try:
            bin = np.arange(bins)
            eval(function)
        except (TypeError, NameError):
            raise TypeError("Axis function must be a valid string")
        ax.function = function
    elif axistype == "lookuptable":
        if len(lookuptable) == bins:
            ax.lookup_table = lookuptable
        else:
            raise ValueError("The length of the lookuptable does not match "
                             "the number of bins")
    else:
        raise ValueError(f"Unrecognized axis type: {axistype}")
    return ax


def ax_combo(nav, sig):
    """
    Generate an axis combination dictionary
    """
    return {
        "navigation": nav,
        "signal": sig,
    }


def gen_image_axes(xinfo, yinfo):
    """
    Generate standardized image axes metadata

    Regular TEM image
    2 signal axes, x and y

    Parameters
    ----------
    xinfo : tuple
        (bins (int), unit (str), scale (float))
    yinfo : tuple
        (bins (int), unit (str), scale (float))

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    binx, unitx, scalx = xinfo
    biny, unity, scaly = yinfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": None,
            "signal": ["x", "y"]
        }
    ]
    dax.x = axis(binx, unitx, "linear", scale=scalx, offset=0)
    dax.y = axis(biny, unity, "linear", scale=scaly, offset=0)
    return dax


def gen_image_stack_axes(xinfo, yinfo, finfo):
    """
    Generate standardized TEM image stack axes metadata

    Regular TEM image
    2 signal axes, x and y
    1 nav (frame) OR 1 nav (time) OR 1 nav Tilt, etc

    Parameters
    ----------
    xinfo : tuple
        (bins (int), unit (str), scale (float))
    yinfo : tuple
        (bins (int), unit (str), scale (float))
    finfo : tuple
        (number of frames (int), time unit (str), exposure time (float))
        If no exposure time is provided no time axis is added.

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    framenum, unitt, exptime = finfo
    binx, unitx, scalx = xinfo
    biny, unity, scaly = yinfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": ["frame"],
            "signal": ["x", "y"]
        },
    ]
    dax.x = axis(binx, unitx, "linear", scale=scalx)
    dax.y = axis(biny, unity, "linear", scale=scaly)
    dax.frame = axis(framenum, None, "linear")
    if exptime is not None:
        dax.time = axis(framenum, unitt, "linear", scale=exptime)
        dax.combos.append({
                           "navigation": ["time"],
                           "signal": ["x", "y"]
                           })
    return dax


def left_to_right_top_to_bottom_scan(binx, biny, frms, pdt, lt, ft):
    """
    Create time axis metadata for standard scanning

    Parameters
    ----------
    binx : int
        number of scan positions in x-direction
    biny : int
        number of scan positions in y-direction
    frms : int
        number of frames
    pdt : float
        pixel dwell time
    lt : float
        time to scan a single line
    ft : float
        time to scan a single frame

    Returns
    -------
    axis : dict
        time axis metadata structure
    """
    pdt = float(pdt)
    ldt = float(lt) - pdt*binx
    if ldt < 0:
        ldt = 0
    frm = float(ft) - ldt*biny - pdt*binx*biny
    if frm < 0:
        frm = 0
    tfunc = (f"bin*{pdt}+bin//{binx}*{ldt}"
             f"+bin//{binx*biny}*{frm}")
    return axis(binx*biny*frms, "s", "function", function=tfunc)


def gen_scan_image_axes(xinfo, yinfo, time_axis=None):
    """
    Generate standardized scan image axes metadata

    Regular TEM image
    2 signal axes, x and y
    2 navigation axes, x and y
    1 navigation axis, time

    Parameters
    ----------
    xinfo : tuple
        (bins (int), unit (str), scale (float))
    yinfo : tuple
        (bins (int), unit (str), scale (float))
    time_axis : dict, optional
        Time axis corresponding to scan positions. Default is None.

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    binx, unitx, scalx = xinfo
    biny, unity, scaly = yinfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": None,
            "signal": ["scan_x", "scan_y"]
        },
        {
            "navigation": ["scan_x", "scan_y"],
            "signal": None
        }
    ]
    dax.scan_x = axis(binx, unitx, "linear", scale=scalx)
    dax.scan_y = axis(biny, unity, "linear", scale=scaly)
    if time_axis is not None:
        dax.time = time_axis
        dax.combos.append({
                           "navigation": ["time"],
                           "signal": None
                           },)
    return dax


def gen_scan_image_stack_axes(xinfo, yinfo, fnum, time_axis=None):
    """
    Generate standardized scan image stack axes metadata

    2 signal axes (x and y), 1 navigation axis (frame)
    3 navigation axes (x, y and frame)
    1 navigation axis (time)

    Parameters
    ----------
    xinfo : tuple
        (bins (int), unit (str), scale (float))
    yinfo : tuple
        (bins (int), unit (str), scale (float))
    fnum : int
        number of frames (int)
    time_axis : dict, optional
        Time axis corresponding to scan positions. Default is None.

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    binx, unitx, scalx = xinfo
    biny, unity, scaly = yinfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": ["frame"],
            "signal": ["scan_x", "scan_y"]
        },
        {
            "navigation": ["scan_x", "scan_y", "frame"],
            "signal": None
        },
    ]
    dax.scan_x = axis(binx, unitx, "linear", scale=scalx)
    dax.scan_y = axis(biny, unity, "linear", scale=scaly)
    dax.frame = axis(fnum, None, "linear")
    if time_axis is not None:
        dax.time = time_axis
        dax.combos.append({
                           "navigation": ["time"],
                           "signal": None
                           })
    return dax


def gen_profile_axes(xinfo):
    """
    Generate standardized intensity profile

    1 nav axis x or 1 signal axis x

    Parameters
    ----------
    xinfo : tuple
        (bins (int), unit (str), scale (float))

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    binc, unitc, disp = xinfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": ["x"],
            "signal": None
        },
        {
            "navigation": None,
            "signal": ["x"]
        },
    ]
    dax.x = axis(binc, unitc, "linear", scale=disp)
    return dax


def gen_spectrum_axes(cinfo):
    """
    Generate standardized single spectrum metadata

    1 signal axis channel

    Parameters
    ----------
    cinfo : tuple
        (bins (int), unit (str), dispersion (float), spectrum offset)

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    binc, unitc, disp, offset = cinfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": None,
            "signal": ["channel"]
        },
    ]
    dax.channel = axis(binc, unitc, "linear", scale=disp, offset=offset)
    return dax


def gen_spectrum_line_axes(linfo, cinfo, time_axis=None):
    """
    Generate standardized spectrum line metadata

    1 signal axis channel + 1 navigation length
    1 signal channel + 1 nav time

    Parameters
    ----------
    linfo : tuple
        (bins (int), unit (str), scale (float))
    cinfo : tuple
        (bins (int), unit (str), dispersion (float), spectrum offset)
    time_axis : dict, optional
        Time axis corresponding to scan positions. Default is None.

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    binc, unitc, disp, offset = cinfo
    binl, unitl, scall = linfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": ["scan_l"],
            "signal": ["channel"]
        },
    ]
    dax.scan_l = axis(binl, unitl, "linear", scale=scall)
    dax.channel = axis(binc, unitc, "linear", scale=disp, offset=offset)
    if time_axis is not None:
        dax.time = time_axis
        dax.combos.append({
                           "navigation": ["time"],
                           "signal": ["channel"]
                           })
    return dax


def gen_spectrum_map_axes(xinfo, yinfo, cinfo, time_axis=None):
    """
    Generate standardized rectangular spectrum map metadata

    1 signal axis channel + 2 navigation (x and y)
    1 signal channel + 1 navigation (time)

    Parameters
    ----------
    xinfo : tuple
        (bins (int), unit (str), scale (float))
    yinfo : tuple
        (bins (int), unit (str), scale (float))
    cinfo : tuple
        (bins (int), unit (str), dispersion (float), spectrum offset)
    time_axis : dict, optional
        Time axis corresponding to scan positions. Default is None.

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    binx, unitx, scalx = xinfo
    biny, unity, scaly = yinfo
    binc, unitc, disp, offset = cinfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": ["scan_x", "scan_y"],
            "signal": ["channel"]
        },
    ]
    dax.scan_x = axis(binx, unitx, "linear", scale=scalx)
    dax.scan_y = axis(biny, unity, "linear", scale=scaly)
    dax.channel = axis(binc, unitc, "linear", scale=disp, offset=offset)
    if time_axis is not None:
        dax.time = time_axis
        dax.combos.append({
                           "navigation": ["time"],
                           "signal": ["channel"]
                           })
    return dax


def gen_spectrum_stream_axes(xinfo, yinfo, fnum, cinfo, time_axis=None):
    """
    Generate standardized spectrum stream metadata

    1 signal axis channel + 3 navigation (x, y, frame)
    1 signal channel + 1 navigation (time)

    Parameters
    ----------
    xinfo : tuple
        (bins (int), unit (str), scale (float))
    yinfo : tuple
        (bins (int), unit (str), scale (float))
    fnum : int
        number of frames
    cinfo : tuple
        (bins (int), unit (str), dispersion (float), spectrum offset)
    time_axis : dict, optional
        Time axis corresponding to scan positions. Default is None.

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    binx, unitx, scalx = xinfo
    biny, unity, scaly = yinfo
    binc, unitc, disp, offset = cinfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": ["scan_x", "scan_y", "frames"],
            "signal": ["channel"]
        },
    ]
    dax.frame = axis(fnum, None, "linear")
    dax.scan_x = axis(binx, unitx, "linear", scale=scalx)
    dax.scan_y = axis(biny, unity, "linear", scale=scaly)
    dax.channel = axis(binc, unitc, "linear", scale=disp, offset=offset)
    if time_axis is not None:
        dax.time = time_axis
        dax.combos.append({
                           "navigation": ["time"],
                           "signal": ["channel"]
                           })
    return dax


def gen_scan_image_map_axes(xinfo, yinfo, s_xinfo, s_yinfo, time_axis=None):
    """
    Generate standardized scan image map (e.g. 4D stem or PED) metadata

    2 signal axes (x, y) + 2 navigation (scan_x, scan_y)
    2 signal axes (x, y) + 1 navigation (time)

    Parameters
    ----------
    xinfo : tuple
        Image x-axis (bins (int), unit (str), scale (float))
    yinfo : tuple
        Image y-axis (bins (int), unit (str), scale (float))
    s_xinfo : tuple
        Scan x-axis (bins (int), unit (str), scale (float))
    s_yinfo : tuple
        Scan y-axis (bins (int), unit (str), scale (float))
    time_axis : dict, optional
        Time axis corresponding to scan positions. Default is None.

    Returns
    -------
    dax : dict
        dictionary containing axis information
    """
    binx, unitx, scalx = xinfo
    biny, unity, scaly = yinfo
    s_binx, s_unitx, s_scalx = s_xinfo
    s_biny, s_unity, s_scaly = s_yinfo
    dax = DotDict()
    dax.combos = [
        {
            "navigation": ["scan_x", "scan_y"],
            "signal": ["x", "y"]
        },
        {
            "navigation": ["time"],
            "signal": ["x", "y"]
        },
    ]
    # the units of the images
    dax.x = axis(binx, unitx, "linear", scale=scalx)
    dax.y = axis(biny, unity, "linear", scale=scaly)
    # the units of the scan
    dax.scan_x = axis(s_binx, s_unitx, "linear", scale=s_scalx)
    dax.scan_y = axis(s_biny, s_unity, "linear", scale=s_scaly)
    if time_axis is not None:
        dax.time = time_axis
        dax.combos.append({
                           "navigation": ["time"],
                           "signal": ["x", "y"]
                           })
    return dax


def gen_dax(datatype, *args):
    """Wrapper function for generating axes metadata"""
    if datatype == "image":
        return gen_image_axes(*args)

    if datatype == "image_series":
        return gen_image_stack_axes(*args)

    if datatype == "scan_image":
        return gen_scan_image_axes(*args)

    if datatype == "scan_image_series":
        return gen_scan_image_stack_axes(*args)

    if datatype == "spectrum_point":
        return gen_spectrum_axes(*args)

    if datatype == "spectrum_line":
        return gen_spectrum_line_axes(*args)

    if datatype == "spectrum_map":
        return gen_spectrum_map_axes(*args)

    if datatype == "spectrum_stream":
        return gen_spectrum_stream_axes(*args)

    if datatype == "scan_image_map":
        return gen_scan_image_map_axes(*args)
