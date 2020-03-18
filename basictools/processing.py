#! python3

from scipy.signal import find_peaks


def get_spectrum_peaks(stream, **kwargs):
    return find_peaks(stream.tot_spectrum, **kwargs)
