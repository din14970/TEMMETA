from scipy.signal import find_peaks


def get_spectrum_peaks(arr, **kwargs):
    """
    Wrapper for scipy.signal.find_peaks

    Parameters
    ----------
    arr : array-like 1D
        1D array representing a spectrum to find peaks.

    Other parameters
    ----------------
    kwargs: optional key word args passed to find_peaks
    """
    return find_peaks(arr, **kwargs)
