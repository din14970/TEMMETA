"""
Module containing a number of utilities relevant to TEM

Functions
---------
getElectronWavelength
    Get the electron wavelength in m from the high voltage in V
"""

import math


def getElectronWavelength(ht):
    """Get the electron wavelength in m from the high voltage in V"""
    # ht in Volts, length unit in meters
    h = 6.6e-34
    m = 9.1e-31
    charge = 1.6e-19
    c = 3e8
    wavelength = h / math.sqrt(2 * m * charge * ht)
    relativistic_correction = 1 / math.sqrt(1 + ht * charge/(2 * m * c * c))
    return wavelength * relativistic_correction
