"""
This module contains a number of simple algebra functions which might
be useful in a number of places
"""
import numpy as np


def rotangle(x1, y1, x2, y2):
    """Get angle in degrees between two points w.r.t. the positive x-axis"""
    le = distance(x1, y1, x2, y2)
    theta = np.arccos((x2-x1)/le)/(2*np.pi)*360
    if y1 == y2:
        return theta
    else:
        return theta*np.sign(y2-y1)


def distance(x1, y1, x2, y2):
    """Get the euclidian distance between two points"""
    le = np.sqrt((x2-x1)**2+(y2-y1)**2)
    return le
