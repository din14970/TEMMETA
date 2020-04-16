"""
This module provides a few utility decorators
"""

import time


def timeit(method):
    '''A decorator for timing the executing of certain methods'''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed
