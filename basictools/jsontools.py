"""
Functions to work with json nested dictionaries

Functions
---------
write_to_json
    Write a nested dictionary to a json file
read_json
    Import a json file and interpret as nested dictionary
get_pretty_dic_str
    Make a nice indented string representation of a nested dictionary
print_pretty
    Print a nice indented string representation of a nested dictionary
find_key
    Find the first key corresponding to a value in a dictionary
search_json_kv
    Search a nested dictionary for a key-value pair. Returns True if found.
search_json
    Searches a nested dictionary for some key word. Both keys and values are
    compared against the keyword
"""
import json
from pathlib import Path


def write_to_json(filepath, dic):
    """Write dict out as JSON file"""
    with open(str(Path(filepath)), "w") as f:
        f.write(get_pretty_dic_str(dic))


def read_json(filepath):
    """Return dictionary from JSON file"""
    with open(str(Path(filepath)), "r") as f:
        dic = json.load(f)
    return dic


def get_pretty_dic_str(dic):
    """Get a sorted and indented json string from a dict"""
    return json.dumps(dic, indent=4, sort_keys=True)


def print_pretty(dic):
    """Print a sorted and indented json string from a dict"""
    print(get_pretty_dic_str(dic))


def find_key(dic, value):
    """Return the first key of a dict corresponding to a value"""
    for k, v in dic.items():
        if v == value:
            return k


def search_json_kv(dic, kw, va):
    """
    Search whether a key/value pair exists in a nested dict/array

    Match can only occur in a nested dictionary

    Parameters
    ----------
    dic : dict, array-like
        Nested dictionary/array structure to search
    kw : str
        Keyword, in JSON always a string
    va : object
        Value corresponding to keywords

    Returns
    -------
    result : bool
        Whether the key value pair exists in dic
    """
    if isinstance(dic, dict):
        if kw in dic:
            if dic[kw] == va:
                return True
            else:
                return False
        for k, v in dic.items():
            if isinstance(v, dict) or isinstance(v, list):
                item = search_json_kv(v, kw, va)
                if item is not False:
                    return item
    if isinstance(dic, list):
        for i in dic:
            if isinstance(v, dict) or isinstance(v, list):
                item = search_json_kv(v, kw, va)
                if item is not False:
                    return item
    else:
        return False


def search_json(dic, kw):
    """
    Search whether a keyword exists in a nested dict/array

    Both keys and values are checked. The key and value are returned
    If the value is found in an array, the value and the array are
    returned

    Parameters
    ----------
    dic : dict, array-like
        Nested dictionary/array structure to search
    kw : str
        Keyword, in JSON always a string

    Returns
    -------
    result : tuple
        Key and value or value and array are returned
    """
    if isinstance(dic, dict):
        if kw in dic:
            return kw, dic[kw]
        if kw in dic.values():
            key = find_key(dic, kw)
            return key, dic[key]
        for k, v in dic.items():
            if isinstance(v, dict) or isinstance(v, list):
                item = search_json(v, kw)
                if item is not None:
                    return item
    if isinstance(dic, list):
        if kw in dic:
            return kw, dic
        for i in dic:
            if isinstance(v, dict) or isinstance(v, list):
                item = search_json(v, kw)
                if item is not None:
                    return item
