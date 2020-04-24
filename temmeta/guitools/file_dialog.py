"""
A working file dialog on Mac OSX because it doesn't play well with Tk

Does not work from jupyter notebooks, keyboard does not connect.
"""
from .widgets import OpenFileDialog, SaveFileDialog
from PyQt5.QtWidgets import QApplication


def open(directory='./', filters="All files (*)"):
    """Open a file dialog, starting in the given directory, and return
    the chosen filename"""
    app = QApplication([directory])
    fd = OpenFileDialog()
    app.quit()
    return fd.filename


def save(directory='./', filters="All files (*)"):
    app = QApplication([directory])
    fd = SaveFileDialog()
    app.quit()
    return fd.filename
