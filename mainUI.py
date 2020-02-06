import logging
import os
import sys
from time import sleep

from PyQt5 import uic
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGraphicsScene, QGraphicsPixmapItem,
QErrorMessage, QMessageBox, QPlainTextEdit, QMainWindow, QTreeWidgetItem)
from PyQt5.QtGui import QImage, QPixmap

from basictools import data_io as io

gui, Window = uic.loadUiType("maingui.ui")

supported_files =   "\
                    EMD (*.emd);;\
                    DM3 (*.dm3);;\
                    EMI (*.dm3);;\
                    TVIPS (*.tvips);;\
                    BLO (*.blo);;\
                    TIFF (*.tiff *.tif)\
                    "

class ConnectedWidget(gui):


    def __init__(self, window):
        super().__init__()
        self.window = window
        self.setupUi(window)
        self.connectUI()

    def connectUI(self):
        self.actionOpen_2.triggered.connect(self.openFile)
        self.actionOpen_many.triggered.connect(self.openManyFiles)


    def openFile(self):
        #open the file browser
        filepath = self.openFileBrowser(supported_files)
        if filepath:
            #try opening the file with known methods
            logging.debug("Trying to open file: {}".format(filepath))
            f = io.read_emd(filepath)
            if f:
                logging.debug("File was succesfully opened")
                filestructure = io.get_emd_tree_view(f)
                root = filestructure[(0,0)]
                fwid = QTreeWidgetItem(self.treeWidget, root)
                for k, v in filestructure.items():
                    if k[0]>0:
                        QTreeWidgetItem(fwid, v)
            else:
                logging.debug("Something wrong with this file, could not open.")


    def parseFile(self, filepath):
        #check the file extension
        path, ext = os.path.splitext(filepath)
        if ext == "emd":
            f = io.read_emd(filepath)
            return f
        else:
            return None


    def openManyFiles(self):
        #open the folder browser
        folderpath = self.openFolderBrowser()
        if folderpath:
            print(folderpath)

    def openFileBrowser(self, fs):
        path, okpres = QFileDialog.getOpenFileName(caption = "Select file", filter = fs)
        if okpres:
            return path
        else:
            return None

    def openFolderBrowser(self):
        path = QFileDialog.getExistingDirectory(caption = "Choose directory")
        if path:
            return path
        else:
            return None

def main():
    app = QApplication([])
    window = Window()
    form = ConnectedWidget(window)
    window.setWindowTitle("Universal TEM data manipulator")
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
