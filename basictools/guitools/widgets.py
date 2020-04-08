from PyQt5.QtWidgets import (QWidget, QFileDialog)


class OpenFileDialog(QWidget):

    def __init__(self, title="Save file", filters="All Files (*)"):
        super().__init__()
        self.title = title
        self.filters = filters
        self.filename = self.openFileDialog()

    def openFilesDialog(self):
        filename, _ = QFileDialog.getOpenFileName(caption=self.title,
                                                  filter=self.filters)
        return filename


class OpenFilesDialog(QWidget):

    def __init__(self, title="Save file", filters="All Files (*)"):
        super().__init__()
        self.title = title
        self.filters = filters
        self.filename = self.openFilesDialog()

    def openFilesDialog(self):
        filenames, _ = QFileDialog.getOpenFileNames(caption=self.title,
                                                    filter=self.filters)
        return filenames


class SaveFileDialog(QWidget):

    def __init__(self, title="Save file", filters="All Files (*)"):
        super().__init__()
        self.title = title
        self.filters = filters
        self.filename = self.saveFileDialog()

    def saveFileDialog(self):
        filename, _ = QFileDialog.getSaveFileName(caption=self.title,
                                                  filter=self.filters)
        return filename
