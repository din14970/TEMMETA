from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget


class SaveFileDialog(QWidget):

    def __init__(self, filters="All Files (*)"):
        super().__init__()
        self.title = "Save file"
        self.filters = filters
        self.filename = self.saveFileDialog()

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(caption=self.title,
                                                  filter=self.filters,
                                                  options=options)
        return filename


def get_fname(directory='./', filters="All files (*)"):
    app = QApplication([directory])
    fd = SaveFileDialog(filters=filters)
    app.quit()
    return fd.filename


if __name__ == "__main__":
    s = get_fname()
    print(s)
