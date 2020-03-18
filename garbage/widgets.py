#!../.venv/bin python3

from PyQt5.QtWidgets import (QApplication, QFileDialog, QMainWindow, QWidget, QCheckBox,
        QComboBox, QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout)

from PyQt5.QtCore import Qt

import sys
import os

from modules import io

#size policies
spbutton = QSizePolicy()
spbutton.setVerticalPolicy(QSizePolicy.Fixed)
spbutton.setHorizontalPolicy(QSizePolicy.Fixed)

spbox = QSizePolicy()
spbox.setVerticalPolicy(QSizePolicy.Maximum)
spbox.setHorizontalPolicy(QSizePolicy.Minimum)

class GetPathWidget(QWidget):

    def __init__(self, master, label, isfile = True, filetype = None):
        super().__init__()
        self.master = master
        self.filetype = filetype
        bx1layout = QHBoxLayout()
        self.label = label
        bx1layout.addWidget(QLabel(self.label))
        self.inpath = QLineEdit()
        self.inpath.textChanged.connect(self.update)
        bx1layout.addWidget(self.inpath)
        self.browsebut = QPushButton("Browse")
        self.browsebut.setSizePolicy(spbutton)
        self.browsebut.clicked.connect(self.open_file)
        bx1layout.addWidget(self.browsebut)
        self.setLayout(bx1layout)

        self.isfile = isfile
        self.ive = False


    def open_file(self):
        '''Open a file dialog and update the text in the textbox'''
        if self.isfile:
            path, okpres = QFileDialog.getOpenFileName(caption = "Select {}".format(self.label), filter = self.filetype)
            if okpres:
                self.inpath.setText(path)
        else:
            path = QFileDialog.getExistingDirectory(caption = "Choose directory")
            if path:
                self.inpath.setText(path)



    def update(self):
        '''Update internally on whether the pathname actually exists and is valid'''
        if self.isfile:
            if os.path.exists(self.path) and os.path.isfile(self.path):
                self.ive = True
                self.inpath.setStyleSheet("color: green;")
                self.master.update()
            else:
                self.ive = False
                self.inpath.setStyleSheet("color: red;")
                self.master.update()
        else:
            if os.path.exists(self.path) and os.path.isdir(self.path):
                self.ive = True
                self.inpath.setStyleSheet("color: green;")
                self.master.update()
            else:
                self.ive = False
                self.inpath.setStyleSheet("color: red;")
                self.master.update()


    @property
    def path(self):
        return self.inpath.text()


class ConvertEMDdialog(QDialog):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Extract series from EMD files")
        self.create_layout()
        self.setFixedSize(self.sizeHint())
        self.fi = None
        #self.setAlignment(Qt.AlignTop)

    def create_layout(self):
        '''Create all the widgets in the dialog'''
        self.mainLayout = QVBoxLayout()

        #input group box
        box1 = QGroupBox("Input")
        box1.setSizePolicy(spbox)
        box1ly = QGridLayout()
        #box1ly.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.w1 = GetPathWidget(self, "EMD file", isfile = True,
                            filetype = "EMD files (*.emd)")
        box1ly.addWidget(self.w1, 1, 1, 1, 4)
        #the dataset selection
        box1ly.addWidget(QLabel("Dataset"), 2, 1)
        self.w2 = QComboBox()
        self.w2.currentIndexChanged.connect(self.update_selected)
        box1ly.addWidget(self.w2, 2, 2)
        box1ly.addWidget(QLabel("Shape"), 2, 3)
        self.w3 = QLineEdit()
        self.w3.setReadOnly(True)
        box1ly.addWidget(self.w3, 2, 4)

        box1.setLayout(box1ly)
        self.mainLayout.addWidget(box1)

        #output box
        box2 = QGroupBox("Output")
        box2.setSizePolicy(spbox)
        box2ly = QGridLayout()
        #the
        self.wo1 = GetPathWidget(self, "Output folder", isfile = False)
        box2ly.addWidget(self.wo1, 1, 1, 1, 4)
        box2ly.addWidget(QLabel("Prefix"), 2, 1)
        self.wo2 = QLineEdit()
        self.wo2.setMaxLength(20)
        box2ly.addWidget(self.wo2, 2, 2, 1, 3)
        self.wo3 = QCheckBox("Scalebar")
        self.wo3.setEnabled(False)
        self.wo4 = QCheckBox("Save metadata")
        self.wo4.setEnabled(False)
        box2ly.addWidget(self.wo3, 3, 1, 1, 2)
        box2ly.addWidget(self.wo4, 3, 3, 1, 2)
        box2.setLayout(box2ly)
        self.mainLayout.addWidget(box2)
        self.setLayout(self.mainLayout)

        box3 = QGroupBox("")
        box3.setSizePolicy(spbox)
        box3ly = QHBoxLayout()
        box3ly.setAlignment(Qt.AlignRight)
        self.but_conv = QPushButton("Extract")
        self.but_conv.setSizePolicy(spbutton)
        self.but_conv.clicked.connect(self.extract_clicked)
        self.but_conv.setToolTip("Test")
        but_canc = QPushButton("Close")
        but_canc.setSizePolicy(spbutton)
        but_canc.clicked.connect(self.close)
        box3ly.addWidget(self.but_conv)
        box3ly.addWidget(but_canc)
        self.statusline = QLineEdit()
        self.statusline.setReadOnly(True)
        self.statusline.setText("")
        #self.statusline.setSizePolicy(spbox)
        box3ly.addWidget(self.statusline)
        box3.setLayout(box3ly)
        self.mainLayout.addWidget(box3)

    def update(self):
        '''Update called when changing the path'''
        if self.w1.ive: #valid input path
            self.fi = io.open_emd(self.w1.path)
            self.generate_combobox()
        else:
            self.fi = None
            self.w2.clear()

    def update_selected(self):
        '''Update called when changing the value in the combobox'''
        #updating the shape field
        try:
            path = self.w2.currentText()
            fs = self.fi["Data"][path]["Data"].shape
            self.w3.setText("{}".format(fs))
        except:
            self.w3.setText("")

        #if it's an image, make scalebar and others accessible
        bool1 = self.w2.currentText().startswith("Image/")
        self.wo3.setEnabled(bool1)
        bool2 = (self.w2.currentText()=="")
        self.wo4.setEnabled(not bool2)



    def generate_combobox(self):
        '''Add datasets fo the combobox from the emd file'''
        self.w2.clear()
        lst = []
        for k, _ in self.fi["Data"].items():
            for sk, _ in self.fi["Data"][k].items():
                self.w2.addItem("{}/{}".format(k, sk))

    def extract_clicked(self):
        '''Action when clicking the extract button'''
        disp = ""
        #is there a file?
        if self.fi is not None:
            if (self.w2.currentText().startswith("Image/") or
                self.w2.currentText().startswith("SpectrumStream/")):
                if self.wo1.ive:
                    if self.wo2.text()!="":
                        disp = self.extract_data()
                    else:
                        disp = "Invalid prefix"
                else:
                    disp = "Invalid output path"
            else:
                disp = "Only image and spectrumstream datasets supported"
        else:
            disp = "Invalid EMD file"
        self.statusline.setText(disp)
        self.statusline.update()


    def extract_data(self):
        # try:
            #images
        if self.w2.currentText().startswith("Image/"):
            f = self.fi
            _, dn =  self.w2.currentText().split("/")
            pfx = self.wo2.text()
            pth = self.wo1.path
            sb = self.wo3.checkState()
            sm = self.wo4.checkState()
            dt = [f, dn, pfx, pth, sb, sm]
            print(dt)
            io.save_all_image_frames_uuid(f, dn, name=pfx, path=pth,
                           scale_bar = sb, show_fig = False, dpi = 100, save_meta = sm,
                           sb_settings = {"location":'lower right', "color" : 'k', "length_fraction" : 0.15},
                           imshow_kwargs = {"cmap" : "Greys_r"})
        elif self.w2.currentText().startswith("SpectrumStream/"):
            pass
        else:
            return "Unknown error"
        return "Succesfully extracted data"
        # except:
        #     return "Unknown error"



def main():
    app = QApplication(sys.argv)
    window = ConvertEMDdialog()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
