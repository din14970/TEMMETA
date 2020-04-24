# TEMMETA

TEMMETA is a library for transmission electron microscopy (TEM) (meta)data manipulation. The aim is to offer a one stop place for very basic to intermediate level operations on (S)TEM data, and be a kind of python version of ImageJ + Digital Micrograph + Velox.

**Author**: Niels Cautaerts, [nielscautaerts@hotmail.com](mailto:nielscautaerts@hotmail.com)

**Last updated**: 18/04/2020

**Try it now**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/din14970/TEMMETA/master?filepath=examples%2FTEMMETA%20demonstration.ipynb)

## What can TEMMETA do?

* For **images**:
	* plotting, rebinning, linear scaling, cropping, filtering...
	* Perform Fourrier filtering by calculating fast-fourrier-		transforms (FFTs), contruct FFT masks and calculating inverse 		fourrier transforms (IFFT)
	* Geometric phase analysis (GPA) [(Hytch et al., 1998)](http://doi.org/10.1016/S0304-3991(98)00035-7) on HRTEM images
	* Finding atomic column peak positions in HR-STEM images and 		fitting Gaussians to them
	* Calculating and plotting intensity line profiles along 		arbitrary directions in images
	* Export to png, tiff, hyperspy dataset

* For **image stacks**:
	* Interactively browsing through the frames with sliders
	* Rebinning, linear scaling, cropping, filtering, of all the 		frames
	* Selecting and excluding frames in an image stack
	* Aligning frames in an image stack with cross-correlation
	* Averaging all the frames to create one image
	* Export frames to png's or full dataset to hyperspy

* For **spectral data**:
	* select/exclude and align frames in SpectrumStream
	* Interactively browse through spectral map with sliders
	* Create images of spectrum maps at specific energies
	* Crop and rebin spectral maps
	* Condense areas to single spectra
	* Create line profiles along arbitrary directions in spectrum maps
	* Find peaks in spectra
	* Export to hyperspy

* For **all datasets**:
	* Support for automated scalebars, keeping track of axis scales, 		units and offsets
	* Keeping all processing history inside the metadata

Currently **only Velox .emd** files can be read in natively. One can still use the library for data coming from other files, but other tools will be needed to import those files and convert to the TEMMETA objects.

## How do I use TEMMETA?

### Prerequisites

* You have [Anaconda](https://www.anaconda.com/distribution/) installed.

### Set-up steps

1. Create a new virtual environment on your system to install TEMMETA in

	```
	$ conda create --name temmeta
	```

2. Activate the virtual environment anywhere in your system with

	```
	$ conda activate temmeta
	```

3. Pip install temmeta

	```
	$ pip install temmeta
	```

### Usage

1. With the environment activated, start Jupyter Notebook with

	```
	$ jupyter notebook
	```

2. Import `TEMMETA` modules like

	```
	from temmeta import data_io as dio
	from temmeta import image_filters as imf
	```

For help on how to use `TEMMETA`, follow the example jupyter notebook file `examples/TEMMETA demonstration.ipynb`. For most commands I tried to add a sufficiently descriptive docstring which you can access with `help(<command name>)`.

## History
TEMMETA started as a simple tool to read and convert Velox EMD data. As I became more familiar with the analysis needs of other people and I started seeing the limitations of other tools, it became clear TEMMETA needed to become more useful and easy for other people to use, especially those with limited python experience. Therefore I completely rewrote the library and modeled the tools after those available in popular software such as Digital Micrograph and ImageJ. TEMMETA still does not have a user interface, but the functions and classes should be familiar and easy enough for people that they can start analyzing their data in no time inside a Jupyter Notebook.

## TODOs
### Functionality
* support for other file types: dm3/dm4 (Gatan), SER/EMI (FEI), TVIPS, TIFF, .blo
* lazy operations on the emd dataset for exporting image frames without loading entire dataset into memory.
* implement strain mapping using atomic peaks
* find peaks in FFT's and diffraction patterns
* support 4D-STEM and PED datasets
* exporting and importing datasets to a single file
* reevaluate metadata structure
* applying shifts to spectrum stream frames is currently impossibly 	slow.
* deeper analysis of EELS/EDX data, implement a periodic table for some basic quantification and element detection.
* implement more filters

### Ease of use
* compile documentation with Sphinx

### Structure and format
* significant refactoring is necessary. Currently most functionality is in the basictools/data_io module - this file is a monster. There is a lot of repeated code which can be factored out with multiple inheritance.
* documentation must be updated and refined in most places
* unit tests must be created for most modules and methods

## How can I contribute?

You may always contribute to the documentation and writing unit tests. I like [`pytest`](https://docs.pytest.org/en/latest/) but don't have much experience with it myself yet. UI tools to interface with the methods would also be appreciated. If you know a faster way to make a method run, definitely contribute.
