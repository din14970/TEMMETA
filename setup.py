from setuptools import setup, find_packages

setup(
    name="temmeta",
    version="0.0.1",
    description="TEMMETA is a library for transmission electron microscopy (TEM) (meta)data manipulation",
    url='https://github.com/din14970/TEMMETA',
    author='Niels Cautaerts',
    author_email='nielscautaerts@hotmail.com',
    license='GPL-3.0',

    classifiers=['Development Status :: 5 - Production/Stable',
                 'Topic :: Scientific/Engineering :: Physics',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7'],

    keywords='TEM',
    packages=find_packages(exclude=["*tests*", "*examples*"]),
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scipy',
        'jupyter',
        'hyperspy',
        'matplotlib_scalebar',
        'tqdm',
        'Pillow',
        'opencv-python',
        'PyQt5'
    ],
)
