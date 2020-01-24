mjoindices - A python package for calculating the Madden-Julian-Oscillation OMI index
=====================================================================================

[![DOI](https://zenodo.org/badge/197774253.svg)](https://zenodo.org/badge/latestdoi/197774253)

Overview
--------

mjoindices is a Python package relevant for users of the atmospheric sciences. It provides functionality to compute an 
index of the Madden-Julian-Oscillation (MJO), particularly the OLR-based MJO index (OMI).

Whereas the package name has been chosen to indicate that further MJO indices should be included in future, the 
implementation is currently limited to the OMI algorithm.

Citation
--------
If you use this software package, you can currently cite the [Zenodo DOI](http://dx.doi.org/10.5281/zenodo.3613752). 
It is planned to publish a peer-reviewed software meta paper in the near-future. Please ask us for the status of this 
paper, if you use mjoindices in published research. 
The OMI algorithm itself is described in [Kiladis (2014)](https://doi.org/10.1175/MWR-D-13-00301.1), but please don't 
forget to also cite the software package, which is an independent development.

Requirements
------------
mjoindices is written for Python 3 and depends on the packages NumPy, Pandas, SciPy, and Matplotlib. It runs on Linux
and Windows. Other operating systems have not been tested. 

Installation
------------
mjoindices in available in the [Python Package Index (PyPI)](https://pypi.org/). It can be installed using, 
e.g., pip.
    
    pip3 install mjoindices
    
It can also be installed from the source, which is available on [GitHub](https://github.com/cghoffmann/mjoindices). 
Download the source, move into the directory containing the file setup.py and run

    python3 setup.py install
    
API documentation
-----------------
The API documentation is found on [GitHub Pages](https://cghoffmann.github.io/mjoindices/index.html) and in the docs
folder of the [source](https://github.com/cghoffmann/mjoindices/tree/master/docs).
    
Getting started / examples
--------------------------
After you have installed mjoindices, you can download an
[example](https://github.com/cghoffmann/mjoindices/tree/master/examples) from the source, which consists of two files: 

* recalculate_original_omi.py: After downloading some data files, which are mentioned and linked in the source
documentation of the example, you can run this example to recalculate the original OMI values. The script will save
the computed Empirical Orthogonal Functions (EOFs) and the Principal Components (PCs) in two individual files, which
can also be configured in the source code. In addition, it will save a few plots into a directory, which can
also be configured in the source. These plots show the agreement with the original OMI values (slight deviations are 
expected due to numerical differences. This will be detailed in the corresponding software meta paper).

    Note that you can use this example also as a template to calculate OMI values with your own OLR data. 
In order to do that, you have to adapt only two parts of the code, which are also marked in the code documentation.

    Note also that this script may run for one or two hours on common personal computer systems.

* evaluate_omi_reproduction.py: This script produces more detailed comparison plots and saves them into a directory.
The script recalculate_original_omi.py has to be run before, as the evaluation script is based on the saved results.
As for recalculate_original_omi.py, some file and directory names have to be adapted in the beginning of the code.

Both files are also available as Jupyter notebook files.

Automated testing
-----------------
After you have installed mjoindices, you can also download
[unit and integration tests](https://github.com/cghoffmann/mjoindices/tree/master/tests) from the source to check
your installation with pytest.

* Download the complete test directory to you local file system.

* Download some external input and reference data files. Which files have to be downloaded and where they have to be
placed is described in a separate [Readme file](https://github.com/cghoffmann/mjoindices/blob/master/tests/testdata/README).

* Move into your local test directory and run

        pytest
        
Note that the tests may run for a few hours on a common personal computer.