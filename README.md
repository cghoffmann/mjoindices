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
The API documantation is found on [GitHub Pages](https://cghoffmann.github.io/mjoindices/index.html) and in the docs
folder of the [source](https://github.com/cghoffmann/mjoindices/tree/master/docs).
    
Getting started / examples
--------------------------
After you have installed mjoindices, you can download an
[example](https://github.com/cghoffmann/mjoindices/tree/master/examples) from the source. 

* recalculate_original_omi.py: After downloading some data files, which are mentioned and linked in the source
documentation of the example, you can run this example to recalculate the original OMI values. Furthermore, you can use 
this example as a template to calculate OMI values with your own OLR data. In order to do that, only two parts of the 
code have to be changed, which are also marked in the code documentation.

* evaluate_omi_reproduction.py: After you have run recalculate_original_omi.py (which saves the results), you can check 
the reproduction quality by using this script. It will show detailed comparison plots.

Both files are also available as Jupyter notebook files.
