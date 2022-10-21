mjoindices - A python package for calculating the Madden-Julian-Oscillation OMI index
=====================================================================================

[![DOI (paper)](https://img.shields.io/badge/DOI%20%28paper%29-10.5334%2Fjors.331-blue.svg)](https://doi.org/10.5334/jors.331)
[![DOI](https://zenodo.org/badge/197774253.svg)](https://zenodo.org/badge/latestdoi/197774253)

Overview
--------

mjoindices is a Python package relevant for atmospheric scientists. It provides functionality to compute an 
index of the Madden-Julian-Oscillation (MJO), particularly the OLR-based MJO index (OMI).

Whereas the package name has been chosen to indicate that further MJO indices should be included in the future, the 
implementation is currently limited to the OMI algorithm.

A scientific description of the package is found in [Hoffmann et al. (2021)](https://doi.org/10.5334/jors.331).

Citation
--------
If you use mjoindices in published research, please cite the corresponding paper: Hoffmann, C.G., Kiladis, G.N., Gehne, M. and von Savigny, C., 2021.
A Python Package to Calculate the OLR-Based Index of the Madden- Julian-Oscillation (OMI) in Climate Science and Weather Forecasting. 
Journal of Open Research Software, 9(1), p.9. DOI: http://doi.org/10.5334/jors.331

Please check our [list of further scientific publications](https://cghoffmann.github.io/mjoindices/references.html), on which the
implementation of the package is based. It is likely that some of these publications should also be cited.

Contributors
------------
Thanks for the contributions from the community!

[![Contributors](https://contrib.rocks/image?repo=cghoffmann/mjoindices)](https://github.com/cghoffmann/mjoindices/graphs/contributors)


Requirements
------------
mjoindices is written for Python 3 (version >= 3.7) and depends on the packages NumPy, Pandas, SciPy, and Matplotlib. It runs on Linux
and Windows. Other operating systems have not been tested. 

Optional requirements are the packages eofs, xarray, pytest, and pytest-pep8.

The next release of mjoindices will probably require Python 3.8 or greater.

Installation
------------
mjoindices is available in the [Python Package Index (PyPI)](https://pypi.org/project/mjoindices/). It can be installed using, 
e.g., pip.
    
    pip3 install mjoindices
    
It can also be installed from the source, which is available on [Zenodo](http://dx.doi.org/10.5281/zenodo.3613752) and [GitHub](https://github.com/cghoffmann/mjoindices). 
Download the source, move into the directory containing the file setup.py and run

    python3 setup.py install
    
Documentation
-----------------
The documentation is found on [GitHub Pages](https://cghoffmann.github.io/mjoindices/index.html) and also in the docs
folder of the [source](docs/index.html).
    
Getting started / examples
--------------------------
*Note for experienced users: We have slightly changed the API for the EOF calculation with version 1.4. to be more flexible 
for changes in the future. Please read the API docs or compare your code with the current example. The old API is still
working but will deprecate with one of the next releases. Adapting to the new interface will only take a few minutes.*

There are three basic entry points, of which you should read the documentation:

* Calculation of the EOFs: [calc_eofs_from_olr](https://cghoffmann.github.io/mjoindices/api/omi_calculator.html#mjoindices.omi.omi_calculator.calc_eofs_from_olr).
* Calculation of the PCs: [calculate_pcs_from_olr](https://cghoffmann.github.io/mjoindices/api/omi_calculator.html#mjoindices.omi.omi_calculator.calculate_pcs_from_olr).
* An OLR data container class, which has to be provided for the calculations: [OLRData](https://cghoffmann.github.io/mjoindices/api/olr_handling.html#mjoindices.olr_handling.OLRData)

After you have installed mjoindices, you can download an
[example](examples/) from the source, which consists of two files: 

* recalculate_original_omi.py: After downloading some data files, which are mentioned and linked in the source
  documentation of the example, you can run this example to recalculate the original OMI values. The script will save
  the computed Empirical Orthogonal Functions (EOFs) and the Principal Components (PCs) in two individual files, which
  can also be configured in the source code. In addition, it will save a few plots into a directory, which can
  also be configured in the source. These plots show the agreement with the original OMI values (slight deviations are 
  expected due to numerical differences. This is explained in detail in the corresponding software meta paper).

  Note that you can use this example also as a template to calculate OMI values with your own OLR data. 
  In order to do that, you have to adapt only two parts of the code, which are also marked in the code documentation.

  Note also that this script may run for one or two hours on common personal computer systems.

* evaluate_omi_reproduction.py: This script produces more detailed comparison plots and saves them into a directory.
  The script recalculate_original_omi.py has to be run first, since the evaluation script is based on the saved results.
  As for recalculate_original_omi.py, some file and directory names have to be adapted in the beginning of the code.

Both files are also available as Jupyter notebook files.

Automated testing
-----------------
After you have installed mjoindices, you can also download
[unit and integration tests](tests/) from the source to check
your installation using pytest:

* Download the complete test directory to you local file system.

* Download the external input and reference data files from [Zenodo](https://doi.org/10.5281/zenodo.3746562). Details are given in a separate [Readme file](tests/testdata/README). 
  * Details are given in a separate [Readme file](tests/testdata/README).
  * Note that some necessary data files are already included in the test directory in the repository. Make sure to download
    those files together with the tests. The data files on Zenodo are complementary and not 
    included in the repository for reasons of file size and ownership.

* Move into your local test directory and run

      pytest

* In case that some tests are failing with FileNotFoundErrors, it is likely that the package code is actually working, but that the test 
  environment is not setup properly. You should check the following before contacting us:
   * Did you download the data files from the repository?
   * Did you download the data files from Zenodo?
   * Did you preserve the directory structure?
   * Did you execute pytest in the tests/ subdirectory (where pytest.ini is located)? 

* Note that the tests may run for a few hours on a common personal computer.
  * To get a quicker impression, you can omit slow tests by executing the following command. However, this will
    not check the core OMI computation, which is most important, of course.

        pytest -m 'not slow' 
