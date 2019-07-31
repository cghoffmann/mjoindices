import os

import mjoindex_omi.plotting as plots

originalOMIDataDirname = (os.path.dirname(__file__)
                          + os.path.sep
                          + ".."
                          + os.path.sep
                          + "tests"
                          + os.path.sep
                          + "testdata"
                          + os.path.sep
                          + "OriginalOMI")

plots.plot_eof_for_doy(originalOMIDataDirname,10)