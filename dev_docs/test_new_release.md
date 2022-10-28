Setup test directory
--------------------
    git clone https://github.com/cghoffmann/mjoindices.git mjoindices_test
    cd mjoindices_test/tests/testdata/
    wget https://zenodo.org/record/3746563/files/omi_reference_data.tar.gz
    tar -xvf omi_reference_data.tar.gz 
    cd ../..

Setup Python environment
------------------------
    python3 -m venv ./venvX.X.X
    source ./venv1.4.0/bin/activate
    pip install mjoindices[full_func,dev]
    pip list

Start unit and integration tests
--------------------------------
    cd tests/
    pytest -v
    cd ..

Start examples
--------------
    cd examples/
    python3 ./recalculate_original_omi.py

