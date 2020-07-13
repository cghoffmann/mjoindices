FROM continuumio/miniconda3

COPY . .

RUN conda install notebook

RUN python ./setup.py install

WORKDIR tests/testdata

RUN wget -q https://zenodo.org/record/3746563/files/omi_reference_data.tar.gz

RUN tar -xf omi_reference_data.tar.gz

WORKDIR ..

RUN pytest test_empirical_orthogonal_functions.py test_evaluation_tools.py test_olr_handling.py test_principal_components.py test_tools.py

WORKDIR ..

EXPOSE 8890

CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8890", "examples/recalculate_original_omi.ipynb"]
