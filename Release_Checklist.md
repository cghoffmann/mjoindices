#Jupyter Notebooks
- clear cell outputs 

# Testing
- pytest in dir tests/

#Documentation
- activate venv with Sphinx
- change to dir docsrc/
- call make html
- call make github
- check changes in dir docs/
- commit dir docs/

# Git
- git tag mjoindices-X.X.X
- git push origin mjoindices-X.X.X

# Setup Tools
- activate venv
- python3 setup.py sdist bdist_wheel

# GitHub
- Create Release from new tag
- Attach package files to release

# PyPi
- python3 -m twine upload dist/*
- Username: __token__

# Zenodo
- Wait for Zenodo to find new GitHub release
- Change Metainformation: Link to Journal, ORCID, Title...

