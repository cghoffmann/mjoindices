# -*- coding: utf-8 -*-

"""
Provides information for the installation routines.
"""

# Copyright (C) 2019 Christoph G. Hoffmann. All rights reserved.

# This file is part of mjoindices

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Contact: christoph.hoffmann@uni-greifswald.de

from setuptools import setup
import versioneer
import os

packages=['mjoindices', 'mjoindices.omi']

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r') as f:
    long_description = f.read()

setup(name='mjoindices',
      packages=packages,
      package_dir={'': 'src'},
#      package_data={'mjoindices': ['./src/mjoindices/omi/sign_reference/eof1/eof001.txt','./src/mjoindices/omi/sign_reference/eof2/eof001.txt'], },
      include_package_data=True,
      zip_safe=False,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      install_requires=['numpy', 'pandas', 'pytest', 'scipy', 'matplotlib'],
      author='Christoph G. Hoffmann',
      author_email="christoph.hoffmann@uni-greifswald.de",
      url="https://github.com/cghoffmann/mjoindices",
      description="Calculation of indices that describe the Madden-Julian-Oscillation (only OMI by now)",
      long_description_content_type="text/markdown",
      long_description=long_description,
      license="GNU General Public License v3")
