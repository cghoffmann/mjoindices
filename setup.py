# -*- coding: utf-8 -*-

""" """

# Copyright (C) 2019 Christoph G. Hoffmann. All rights reserved.

# This file is part of mjoindex_omi

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

packages=['mjoindex_omi','mjoindex_omi.tests']

setup(name='mjoindex_omi',
      packages=packages,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      install_requires=['numpy', 'pandas', 'pytest', 'scipy', 'matplotlib'])
