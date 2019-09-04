# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:04:59 2019

@author: ch
"""

from setuptools import setup
import versioneer

packages=['mjoindex_omi','mjoindex_omi.tests']

setup(name='mjoindex_omi',
      packages=packages,
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      install_requires=['numpy', 'pandas', 'pytest', 'scipy'])
