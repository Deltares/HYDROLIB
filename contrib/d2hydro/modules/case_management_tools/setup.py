# -*- coding: utf-8 -*-
from setuptools import setup
import pkg_resources  # part of setuptools


from src.cmt import __version__

#%%
with open('README.md', encoding='utf8') as f:
    long_description = f.read()

setup(
    name='cmt',
    version=__version__,
    description='Case Management Tools for DHYDRO',
    long_description=long_description,
    author='Daniel Tollenaar',
    author_email='daniel@d2hydro.nl',
    license='MIT',
    packages=['cmt'],
    package_dir={'':'src'},
    python_requires='>=3.6',
    install_requires=[
        'pandas',
        'pathlib'
    ]
)
