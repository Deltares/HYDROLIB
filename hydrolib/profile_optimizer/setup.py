from codecs import open
from os import path

import profile_optimizer
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#     long_description = f.read()

setup(
    name=profile_optimizer.__title__,
    version=profile_optimizer.__version__,
    description=profile_optimizer.__description__,
    # long_description=long_description,
    # The project's main homepage.
    url=profile_optimizer.__url__,
    # Author details
    author=profile_optimizer.__author__,
    author_email=profile_optimizer.__author_email__,
    # Choose your license
    license=profile_optimizer.__license__,
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Topic :: RHDHV :: Water Management",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
    ],
    keywords=profile_optimizer.__keywords__,
    packages=find_packages(include=["profile_optimizer", "figures", "src"]),
    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # install_requires=[
    # ],
    # You can install these using the following syntax, for example:
    # $ pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['tests'],
    # },
    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #     'sobek2': ['sobek2/ls_language_dictionary.csv', ],
    # },
    # include_package_data=True,
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'xsb=xsboringen.scripts.xsb:main',
    #     ],
    # },
)
