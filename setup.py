#! /usr/bin/env python

import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('fuzzytree', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'fuzzytree'
DESCRIPTION = 'A scikit-learn compatible implementation of fuzzy decision tree estimator.'
with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'J. Balinski'
MAINTAINER_EMAIL = 'balinski.jakub@gmail.com'
URL = 'https://balins.github.io/fuzzytree/index.html'
DOWNLOAD_URL = 'https://pypi.org/project/fuzzytree'
PROJECT_URLS = {
    'Bug Tracker': 'https://github.com/balins/fuzzytree/issues',
    'Documentation': 'https://balins.github.io/fuzzytree/index.html',
    'Source Code': 'https://github.com/balins/fuzzytree'
}
LICENSE = 'new BSD'
VERSION = __version__
INSTALL_REQUIRES = ['numpy', 'scikit-learn']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Development Status :: 3 - Alpha',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9'
               ]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest'
    ],
    'docs': [
        'sphinx',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      project_urls=PROJECT_URLS,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/x-rst',
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      python_requires=">=3.6",
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
