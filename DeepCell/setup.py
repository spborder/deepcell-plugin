#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import find_packages

from setuptools import setup


with open('README.rst', 'rt') as readme_file:
    readme = readme_file.read()


def prerelease_local_scheme(version):
    """
    Return local scheme version unless building on master in CircleCI.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on CircleCI for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if os.getenv('CIRCLE_BRANCH') in {'master'}:
        return ''
    else:
        return get_local_node_and_date(version)


setup(
    name='DeepCell_Plugin',
    use_scm_version={'local_scheme': prerelease_local_scheme},
    description='Nuclei and cytoplasm segmentation for CODEX images',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Sam Border',
    author_email='samuel.border@medicine.ufl.edu',
    url='https://github.com/spborder/deepcell-plugin/',
    packages=find_packages(exclude=['tests', '*_test']),
    package_dir={
        'DeepCell_Plugin': 'DeepCell_Plugin',
    },
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'Pillow',
        'scikit-image',
        'girder-slicer-cli-web',
        'girder-client',
        'ctk-cli',
        'wsi-annotations-kit==1.4.4',
        'deepcell'
    ],
    license='Apache Software License 2.0',
    keywords='DeepCell_Plugin',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
)