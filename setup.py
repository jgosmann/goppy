#!/usr/bin/env python

from distutils.core import setup


with open('README.rst') as f:
    long_description = f.read()

setup(
    name='GopPy',
    version='0.9',
    description='GopPy (Gaussian Online Processes for Python) is a pure '
    'Python module providing a Gaussian process implementation which allows '
    'to efficiently add new data online.',
    long_description=long_description,
    author='Jan Gosmann',
    author_email='jan@hyper-world.de',
    url='http://jgosmann.github.io/goppy/',
    packages=['goppy', 'goppy.test'],
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
    ]
)
