"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyhail",
    version="2.4",
    description="Python Hail Weather Radar Retrievals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joshuass/pyhail",
    author="Joshua Soderholm",
    author_email="joshua.soderholm@bom.gov.au",  # Optional
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires='>=3.7',
    package_dir = {'': 'src'},
    packages=['pyhail'],
    project_urls={"Source": "https://github.com/joshuass/pyhail",},
)
