#!/usr/bin/env python
from setuptools import setup, find_packages
from os.path import join, dirname

with open("README.md", "r") as fh:
    long_description = fh.read()

requirementstxt = join(dirname(__file__), "requirements.txt")
with open(requirementstxt, "r") as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='thermal_faces',
    version=0.1,
    description='Read frames from a SEQ file, extract faces and calculate temperature time series.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jarno Rantaharju',
    author_email='jarno.rantaharju@aalto.fi',
    url='https://github.com/rantahar/seq_file',
    packages=find_packages(where='.'),
    py_modules=["mean_temp_image", "read_temperatures"],
    entry_points={
       'console_scripts': [
            'heads_from_mean_temp=mean_temp_image:main',
            'read_temperatures=read_temperatures:main',
        ],
    },
    python_requires=">=3.6",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
    ],
)