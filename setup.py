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
    scripts=['mean_temp_image.py', 'read_temperatures.py', 'read_temps_track.py'],
    python_requires=">=3.6",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
    ],
)