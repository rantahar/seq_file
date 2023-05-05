
# Installation

pip install git+https://github.com/rantahar/seq_file

Download and install exiftool from https://exiftool.org/

# Scripts

### heads_from_mean_temp

Calculate mean temperature and find head locations from the result. This should find stable head locations.

### check_head_locations

Displays each head location in the mean temperature image one at a time and asks whether to keep that location or delete. Press "k" to keep or "d" to delete. Pressing "b" will go back one step.

### read_temperatures

Processes the entire video, extracting temperature readings based on the previously found head locations.
