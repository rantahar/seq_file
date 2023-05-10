
# Installation

pip install git+https://github.com/rantahar/seq_file

Download and install exiftool from https://exiftool.org/

# Scripts

### heads_from_mean_temp

Calculate mean temperature and find head locations from the result. This should find stable head locations.

The script also saves the temperature image every 1000 frames. Use the `--firstframe` and `--lastframe` options to include a given time period.

### check_head_locations

Check and edit head locations graphically.

Instructions: 
Enter or n: Select the next head location
b: Select the previous head location
Backspace: Delete the current head location
Escape or q: Save the result and quit
w: Move the current head location up
s: Move the current head location down
a: Move the current head location left
d: Move the current head location right
i: Increase the current oval height
k: Decrease the current oval height
l: Increase the current oval width
j: Decrease the current oval width
Left mouse click: Add a new head location

### read_temperatures

Processes the entire video, extracting temperature readings based on the previously found head locations.

For each frame and each head location, the script looks for an oval of that size and close to the given location. If one is found,
it is recorded to a csv file.
