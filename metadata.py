import subprocess
import sys
import numpy as np
import struct
import tifffile


seq_file = 'Rec-000781.seq'

# Use ExifTools to extract metadata
metadata_str = subprocess.check_output(['exiftool', seq_file]).decode(sys.stdout.encoding)
print(metadata_str)

# create a dictionary to store the metadata values
metadata = {}

# split the metadata string into lines and iterate over them
for line in metadata_str.split("\n"):
    # skip any empty lines or lines that don't contain a colon separator
    if not line or ":" not in line:
        continue
    
    # split the line into a key-value pair using the colon separator
    key, value = line.split(":", 1)
    
    # strip any whitespace from the key and value
    key = key.strip()
    value = value.strip()
    
    # store the key-value pair in the metadata dictionary
    metadata[key] = value

print(metadata)


width = int(metadata["Raw Thermal Image Width"])
height = int(metadata["Raw Thermal Image Height"])

with open(seq_file, 'rb') as f:
    f.seek(512)  # skip the SEQ header
    raw_data = f.read(4096)

# Convert the raw image data to a NumPy array
pixel_data = tifffile.memmap(raw_data, shape=(height, width), dtype=np.uint16)


# Open the SEQ file for reading in binary mode
with open('Rec-000781.seq', 'rb') as f:
    # Read the SEQ file header
    
    bitdepth = 16
    pr1 = float(metadata['Planck R1'])
    pr2 = float(metadata['Planck R2'])
    po = float(metadata['Planck O'])
    a1 = float(metadata['Atmospheric Trans Alpha 1'])
    a2 = float(metadata['Atmospheric Trans Alpha 2'])
    b1 = float( metadata['Atmospheric Trans Beta 1'])
    b2 = float( metadata['Atmospheric Trans Beta 2'])
    atx = float( metadata['Atmospheric Trans X'])
    e = float( metadata['Emissivity'])

    # Temperature readings, should be in C
    at, unit = metadata['Atmospheric Temperature'].split(' ')
    assert unit == 'C'
    at = float(at)
    rat, unit = metadata['Reflected Apparent Temperature'].split(' ')
    assert unit == 'C'
    rat = float(rat)

    # Distance in m
    od, unit = metadata['Object Distance'].split(' ')
    assert unit == 'm'
    od = float(od)

    # Calculate the size of each frame in bytes
    frame_size = width * height * (bitdepth // 8)
    
    # Seek to the start of the first frame
    f.seek(512)
    
    # Read the data for the first frame
    frame_data = f.read(frame_size)
    
    # Convert the raw data to temperature values
    temperature_data = []
    for i in range(height):
        row = []
        for j in range(width):
            pixel_data = frame_data[i * width + j]
            temperature = (pixel_data - pr1) / (pr2 * (pixel_data - po))
            temperature -= a1 * at ** 3
            temperature -= a2 * at ** 2
            temperature += b1 * od ** 3
            temperature += b2 * od ** 2
            temperature += atx * od
            temperature -= rat
            temperature /= e
            print(pixel_data)
            row.append(temperature)
        temperature_data.append(row)
        



