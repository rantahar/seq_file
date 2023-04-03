import subprocess
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

def seq_frames(filename):
    block_size = 1024*1024 # 1MB

    with open(filename, 'rb') as seq_file:
        # read the file in blocks and split into frames at the magic pattern, "\x46\x46\x46\x00"
        marker = "\x46\x46\x46\x00".encode()
        frame = b''
        index = 1
        for block in iter(lambda: seq_file.read(block_size), ''):
            frame += block
            while True:
                marker_position = frame.find(marker, len(marker))
                if marker_position == -1:
                    break
                if marker_position == 0:
                    frame = frame[len(marker):]
                    continue
                yield frame[:marker_position]
                frame = frame[marker_position + len(marker):]
                index += 1


seq_file = 'Rec-000781.seq'

# Use ExifTools to extract metadata
metadata_str = subprocess.check_output(['exiftool', seq_file]).decode(sys.stdout.encoding)

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

for frame_index, frame in enumerate(seq_frames(seq_file)):

    image = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)
    img = Image.fromarray(image)
    img.save(f'raw_data_{frame_index}.tiff')

    # Convert the raw data to temperature values
    temperature_data = np.zeros_like(image, dtype=np.float32)
    for i in range(height):
        for j in range(width):
            temperature_data[i][j] = pr1 / (pr2 * (image[i][j] - po))
            temperature_data[i][j] -= a1 * at ** 3
            temperature_data[i][j] -= a2 * at ** 2
            temperature_data[i][j] += b1 * od ** 3
            temperature_data[i][j] += b2 * od ** 2
            temperature_data[i][j] += atx * od
            temperature_data[i][j] -= rat
            temperature_data[i][j] /= e
        
    # Create a PIL Image object from the pixel data
    print(temperature_data[100:110,200])
    temperature_data = (temperature_data/40.0*255).astype(np.uint8)
    print(temperature_data[100:110,200])
        
    # Save the image to a TIFF file
    temperature_data.save(f'thermal_image_{frame_index}.tiff')

    #plt.imshow(temperature_data, vmin=temperature_data.min(), vmax=temperature_data.max())
    #plt.show()



