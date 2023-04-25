import subprocess
import sys
import numpy as np
from PIL import Image


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


def extract_metadata(seq_file):
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

    return metadata

def extract_temperature_images(seq_file, metadata, bitdepth = 16):
    width = int(metadata["Raw Thermal Image Width"])
    height = int(metadata["Raw Thermal Image Height"])

    frame_size = width * height * (bitdepth // 8)

    for frame_index, frame in enumerate(seq_frames(seq_file)):
        raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)
        img = Image.fromarray(raw_data)
        img.save(f'raw_data/{frame_index}.tiff')

        temperature = convert_to_temperature(raw_data)



def convert_to_temperature(frame, metadata):
    planck_R1 = float(metadata['Planck R1'])
    planck_R2 = float(metadata['Planck R2'])
    Planck_B = float(metadata['Planck B'])
    planck_O = float(metadata['Planck O'])
    planck_F = float(metadata['Planck F'])
    Atmos_A1 = float(metadata['Atmospheric Trans Alpha 1'])
    Atmos_A2 = float(metadata['Atmospheric Trans Alpha 2'])
    Atmos_B1 = float(metadata['Atmospheric Trans Beta 1'])
    Atmos_B2 = float(metadata['Atmospheric Trans Beta 2'])
    Atmos_X = float(metadata['Atmospheric Trans X'])
    emissivity = float(metadata['Emissivity'])

    # Temperature readings, should be in C
    Atmos_T, unit = metadata['Atmospheric Temperature'].split(' ')
    assert unit == 'C'
    Atmos_T = float(Atmos_T)
    RA_TEMP, unit = metadata['Reflected Apparent Temperature'].split(' ')
    assert unit == 'C'
    RA_TEMP = float(RA_TEMP)
    
    # Distance in m
    Object_dist, unit = metadata['Object Distance'].split(' ')
    assert unit == 'm'
    Object_dist = 2 #float(Object_dist)

    # Convert the raw data to temperature values
    #temperature_data = planck_R1 / (planck_R2 * (frame - planck_O))
    #temperature_data -= Atmos_A1 * Atmos_T ** 3
    #temperature_data -= Atmos_A2 * Atmos_T ** 2
    #temperature_data += Atmos_B1 * Object_dist ** 3
    #temperature_data += Atmos_B2 * Object_dist ** 2
    #temperature_data += Atmos_X * Object_dist
    #temperature_data -= RA_TEMP
    #temperature_data /= emissivity
    temperature_data = Planck_B/np.log(planck_R1/(planck_R2*(frame+planck_O))+planck_F)-273.15
    return temperature_data


def save_image(data, filename, scaled=False):
    if scaled:
        data = ((data-10)/40.0)*255
    img = Image.fromarray(data.astype(np.uint8))
    img.save(filename)

