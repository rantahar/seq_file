import numpy as np
from seq_reader import extract_metadata, convert_to_temperature, save_image, seq_frames

seq_file = 'Rec-000781.seq'

metadata = extract_metadata(seq_file)

width = int(metadata["Raw Thermal Image Width"])
height = int(metadata["Raw Thermal Image Height"])
bitdepth = 16
frame_size = width * height * (bitdepth // 8)


for frame_index, frame in enumerate(seq_frames(seq_file)):
    raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)
    
    save_image(raw_data, f'raw_data/{frame_index}.tiff')

    temperature = convert_to_temperature(raw_data, metadata)

    save_image(raw_data, f'thermal_image/{frame_index}.tiff')

    



