import numpy as np
from seq_reader import extract_metadata, convert_to_temperature, save_image, seq_frames
#from watershead import detect_heads
#from haar_cascade import detect_heads
from facenet import detect_heads
import matplotlib.pyplot as plt

seq_file = 'Rec-000781.seq'

metadata = extract_metadata(seq_file)

width = int(metadata["Raw Thermal Image Width"])
height = int(metadata["Raw Thermal Image Height"])
bitdepth = 16
frame_size = width * height * (bitdepth // 8)


for frame_index, frame in enumerate(seq_frames(seq_file)):
    raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)
    
    #save_image(raw_data, f'raw_data/{frame_index}.png')

    temperature = convert_to_temperature(raw_data, metadata)

    save_image(temperature, f'thermal_image/{frame_index}.png', scaled=True)

    heads = detect_heads(temperature)
    print(heads)
    

    



