import numpy as np
from seq_reader import extract_metadata, convert_to_temperature, seq_frames
from convolution import detect_heads
#from watershead import detect_heads
#from facenet import detect_heads
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from utils import save_head_image, save_image, get_rectangle

seq_file = 'Rec-000781.seq'

metadata = extract_metadata(seq_file)

width = int(metadata["Raw Thermal Image Width"])
height = int(metadata["Raw Thermal Image Height"])
bitdepth = 16
frame_size = width * height * (bitdepth // 8)


heads = []
for frame_index, frame in enumerate(seq_frames(seq_file)):
    print("frame", frame_index, len(heads))
    raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)
    
    #save_image(raw_data, f'raw_data/{frame_index}.png')

    temperature = convert_to_temperature(raw_data, metadata)

    #save_image(temperature, f'thermal_image/{frame_index}.png', scaled=True)

    frame_heads = detect_heads(temperature)

    for head in heads:
        rect, min_corner = get_rectangle(temperature, head, margin = 30)
        h = detect_heads(rect, width=head["width"], height=head["height"])
        if len(h) > 0:
            h[0]["x"] += min_corner[0]
            h[0]["y"] += min_corner[1]
            frame_heads += [h[0]]
    
    if frame_index%10 == 0:
      for f, frame_head in enumerate(frame_heads):
        matched = False
        for i, head in enumerate(heads):
            dx = head["x"] - frame_head["x"]
            dy = head["y"] - frame_head["y"]
            a = head["width"]/2
            b = head["height"]/2
            if (dx**2/a**2 + dy**2/b**2) < 0.2:
                matched = True
                for key, val in frame_head.items():
                    heads[i][key] = frame_head[key]
                break
        if not matched:
            print("new subject", f, len(frame_heads))
            heads.append(frame_head)
            
    for i, head in enumerate(heads):
        heads[i]["frame"] = frame_index
        if "subject_id" not in head:
            heads[i]["subject_id"] = i
            save_head_image(temperature, head)
            

    df = pd.DataFrame(heads)
    if frame_index == 0:
        df.to_csv("temperatures.csv", index=False)
    else:
        df.to_csv("temperatures.csv", mode="a", index=False, header=False)
    

    



