import numpy as np
import pandas as pd
import cv2
import json
from convolution import detect_heads
from seq_reader import extract_metadata, convert_to_temperature, seq_frames
from utils import save_head_image, save_image, get_rectangle, heads_overlap


# Find faces close to given initial values and track temperatures.


# Video file data
seq_file = 'Rec-000781.seq'
metadata = extract_metadata(seq_file)
width = int(metadata["Raw Thermal Image Width"])
height = int(metadata["Raw Thermal Image Height"])
framerate = int(metadata["Frame Rate"])
bitdepth = 16
frame_size = width * height * (bitdepth // 8)

# Read 
with open("head_locations.json", "r") as file:
    heads = json.load(file)

for frame_index, frame in enumerate(seq_frames(seq_file)):
    # read temperature data
    raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)
    temperature = convert_to_temperature(raw_data, metadata)

    # Reset the matched value for knowns heads
    for head in heads:
        head["matched"] = False

    # Check an area around each head location to find it in this frame.
    # We only extract one head here.
    frame_heads = []
    for head in heads:
        rect, min_corner = get_rectangle(temperature, head, margin = 30)
        h = detect_heads(rect, width=head["width"], height=head["height"], threshold=2.4)
        if len(h) > 0:
            h = h[0]
            h["x"] += min_corner[0]
            h["y"] += min_corner[1]
            h["subject_id"] = head["subject_id"]
            frame_heads += [h]

    print("frame", frame_index, [h["subject_id"] for h in frame_heads])

    df = pd.DataFrame(frame_heads)
    if frame_index == 0:
        df.to_csv("temperatures.csv", index=False)
    else:
        df.to_csv("temperatures.csv", mode="a", index=False)
    

    



