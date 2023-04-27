import os
import json
import numpy as np
import pandas as pd
import cv2
from convolution import detect_heads
from seq_reader import extract_metadata, convert_to_temperature, seq_frames
from utils import save_head_image, save_image, get_rectangle, heads_overlap
from utils import JsonNumpyEncoder
from tqdm import tqdm

# Find heads in the average temperature over the entire video. This should
# find the most stable head locations, and these people are most likely
# watching the performance.


# Video file data
seq_file = 'Rec-000781.seq'
metadata = extract_metadata(seq_file)
width = int(metadata["Raw Thermal Image Width"])
height = int(metadata["Raw Thermal Image Height"])
framerate = int(metadata["Frame Rate"])
bitdepth = 16
frame_size = width * height * (bitdepth // 8)

# If not already done, find the mean temperature.
if not os.path.isfile("mean.npy"):
    frames = 0
    temperature_sum = None
    for frame in tqdm(seq_frames(seq_file)):
        # read temperature data
        raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)
        temperature = convert_to_temperature(raw_data, metadata)
        if temperature_sum is None:
            temperature_sum = temperature
        else:
            temperature_sum += temperature

        frames += 1

    # Get the mean
    temperature_mean = temperature_sum/frames
    np.save("mean.npy", temperature_mean)

else:
    temperature_mean = np.load("mean.npy")


heads = detect_heads(temperature_mean, min_width=30, max_width=50, threshold=2)
for i, head in enumerate(heads):
    y, x = head["y"], head["x"]
    h, w = head["height"]//2, head["width"]//2
    cv2.ellipse(temperature_mean, (x,y), (w, h), 0, 0, 360, thickness=2, color=255)
    cv2.putText(temperature_mean, str(i), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, 255, 2, cv2.LINE_AA)

    head["subject_id"] = i

save_image(temperature_mean, "mean.png", scaled=True)

with open("head_locations.json", "w") as file:
    json.dump(heads, file, indent=4, cls=JsonNumpyEncoder)



