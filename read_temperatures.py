import numpy as np
import pandas as pd
import cv2
from convolution import detect_heads
from seq_reader import extract_metadata, convert_to_temperature, seq_frames
from utils import save_head_image, save_image, get_rectangle, heads_overlap


# This version runs through the entire video and tries to automatically
# identify and track subjects.

# Subject identification parameters
subject_still_for = 60 # Save subject after this many seconds.
min_match_fraction = 0.8 # Save subject of found at least this often.
fast_drop_fraction = 0.2 # Drop subject if detection rate falls below.
find_all_heads_every = 2 # Check for new heads this often.


# Video file data
seq_file = 'Rec-000781.seq'
metadata = extract_metadata(seq_file)
width = int(metadata["Raw Thermal Image Width"])
height = int(metadata["Raw Thermal Image Height"])
framerate = int(metadata["Frame Rate"])
bitdepth = 16
frame_size = width * height * (bitdepth // 8)

# adjust from seconds to frames
subject_still_for = subject_still_for * framerate
find_all_heads_every = find_all_heads_every * framerate
print(find_all_heads_every)

next_subject_id = 1
heads = []
for frame_index, frame in enumerate(seq_frames(seq_file)):
    print("frame", frame_index, len(heads))

    # read temperature data
    raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)
    temperature = convert_to_temperature(raw_data, metadata)

    # Check an area around each head location to find it in this frame.
    # We only extract one head here.
    frame_heads = []
    for head in heads:
        rect, min_corner = get_rectangle(temperature, head, margin = 30)
        h = detect_heads(rect, width=head["width"], height=head["height"], threshold=2.5)
        if len(h) > 0:
            h[0]["x"] += min_corner[0]
            h[0]["y"] += min_corner[1]
            frame_heads += [h[0]]
        #else:
        #    print(f"subject {head['subject_id']} not found")

    # Detect all heads occationally
    if frame_index%find_all_heads_every == 0:
        frame_heads += detect_heads(temperature)
        i = len(heads)
        while i < len(frame_heads):
            removed = False
            for head in frame_heads[:i]:
                if heads_overlap(frame_heads[i], head):
                    frame_heads.pop(i)
                    removed = True
                    break
            if not removed:
                i += 1

    # Check each found head and match to existing
    for f, frame_head in enumerate(frame_heads):
        matched = False
        for i, head in enumerate(heads):
            dx = head["x"] - frame_head["x"]
            dy = head["y"] - frame_head["y"]
            a = head["width"]/2
            b = head["height"]/2
            if (dx**2/a**2 + dy**2/b**2) < 0.5:
                matched = True
                for key, val in frame_head.items():
                    heads[i][key] = frame_head[key]
                heads[i]["hits"] += 1
                frame_heads[f]["matched"] = True
                break

    for f, frame_head in enumerate(frame_heads):
        if "matched" not in frame_head:
            heads.append(frame_head)
            
    for i, head in enumerate(heads):
        if "subject_id" not in head:
            heads[i]["subject_id"] = next_subject_id
            heads[i]["age"] = 0
            heads[i]["hits"] = 1
            next_subject_id += 1
        if heads[i]["age"] < subject_still_for:
            if heads[i]["hits"] < heads[i]["age"]*fast_drop_fraction:
                heads[i]["remove"] = True
                print("fast remove", i, heads[i]["hits"]/heads[i]["age"])
        if heads[i]["age"] == subject_still_for:
            if heads[i]["hits"] < heads[i]["age"]*min_match_fraction:
                heads[i]["remove"] = True
                print("remove", i, heads[i]["hits"]/heads[i]["age"])
            else:
                save_head_image(temperature, head)
        heads[i]["frame"] = frame_index
        heads[i]["age"] += 1

    # Remove heads with too few hits
    heads = [h for h in heads if "remove" not in h]

    #if frame_index % 1000 == 0:
    #    for i, head in enumerate(heads):
    #        save_head_image(temperature, head, index=frame_index)

            

    df = pd.DataFrame(heads)
    if frame_index == 0:
        df.to_csv("temperatures.csv", index=False)
    else:
        df.to_csv("temperatures.csv", mode="a", index=False)
    

    



