#!/usr/bin/env python
import numpy as np
import pandas as pd
from thermal_faces.convolution import detect_heads
from thermal_faces.utils import save_head_image, get_rectangle, heads_overlap
from thermal_faces.seq_reader import extract_metadata, convert_to_temperature, seq_frames


# This version runs through the entire video and tries to automatically
# identify and track subjects.

# Subject identification parameters
subject_follow_time = 40 # Follow for this many seconds before saving image
subject_average_hit_time = 20 # Window for rolling hit rate to determine dropping
find_all_heads_every = 5 # Check for new heads this often.


# Video file data
seq_file = 'Rec-000781.seq'
metadata = extract_metadata(seq_file)
width = int(metadata["Raw Thermal Image Width"])
height = int(metadata["Raw Thermal Image Height"])
framerate = int(metadata["Frame Rate"])
bitdepth = 16
frame_size = width * height * (bitdepth // 8)

# adjust from seconds to frames
subject_follow_time = subject_follow_time * framerate
subject_average_hit_time = subject_average_hit_time * framerate
find_all_heads_every = find_all_heads_every * framerate
subject_average_rate = 1.0/subject_average_hit_time
print(subject_average_rate)

next_subject_id = 1
heads = []
for frame_index, frame in enumerate(seq_frames(seq_file)):
    print("frame", frame_index, len(heads))

    # read temperature data
    raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)
    temperature = convert_to_temperature(raw_data, metadata)

    # Reset the matched value for knowns heads
    for head in heads:
        head["matched"] = False
        head["hit_rate"] *= (1-subject_average_rate)

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
        frame_heads[f]["matched"] = False
        for i, head in enumerate(heads):
            dx = head["x"] - frame_head["x"]
            dy = head["y"] - frame_head["y"]
            a = head["width"]/2
            b = head["height"]/2
            if (dx**2/a**2 + dy**2/b**2) < 0.5:
                for key, val in frame_head.items():
                    heads[i][key] = frame_head[key]
                heads[i]["hit_rate"] += subject_average_rate
                frame_heads[f]["matched"] = True
                break

    for frame_head in frame_heads:
        if not frame_head["matched"]:
            heads.append(frame_head)
            
    for i, head in enumerate(heads):
        if "subject_id" not in head:
            heads[i]["subject_id"] = next_subject_id
            heads[i]["age"] = 0
            heads[i]["hit_rate"] = 1
            next_subject_id += 1
        if heads[i]["hit_rate"] < 0.5:
            heads[i]["remove"] = True
            print("remove", i, heads[i]["hit_rate"])
        if heads[i]["age"] == subject_follow_time:
            save_head_image(temperature, head)
            print(heads[i]["subject_id"], heads[i]["matched"], heads[i]["hit_rate"])
        heads[i]["frame"] = frame_index
        heads[i]["age"] += 1

    # Remove heads with too few hits
    heads = [h for h in heads if "remove" not in h]

    #if frame_index % 1000 == 0:
    #    for i, head in enumerate(heads):
    #        save_head_image(temperature, head, index=frame_index)

    df = pd.DataFrame([h for h in heads if h["matched"]])
    if frame_index == 0:
        df.to_csv("temperatures.csv", index=False)
    else:
        df.to_csv("temperatures.csv", mode="a", index=False)
    

    



