import numpy as np
import pandas as pd
import json
import argparse
from thermal_faces.convolution import detect_heads
from thermal_faces.utils import get_rectangle
from thermal_faces.seq_reader import extract_metadata, convert_to_temperature, seq_frames

# Find faces close to given initial values and track temperatures.
def main():
    parser = argparse.ArgumentParser(description="Calculate the average temperature in a SEQ-video file and find faces in the average temperature. These are recorded in a json file for later use.")
    parser.add_argument("-f", "--filename", required=True, help="The name of the SEQ video file.")
    parser.add_argument("-i", "--input_heads", default="head_locations.json", help="The output json file name.")
    parser.add_argument("-o", "--outfile", default="temperatures.csv", help="The output csv file name.")
    parser.add_argument("--threshhold", default=2.4, help="Threshhold for detecting a head. Lower threshold means more heads and more false positives.")
    parser.add_argument("--keep", default=[], nargs='*', help="A list of subject ids to keep. By default keep all.")
    args = parser.parse_args()

    # Video file data
    metadata = extract_metadata(args.filename)
    width = int(metadata["Raw Thermal Image Width"])
    height = int(metadata["Raw Thermal Image Height"])
    bitdepth = 16
    frame_size = width * height * (bitdepth // 8)

    # Read head locations
    with open(args.input_heads, "r") as file:
        heads = json.load(file)

    for frame_index, frame in enumerate(seq_frames(args.filename)):
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
            h = detect_heads(rect, width=head["width"], height=head["height"], threshold=float(args.threshhold))
            if len(h) > 0:
                h = h[0]
                h["x"] += min_corner[0]
                h["y"] += min_corner[1]
                h["subject_id"] = head["subject_id"]
                frame_heads += [h]

        if len(frame_heads) < 10:
            print("frame", frame_index, [h["subject_id"] for h in frame_heads])
        else:
            print(f"frame {frame_index}, {len(frame_heads)} subjects found")

        df = pd.DataFrame(frame_heads)
        if frame_index == 0:
            df.to_csv(args.outfile, index=False)
        else:
            df.to_csv(args.outfile, mode="a", index=False, header=False)



if __name__ == "__main__":
    main()


