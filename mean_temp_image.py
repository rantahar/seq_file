import os
import argparse
import json
import numpy as np
import cv2
from thermal_faces.convolution import detect_heads
from thermal_faces.seq_reader import extract_metadata, convert_to_temperature, seq_frames
from thermal_faces.utils import save_image, JsonNumpyEncoder
from tqdm import tqdm

# Find heads in the average temperature over the entire video. This should
# find the most stable head locations, and these people are most likely
# watching the performance.

def main():
    parser = argparse.ArgumentParser(description="Calculate the average temperature in a SEQ-video file and find faces in the average temperature. These are recorded in a json file for later use.")
    parser.add_argument("-f", "--filename", required=True, help="The name of the SEQ video file.")
    parser.add_argument("-o", "--outfile", default="head_locations.json", help="The output json file name.")
    parser.add_argument("-i", "--imagefile", default="mean.png", help="The output image file name.")
    parser.add_argument("-p", "--npy", default="mean.npy", help="The name of the pickled numpy file containing the mean temperature.")
    parser.add_argument("--firstframe", default=0, help="First frame to include.")
    parser.add_argument("--lastframe", default=None, help="Last frame to include.")
    parser.add_argument("--minwidth", default=30, help="Minimum head width.")
    parser.add_argument("--maxwidth", default=50, help="Maximum head width.")
    parser.add_argument("--threshhold", default=2, help="Threshhold for detecting a head. Lower threshold means more heads and more false positives.")
    parser.add_argument("--keep", default=[], nargs='*', help="A list of subject ids to keep. By default keep all.")
    args = parser.parse_args()

    keep = [int(k) for k in args.keep]
    first_frame = int(args.firstframe)
    if args.lastframe is None:
        last_frame = float("inf")
    else:
        last_frame = int(args.lastframe)

    # Video file data
    metadata = extract_metadata(args.filename)
    width = int(metadata["Raw Thermal Image Width"])
    height = int(metadata["Raw Thermal Image Height"])
    bitdepth = 16
    frame_size = width * height * (bitdepth // 8)

    # If not already done, find the mean temperature.
    if not os.path.isfile(args.npy):
        frame_index = 0
        temperature_sum = None
        for frame in tqdm(seq_frames(args.filename)):
            # read temperature data
            raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)

            if frame_index > first_frame and frame_index < last_frame:
                temperature = convert_to_temperature(raw_data, metadata)
                if temperature_sum is None:
                    temperature_sum = temperature
                else:
                    temperature_sum += temperature

            if frame_index % 1000 == 0:
                temperature = convert_to_temperature(raw_data, metadata)
                if not os.path.exists('temperature_frames'):
                    os.makedirs('temperature_frames')
                save_image(temperature, f"temperature_frames/{frame_index}.png", scaled=True)

            frame_index += 1

        if args.lastframe is None:
            last_frame = frame_index
        
        frames = last_frame - first_frame

        # Get the mean
        temperature_mean = temperature_sum/frames
        np.save(args.npy, temperature_mean)

    else:
        temperature_mean = np.load(args.npy)


    heads = detect_heads(
        temperature_mean,
        min_width=int(args.minwidth),
        max_width=int(args.maxwidth),
        threshold=float(args.threshhold)
    )

    if keep:
        heads = [h for i, h in enumerate(heads) if i in keep]

    for i, head in enumerate(heads):
        y, x = head["y"], head["x"]
        h, w = head["height"]//2, head["width"]//2
        y_max, x_max = head["max temp y"], head["max temp x"]
        cv2.ellipse(temperature_mean, (x,y), (w, h), 0, 0, 360, thickness=2, color=255)
        cv2.circle(temperature_mean, (x_max, y_max), 0, thickness=5, color=255)
        cv2.putText(temperature_mean, str(i), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, 255, 2, cv2.LINE_AA)

        head["subject_id"] = i

    save_image(temperature_mean, args.imagefile, scaled=True)

    with open(args.outfile, "w") as file:
        json.dump(heads, file, indent=4, cls=JsonNumpyEncoder)



if __name__ == "__main__":
    main()

