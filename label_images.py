import os
import argparse
import json
import numpy as np
import cv2
from PIL import Image
from thermal_faces.convolution import detect_heads
from thermal_faces.seq_reader import extract_metadata, convert_to_temperature, seq_frames
from thermal_faces.utils import save_image, JsonNumpyEncoder
from tqdm import tqdm

label_names = {
    1: "forehead",
    2: "eye",
    3: "nose"
}

instructions = f"""
number_keys: Choose label
left mouse click: insert label

n: Select the next label
b: Select the previous label
Backspace or d: Delete the current label

Escape or q: Save the result and quit
Enter or s: Save the result and open next frame

+: Increase window size
-: Decrease window size

Label numbers:
{label_names}
"""


def add_label(event, x, y, flags, param):
    global labels
    global l
    global i
    if event == cv2.EVENT_LBUTTONDOWN:
        labels.append({"x": x, "y": y, "l": l})
        i = len(labels)-1

def set_label(_l):
    global l
    l = _l
    if l in label_names:
        print("label", l, ",", label_names[l])
    else:
        print("label", l)


labels = []
i = 0
l = 1
window_width = None
window_height = None
window_name = "Thermal image"
window = cv2.namedWindow(window_name, 0)


def label_image(image, frame_index, filename):
    global labels
    global l
    global i
    global window_name
    global window
    cv2.setMouseCallback(window_name, add_label)

    json_filename = f"train_data/{filename}_{frame_index}.json"
    
    if os.path.exists(json_filename):
        with open(json_filename, "r") as file:
            content = json.load(file)
            labels = content["labels"]
            i = len(labels)-1
    else:
        labels = []
        i = 0
    

    print(instructions)
    
    l = 1
    done = False
    quit = False
    display = None
    while done == False:
        display = image.copy()
        display = cv2.normalize(display, None, 0, 1.0, cv2.NORM_MINMAX)

        for _label in labels:
            y, x, _l = _label["y"], _label["x"], _label["l"]
            cv2.circle(display, (x,y), 0, thickness=3, color=0)
            cv2.putText(display, str(_l), (x+5,y+5), cv2.FONT_HERSHEY_PLAIN, 3, 0, 1, cv2.LINE_AA)
        
        if i < len(labels):
            label = labels[i]
            y, x, _l = label["y"], label["x"], label["l"]
            cv2.circle(display, (x,y), 0, thickness=5, color=0)
            cv2.putText(display, str(_l), (x+5,y+5), cv2.FONT_HERSHEY_PLAIN, 3, 0, 1, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20)
        if key == 13: # Enter
            done = True
        elif key == 8: # Delete
            if len(labels) > 0:
                labels.pop(i)
            if len(labels) > 0:
                i = i%len(labels)
        elif key == 27: # Escape
            done = True
            quit = True
        elif key == ord('s'):
            done = True
        elif key == ord('q'):
            done = True
            quit = True
        elif key == ord('d'):
            if len(labels) > 0:
                labels.pop(i)
            if len(labels) > 0:
                i = i%len(labels)
        elif key == ord('n'):
            if len(labels) > 0:
                i = (i+1)%len(labels)
        elif key == ord('b'):
            if len(labels) > 0:
                i = (i-1)%len(labels)
        elif key == ord('0'):
            set_label(0)
        elif key == ord('1'):
            set_label(1)
        elif key == ord('2'):
            set_label(2)
        elif key == ord('3'):
            set_label(3)
        elif key == ord('4'):
            set_label(4)
        elif key == ord('5'):
            set_label(5)
        elif key == ord('6'):
            set_label(6)
        elif key == ord('7'):
            set_label(7)
        elif key == ord('8'):
            set_label(8)
        elif key == ord('9'):
            set_label(9)
        elif key == ord('+'):
            x, y, w, h = cv2.getWindowImageRect(window_name)
            cv2.resizeWindow(window_name, int(w*1.5), int(h*1.5))
        elif key == ord('-'):
            x, y, w, h = cv2.getWindowImageRect(window_name)
            cv2.resizeWindow(window_name, int(w//1.5), int(h//1.5))

    if not os.path.exists('train_data'):
        os.makedirs('train_data')
    save_image(display, f"train_data/{filename}_{frame_index}.png", scaled=True)
    np.save(f"train_data/{filename}_{frame_index}.npy", image)

    with open(json_filename, "w") as file:
        json.dump(
            {
                "labels": labels,
                "frame": frame_index,
                "filename": filename
            },
            file,
            indent=4,
            cls=JsonNumpyEncoder
        )



def main():
    parser = argparse.ArgumentParser(description="User interface for labeling facial features of many faces in a thermal camera image.")
    parser.add_argument("-f", "--filename", required=True, help="The name of the SEQ video file.")
    parser.add_argument("-s", "--skipframes", default=2000, help="First frame to include.")
    parser.add_argument("-n", "--firstframe", default=500, help="First frame to include.")
    parser.add_argument("-l", "--lastframe", default=None, help="Last frame to include.")
    args = parser.parse_args()

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

    frame_index = 0
    temperature_sum = None
    for frame in seq_frames(args.filename):
        raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width)

        if frame_index > first_frame and frame_index < last_frame:
            if (frame_index - first_frame) % args.skipframes == 0:
                temperature = convert_to_temperature(raw_data, metadata)
                labels = label_image(temperature, frame_index, args.filename)
        
        frame_index += 1


if __name__ == "__main__":
    main()
