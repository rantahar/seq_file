import argparse
import json
import numpy as np
import cv2
from thermal_faces.utils import save_image, JsonNumpyEncoder

# Find heads in the average temperature over the entire video. This should
# find the most stable head locations, and these people are most likely
# watching the performance.

instructions = """
Enter or n: Select the next head location
b: Select the previous head location
Backspace: Delete the current head location
Escape or q: Save the result and quit
w: Move the current head location up
s: Move the current head location down
a: Move the current head location left
d: Move the current head location right
i: Increase the current oval height
k: Decrease the current oval height
l: Increase the current oval width
j: Decrease the current oval width
"""

heads = []

def add_ellipse(event, x, y, flags, param):
    global heads
    if event == cv2.EVENT_LBUTTONDOWN:
        heads.append({"x": x, "y": y, "height": 30, "width": 30})


def main():
    global heads
    parser = argparse.ArgumentParser(description="Display each head location and ask wether to keep or discard.")
    parser.add_argument("-i", "--infile", default="head_locations.json", help="The name of the SEQ video file.")
    parser.add_argument("-o", "--outfile", default="head_locations_checked.json", help="The output json file name.")
    parser.add_argument("-r", "--imagefile", default="mean_checked.png", help="The output image file name.")
    parser.add_argument("-p", "--npy", default="mean.npy", help="The name of the pickled numpy file containing the mean temperature.")
    args = parser.parse_args()

    temperature_mean = np.load(args.npy)

    with open(args.infile, "r") as file:
        heads = json.load(file)

    window_name = "head locations"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, add_ellipse)
    print(instructions)

    done = False
    i = 0
    while done == False:
        head = heads[i]
        display = temperature_mean.copy()
        display = cv2.normalize(display, None, 0, 1.0, cv2.NORM_MINMAX)

        for _index, _head in enumerate(heads):
            y, x = _head["y"], _head["x"]
            h, w = _head["height"]//2, _head["width"]//2
            cv2.ellipse(display, (x,y), (w, h), 0, 0, 360, thickness=1, color=10)
            cv2.putText(display, str(_index), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, 10, 1, cv2.LINE_AA)

        y, x = head["y"], head["x"]
        h, w = head["height"]//2, head["width"]//2
        cv2.ellipse(display, (x,y), (w, h), 0, 0, 360, thickness=4, color=0)
        cv2.putText(display, str(i), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, 0, 4, cv2.LINE_AA)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20)
        if key == 13:
            i = (i+1)%len(heads)
        elif key == ord('n'):
            i = (i+1)%len(heads)
        elif key == 8:
            heads.pop(i)
            i = i%len(heads)
        elif key == ord('w'):
            heads[i]["y"] -= 1
        elif key == ord('s'):
            heads[i]["y"] += 1
        elif key == ord('a'):
            heads[i]["x"] -= 1
        elif key == ord('d'):
            heads[i]["x"] += 1
        elif key == ord('i'):
            heads[i]["height"] += 1
        elif key == ord('k'):
            heads[i]["height"] -= 1
        elif key == ord('j'):
            heads[i]["width"] -= 1
        elif key == ord('l'):
            heads[i]["width"] += 1
        elif key == ord('b'):
            i = (i-1)%len(heads)
        elif key == ord('q'):
            done = True
        elif key == 27:
            done = True
        
    for i, head in enumerate(heads):
        y, x = head["y"], head["x"]
        h, w = head["height"]//2, head["width"]//2
        cv2.ellipse(temperature_mean, (x,y), (w, h), 0, 0, 360, thickness=2, color=255)
        cv2.putText(temperature_mean, str(i), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, 255, 2, cv2.LINE_AA)

        head["subject_id"] = i

    save_image(temperature_mean, args.imagefile, scaled=True)

    with open(args.outfile, "w") as file:
        json.dump(heads, file, indent=4, cls=JsonNumpyEncoder)
    
    print(f"Wrote to {args.outfile}")



if __name__ == "__main__":
    main()

