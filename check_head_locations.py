import os
import argparse
import json
import numpy as np
import cv2
from thermal_faces.utils import save_image, JsonNumpyEncoder

# Find heads in the average temperature over the entire video. This should
# find the most stable head locations, and these people are most likely
# watching the performance.

def main():
    parser = argparse.ArgumentParser(description="Display each head location and ask wether to keep or discard.")
    parser.add_argument("-i", "--infile", default="head_locations.json", help="The name of the SEQ video file.")
    parser.add_argument("-o", "--outfile", default="head_locations_checked.json", help="The output json file name.")
    parser.add_argument("-r", "--imagefile", default="mean_checked.png", help="The output image file name.")
    parser.add_argument("-p", "--npy", default="mean.npy", help="The name of the pickled numpy file containing the mean temperature.")
    args = parser.parse_args()

    temperature_mean = np.load(args.npy)

    with open(args.infile, "r") as file:
        heads = json.load(file)

    i = 0
    while i < len(heads):
        head = heads[i]
        display = temperature_mean.copy()
        display = cv2.normalize(display, None, 0, 1.0, cv2.NORM_MINMAX)


        y, x = head["y"], head["x"]
        h, w = head["height"]//2, head["width"]//2
        cv2.ellipse(display, (x,y), (w, h), 0, 0, 360, thickness=2, color=0)
        cv2.putText(display, str(i), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, 0, 2, cv2.LINE_AA)

        key = None
        cv2.imshow("Press K to keep, D to delete, B to go back", display)
        while True:
            key = cv2.waitKey(0)
            if key == ord('k'):
                heads[i]["keep"] = True
                i += 1
                break
            elif key == ord('d'):
                heads[i]["keep"] = False
                i+=1
                break
            elif key == ord('b'):
                if i > 0:
                    i -= 1
                break

    heads = [h for h in heads if h["keep"]]

    for i, head in enumerate(heads):
        y, x = head["y"], head["x"]
        h, w = head["height"]//2, head["width"]//2
        cv2.ellipse(temperature_mean, (x,y), (w, h), 0, 0, 360, thickness=2, color=255)
        cv2.putText(temperature_mean, str(i), (x+w,y+h), cv2.FONT_HERSHEY_PLAIN, 3, 255, 2, cv2.LINE_AA)

        head["subject_id"] = i

    save_image(temperature_mean, args.imagefile, scaled=True)

    with open(args.outfile, "w") as file:
        json.dump(heads, file, indent=4, cls=JsonNumpyEncoder)



if __name__ == "__main__":
    main()

