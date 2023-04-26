import cv2
import numpy as np
from PIL import Image


def save_image(data, filename, scaled=False):
    if scaled:
        data = ((data-10)/40.0)*255
    img = Image.fromarray(data.astype(np.uint8))
    img.save(filename)

def get_rectangle(image, head, margin = 40):
    y, x = head["y"], head["x"]
    h = head["height"]//2
    w = head["width"]//2

    xmin = max([x-w-margin, 0])
    ymin = max([y-h-margin, 0])
    xmax = min([x+w+margin, image.shape[1]-1])
    ymax = min([y+h+margin, image.shape[0]-1])

    rect = image[ymin:ymax, xmin:xmax]
    return rect, (xmin, ymin)

def save_head_image(image, head):
    rect, _ = get_rectangle(image, head)
    h = head["height"]//2
    w = head["width"]//2
    cv2.ellipse(rect, (w+40, h+40), (w, h), 0, 0, 360, thickness=2, color=255)
    save_image(rect, f'subjects/{head["subject_id"]}.png', scaled=True)
    gray = ((image - 10) / 40*255).astype('uint8')
    #cv2.imshow("gray", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()