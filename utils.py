import cv2
import numpy as np
from PIL import Image
import json


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

def save_head_image(image, head, index=None):
    rect, min_coord = get_rectangle(image, head, margin=80)
    y, x = head["y"], head["x"]
    h = head["height"]//2
    w = head["width"]//2
    cv2.ellipse(rect, (x-min_coord[0], y-min_coord[1]), (w, h), 0, 0, 360, thickness=2, color=255)
    filename = f'{head["subject_id"]}.png'
    if index is not None:
        filename = f"frame_{index}_" + filename
    save_image(rect, "subjects/"+filename, scaled=True)
    gray = ((image - 10) / 40*255).astype('uint8')
    #cv2.imshow("gray", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def heads_overlap(head1, head2):
    ellipses_overlap(
        (head1["y"], head1["x"], 0, head1["width"], head1["height"]),
        (head2["y"], head2["x"], 0, head2["width"], head2["height"])
    )

def ellipses_overlap(ellipse1, ellipse2):
    y1, x1, _, width1, height1 = ellipse1
    y2, x2, _, width2, height2 = ellipse2

    angle = np.linspace(0, 2 * np.pi, 100)
    y1s = y1 + height1/2 * np.cos(angle)
    x1s = x1 + width1/2 * np.sin(angle)
    y2s = y2 + height2/2 * np.cos(angle)
    x2s = x2 + width2/2 * np.sin(angle)

    a, b = width2/2, height2/2
    dist1 = np.sqrt((y1s - y2)**2/b**2 + ((x1s - x2)**2)/a**2)
    a, b = width1/2, height1/2
    dist2 = np.sqrt((y2s - y1)**2/b**2 + ((x2s - x1)**2)/a**2)

    return np.any(dist1 <= 1.0) or np.any(dist2 <= 1.0)


class JsonNumpyEncoder(json.JSONEncoder):
    ''' Extend the JSON encoder to accept numpy number types '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(MyEncoder, self).default(obj)