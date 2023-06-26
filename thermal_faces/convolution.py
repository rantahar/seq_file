import cv2
import numpy as np
from numpy import ma
from .utils import get_rectangle, ellipses_overlap


def make_kernel(width, height):
    """
    Creates a kernel of size (2*height, 2*width) with a center at (height, width).
    The kernel has values of 1 within an ellipse with the given width and height
    on the boundary. The values are normalized by the total number of elements in the
    kernel.

    Args:
    - width (int): The width of the center of the kernel.
    - height (int): The height of the center of the kernel.

    Returns:
    - kernel (numpy.ndarray): The kernel.
    """
    kernel_center = (height, width)
    a, b = width // 2, height // 2

    kernel = np.zeros((2*height, 2*width))
    y_indices, x_indices = np.mgrid[0:2*height, 0:2*width]
    x_ind = x_indices - kernel_center[1]
    y_ind = y_indices - kernel_center[0]
    distances = np.sqrt((y_ind)**2/b**2 + (x_ind)**2/a**2)

    kernel[distances < 1.3] = -1
    kernel[distances < 1.0] = 1
    kernel[(np.abs(x_ind) < 1.5*width) & (y_ind < -0.7*height)] = 0.5
    kernel = kernel/(width*height)
    return kernel


def filter_faces(faces):
    """ Filter out overlapping ellipses in a list of faces. Keep large ellipses first, and
    better fits second.

    Args:
    - faces (list): A list of tuples representing faces. Each tuple contains five values:
       y-coordinate of the center, x-coordinate of the center, confidence in the detection, height
       and width.

    Returns:
    - faces (list): A list of tuples representing faces with overlapping ellipses filtered out.
    """
    i = 0
    while i < len(faces):
        j = i+1
        while j < len(faces):
            if ellipses_overlap(faces[i], faces[j]):
                if faces[i][3] > faces[j][3]:
                    faces.pop(j)
                elif faces[i][3] < faces[j][3]:
                    faces.pop(i)
                    i -= 1
                    break
                elif faces[i][2] > faces[j][2]:
                    faces.pop(j)
                else:
                    faces.pop(i)
                    i = i-1
                    break
            else:
                j += 1
        i += 1
    return faces


def display_faces(img, faces):
    gray = ((img - 10) / 40*255).astype('uint8')
    for face in faces:
        y, x = face["y"], face["x"]
        h, w = face["height"]//2, face["width"]//2
        y_max, x_max = face["max temp y"], face["max temp x"]
        cv2.ellipse(gray, (x, y), (w//2, h//2), 0, 0, 360, color=(0, 255, 0), thickness=2)
        cv2.circle(gray, (x_max, y_max), 0, thickness=5, color=255)

    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def masked_rect(rect, center_y, center_x, height, width):
    a, b = width // 2, height // 2

    mask = np.zeros(rect.shape)
    y_indices, x_indices = np.mgrid[0:rect.shape[0], 0:rect.shape[1]]
    x_ind = x_indices - center_x
    y_ind = y_indices - center_y
    distances = np.sqrt((y_ind)**2/b**2 + (x_ind)**2/a**2)

    mask[distances > 1.0] = 1
    return ma.array(rect, mask=mask)

def detect_heads(img, width=None, height=None, min_width=30, max_width=60, width_step=1.1, height_ratios=[1.2, 1.4], clip_range=[25, 37], threshold=2.5):
    # Scale the temperature values to 0-255 range

    if width is None:
        widths = []
        width = min_width
        while width <= max_width:
            widths.append(int(width))
            width *= width_step
    else:
        widths = [width]

    faces = []
    for w in widths:
        if height is None:
            heights = [int(ratio*w) for ratio in height_ratios]
        else:
            heights = [height]

        for h in heights:
            kernel = make_kernel(w, h)
            
            img_clipped = img.copy() - clip_range[0]
            img_clipped[img<clip_range[0]] = 0
            img_clipped[img>clip_range[1]] = 0
            z = cv2.filter2D(img_clipped, -1, kernel)

            for i in range(100):
                y, x = np.unravel_index(np.argmax(z), z.shape)
                value = z[y, x]
                if value < threshold:
                    break

                cv2.ellipse(z, (x, y), (w, h), 0, 0, 360, color=0, thickness=-1)
                faces.append((y, x, value, w, h))

    filter_faces(faces)

    for i, face in enumerate(faces):
        y, x, value, width, height = face
        faces[i] = {
            "y": y,
            "x": x,
            "match_rating": value,
            "width": width,
            "height": height,
        }
        rect, (rect_x, rect_y) = get_rectangle(img, faces[i], margin=0)
        rect = masked_rect(rect.copy(), y-rect_y, x-rect_x, height, width)
        faces[i]["max temp"] = rect.max()
        max_y, max_x = np.unravel_index(np.argmax(rect), rect.shape)

        faces[i]["max temp x"] = max_x + rect_x
        faces[i]["max temp y"] = max_y + rect_y
        faces[i]["mean temp"] = rect.mean()
        faces[i]["min temp"] = rect.min()

    display_faces(img, faces)

    return sorted(faces, key=lambda x: x['match_rating'], reverse=True) 


