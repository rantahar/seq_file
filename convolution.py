import cv2
import numpy as np


def make_kernel(width, height):
    kernel_center = (height, width)
    a, b = width // 2, height // 2

    kernel = np.zeros((2*height, 2*width))
    y_indices, x_indices = np.mgrid[0:2*height, 0:2*width]
    x_ind = x_indices - kernel_center[1]
    y_ind = y_indices - kernel_center[0]
    distances = np.sqrt((y_ind)**2/b**2 + (x_ind)**2/a**2)

    kernel[distances < 1.0] = 1
    kernel[(distances < 1.2) & (distances >= 1.0)] = -0.8
    kernel[distances < 0.2] = -0.5
    kernel = kernel/(width*height)
    return kernel


def ellipses_overlap(ellipse1, ellipse2):
    y1, x1, value1, width1, height1 = ellipse1
    y2, x2, value2, width2, height2 = ellipse2

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

def filter_faces(faces):
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


def detect_heads(img, min_width = 20, max_width=80, width_step=1.2, height_ratios = [1.2, 1.3, 1.4], clip_range = [25, 37], threshold=16):
    # Scale the temperature values to 0-255 range
    gray = ((img - 10) / 40*255).astype('uint8')

    faces = []
    width = min_width
    while width < max_width:
        for height_ratio in height_ratios:
            kernel = make_kernel(int(width), int(height_ratio*width))
            img_clipped = img.copy()
            img_clipped[img<clip_range[0]] = 0
            img_clipped[img>clip_range[1]] = 0
            z = cv2.filter2D(img_clipped, -1, kernel)

            for i in range(100):
                y, x = np.unravel_index(np.argmax(z), z.shape)
                value = z[y, x]
                if value < threshold:
                    break

                cv2.ellipse(z, (x, y), (int(width), int(height_ratio * width)), 0, 0, 360, color=0, thickness=-1)
                faces.append((y, x, value, int(width), int(height_ratio * width)))
        width *= width_step

    filter_faces(faces)

    for face in faces:
        y, x, value, width, height = face
        print(y,x,value,width)
        cv2.ellipse(gray, (x, y), (width // 2, height // 2), 0, 0, 360, color=(0, 255, 0), thickness=2)

    img_clipped = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

