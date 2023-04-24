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
    #kernel[(y_ind > 0) & (y_ind < 0.4*width)] = -1
    kernel[(distances < 1.3) & (distances >= 1.0)] = -0.8
    kernel[distances < 0.3] = -1
    #kernel[distances > 2] = 0
    kernel[y_ind>height*0.1] = 0
    kernel = kernel/(width*height)
    return kernel


def detect_heads(img, widths = [30, 40, 60], height_ratios = [1.2, 1.4, 1.6], max_temperature = 37):
    # Scale the temperature values to 0-255 range
    max_temperature = ((max_temperature - 10) / 50*255)
    gray = ((img - 10) / 50*255).astype('uint8')

    result = np.zeros(gray.shape)
    mask = np.zeros(gray.shape)
    i = 0
    for width in widths:
        for height_ratio in height_ratios:
            kernel = make_kernel(width, int(height_ratio*width))
            img_clipped = np.clip(img, 0, max_temperature)
            z = cv2.filter2D(img_clipped, -1, kernel)
            #kernel[kernel <= 0] = 0
            #kernel = kernel
            #img_clipped = np.clip(img, max_temperature, 100) - max_temperature
            #print(z.max(), z.min())
            #z += -2*cv2.filter2D(img_clipped, -1, kernel)
            result += z

            #mask = np.zeros(gray.shape)
            mask[z > 4.7] = 1

            y, x = np.unravel_index(np.argmax(z), z.shape)
            print(i, width, height_ratio, y, x, z.max(), z.min())

            kernel = cv2.normalize(kernel, None, 0, 1.0, cv2.NORM_MINMAX)
            #cv2.imshow(f'gray{i}', gray)
            #cv2.imshow(f'kernel{i}', kernel)
            #cv2.imshow(f'mask{i}', mask)
            z = cv2.normalize(z, None, 0, 1.0, cv2.NORM_MINMAX)
            #cv2.imshow(f'z{i}', z)

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            i = i+1

    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(gray, contours, -1, (0, 255, 0), 2)

    cv2.imshow("gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


