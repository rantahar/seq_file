import cv2
import numpy as np

def detect_heads(img, thresh_lower=28, thresh_upper=35, min_size=100, min_circularity=0.5):
    # Scale the temperature values to 0-255 range
    #img_gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    thresh_lower = (thresh_lower - 10) / 40*255
    thresh_upper = (thresh_upper - 10) / 40*255
    img_gray = ((img - 10) / 40*255).astype('uint8')

    # Threshold the image so that only temperatures in the range of 20-24 C are kept
    _, thresh = cv2.threshold(img_gray, thresh_upper, 255, cv2.THRESH_BINARY_INV)
    _, thresh_inv = cv2.threshold(img_gray, thresh_lower, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, thresh_inv)
    cv2.imshow('Thresholded Image', thresh)

    # Apply morphological closing to remove small holes in the segmented regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('thresh2', thresh)

    # Perform a distance transform to create a gradient image
    dist_transform = cv2.distanceTransform(thresh.astype('uint8'), cv2.DIST_L2, 5)
    grad_thresh = cv2.threshold(dist_transform, 0.1*dist_transform.max(), 255, 0)[1]
    cv2.imshow('grad_thresh', grad_thresh)

    # Apply a binary threshold to the gradient image to segment the different heads
    grad_thresh = np.uint8(grad_thresh)
    contours, _ = cv2.findContours(grad_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("dist_transform", dist_transform)

    contours = [c for c in contours if cv2.contourArea(c) > min_size]

    # Loop through the contour regions to calculate the temperature and position of each head
    heads = []
    for i in range(len(contours)):
        # Compute the centroid of the contour region
        moments = cv2.moments(contours[i])
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

            # Compute circularity
            ((x, y), radius) = cv2.minEnclosingCircle(contours[i])
            area = cv2.contourArea(contours[i])
            perimeter = cv2.arcLength(contours[i], True)
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            print(x,y,radius, circularity)

            if circularity < min_circularity:
                continue

            # Calculate the average temperature of the head by computing the mean temperature
            # value of the pixels within the segmented region
            mask = np.zeros(img_gray.shape, np.uint8)
            cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
            img_masked = img_gray.copy()
            img_masked[np.where(mask == 0)] = 0

            mean_temp = np.mean(img_masked)*40/255+10,
            min_temp = np.min(img_masked)*40/255+10,
            max_temp = np.max(img_masked)*40/255+10,

            max_y, max_x = np.unravel_index(np.argmax(img_masked), img_masked.shape)
            print(max_x, max_y)
            cv2.circle(img_gray, (max_x, max_y), 5, (0, 0, 255), -1)

            # Add the head's temperature and position (x, y) to the list
            head = {"mean temp": mean_temp, "max temp": max_temp, "min temp": min_temp, "position": (cx, cy), "area": cv2.contourArea(contours[i])}
            heads.append(head)
            print(head)
    cv2.drawContours(img_gray, contours, -1, (0, 255, 0), 2)

    cv2.imshow("img_gray", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return heads
