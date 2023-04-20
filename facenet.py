import torch
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_heads(temp_array):
    # Create a tensor from the input numpy array, with the same shape and data type
    img = np.stack([temp_array]*3, axis=2)
    temp_image = ((temp_array - temp_array.min()) / (temp_array.max() - temp_array.min()) * 255).astype(np.uint8)
    cv2.imwrite('temp.png', temp_image)

    result = model('temp.png')

    print(result.pandas().xyxy[0])

    
