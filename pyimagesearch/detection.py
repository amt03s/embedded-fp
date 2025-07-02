# import the necessary packages
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2

def detect_people(frame, net, personIdx=15):
    # MobileNet SSD's 'person' class is index 15 in the COCO/VOC dataset
    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > MIN_CONF:
            idx = int(detections[0, 0, i, 1])

            if idx == personIdx:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                cX = int((startX + endX) / 2.0)
                cY = int((startY + endY) / 2.0)

                r = (confidence, (startX, startY, endX, endY), (cX, cY))
                results.append(r)

    return results
