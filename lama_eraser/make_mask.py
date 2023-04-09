import cv2
import numpy as np

image = cv2.imread("test.jpg")
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (21, 511), (727, 713), 255, -1)
cv2.rectangle(mask, (300, 725), (467, 758), 255, -1)
np.save("mask.npy", mask)
