import cv2
import numpy as np
from lama_model import LaMa

lama = LaMa(model_path="model/big-lama.pt", device="cpu")

image = cv2.imread("test_data/test.jpg")
mask = np.load("test_data/mask.npy")

res = lama(image, mask)
res = res.astype("float32")
res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

cv2.imwrite("test_data/result.jpg", res)
