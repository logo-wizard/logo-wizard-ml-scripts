import cv2
from auto_eraser import AutoEraser

if __name__ == "__main__":
    model = AutoEraser(eraser="model/big-lama.pt", detector="dbnetpp", device="cuda")

    image = cv2.imread("test_data/test.jpg")
    result = model(image)
    cv2.imwrite("test_data/result.jpg", result)
