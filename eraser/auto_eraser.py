import cv2
import numpy as np
from text_detector import Detector
from text_eraser import LaMa


class AutoEraser:
    def __init__(
        self, eraser: str, detector: str = "dbnetpp", device: str = "cuda"
    ) -> None:
        self.detector = Detector(detector=detector, device=device)
        self.eraser = LaMa(model_path=eraser, device=device)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        mask = self.detector(image)
        result = self.eraser(image, mask)

        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
