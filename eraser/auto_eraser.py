import cv2
import numpy as np
from text_detector import Detector
from text_eraser import LaMa


class AutoEraser:
    """Automatically erase text from an image using a combination of a text detector and an inpainting model."""

    def __init__(
        self, eraser: str, detector: str = "dbnetpp", device: str = "cuda"
    ) -> None:
        self.detector = Detector(detector=detector, device=device)
        self.eraser = LaMa(model_path=eraser, device=device)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Processes an image by detecting text regions using the MMOCRInferencer API, generating a binary mask where the text regions are marked with white pixels, and passing the image and mask through the PyTorch model to fill in the missing parts of the image.

        Args:
            image (np.ndarray): image to process.

        Returns:
            np.ndarray: inpainted image.
        """
        mask = self.detector(image)
        result = self.eraser(image, mask)

        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
