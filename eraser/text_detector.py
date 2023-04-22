from typing import List

import cv2
import numpy as np
from mmocr.apis import MMOCRInferencer


class Detector:
    """Responsible for detecting text in an image using the MMOCRInferencer API."""

    def __init__(self, detector: str, device: str) -> None:
        self.model = MMOCRInferencer(det=detector, device=device)

    @staticmethod
    def _split_array(array: List[int], n: int) -> List[List[int]]:
        """Splits an array into n subarrays of equal size (or as close as possible).

        Args:
            array (List[int]): list to split into subarrays.
            n (int): number of subarrays.

        Returns:
            List[List[int]]: a list of subarrays.
        """
        k, m = divmod(len(array), n)
        return (
            array[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
        )

    @staticmethod
    def _scale_contour(points: np.ndarray, scale: float) -> np.ndarray:
        """Scales a contour by a given factor around its centroid and returns the new contour.

        Args:
            points (np.ndarray): contour to scale.
            scale (float): scale factor.

        Returns:
            np.ndarray: scaled contour.
        """
        M = cv2.moments(points)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cnt_norm = points - [cx, cy]
        return (cnt_norm * scale + [cx, cy]).astype(np.int32)

    def __call__(self, image: np.ndarray, contour_scale: float = 1.4) -> np.ndarray:
        """Processes an image by detecting text regions using the MMOCRInferencer API, scaling the detected contours by a given factor.

        Args:
            image (np.ndarray): image to process.
            contour_scale (float, optional): scale factor. Defaults to 1.4.

        Returns:
            np.ndarray: a binary mask where the text regions are marked with white pixels.
        """
        result = self.model(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        mask = np.zeros(image.shape[:2], dtype="uint8")
        text_boxes = result["predictions"][0]["det_polygons"]
        for text_box in text_boxes:
            text_box = list(map(int, text_box))
            points = np.array(list(self._split_array(text_box, 4)))
            points = self._scale_contour(points, scale=contour_scale)
            cv2.fillPoly(mask, [points], 255)

        return mask
