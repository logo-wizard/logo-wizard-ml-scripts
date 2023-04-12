import cv2
import numpy as np
import torch


class LaMa:
    pad_mod = 8

    def __init__(self, model_path: str, device: str) -> None:
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device).to(device)
        self.model.eval()

    @staticmethod
    def _normalize_img(np_img: np.ndarray) -> np.ndarray:
        if len(np_img.shape) == 2:
            np_img = np_img[:, :, np.newaxis]
        np_img = np.transpose(np_img, (2, 0, 1))
        np_img = np_img.astype("float32") / 255
        return np_img

    def _forward(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image = self._normalize_img(image)
        mask = self._normalize_img(mask)

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        result = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()
        result = np.clip(result * 255, 0, 255).astype("uint8")

        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _ceil_size(x: int, mod: int) -> int:
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    @staticmethod
    def _pad_img(img: np.ndarray, mod: int) -> np.ndarray:
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        height, width = img.shape[:2]
        out_height = LaMa._ceil_size(height, mod)
        out_width = LaMa._ceil_size(width, mod)

        return np.pad(
            img,
            ((0, out_height - height), (0, out_width - width), (0, 0)),
            mode="symmetric",
        )

    @torch.no_grad()
    def __call__(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        origin_height, origin_width = image.shape[:2]
        pad_image = self._pad_img(image, mod=self.pad_mod)
        pad_mask = self._pad_img(mask, mod=self.pad_mod)

        result = self._forward(pad_image, pad_mask)
        result = result[0:origin_height, 0:origin_width, :]

        mask = mask[:, :, np.newaxis]
        result = result * (mask / 255) + image[:, :, ::-1] * (1 - (mask / 255))

        return result.astype("float32")
