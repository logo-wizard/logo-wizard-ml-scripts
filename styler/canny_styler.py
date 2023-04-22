import cv2
import numpy as np
import PIL
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)


class Canny:
    """Takes an input image and generates a new image based on a given style prompt using the Canny edge detection algorithm."""

    def __init__(
        self,
        controlnet_path: str = "thepowefuldeez/sd21-controlnet-canny",
        sd_path: str = "stabilityai/stable-diffusion-2-1-base",
    ) -> None:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_path,
            torch_dtype=torch.float16,
            controlnet=controlnet,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

    def _image_to_canny(
        self, image: PIL.Image.Image, low_th: int = 100, high_th: int = 200
    ) -> PIL.Image.Image:
        """Converts an input PIL image to a Canny edge-detected image using Canny function and concatenates the result to create a 3-channel image.

        Args:
            image (PIL.Image.Image): image to process.
            low_th (int, optional): lower threshold for Canny algorithm. Defaults to 100.
            high_th (int, optional): higher threshold for Canny algorithm. Defaults to 200.

        Returns:
            PIL.Image.Image: canny (edges) image.
        """
        image = cv2.Canny(np.array(image), low_th, high_th)[:, :, None]
        image = np.concatenate([image, image, image], axis=2)

        return PIL.Image.fromarray(image)

    def __call__(
        self,
        image: PIL.Image.Image,
        style: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 9,
        controlnet_conditioning_scale: float = 1.0,
    ) -> PIL.Image.Image:
        """Generates a new image based on the input image and style prompt using the Canny edge detection algorithm and the Stable Diffusion ControlNet Pipeline.

        Args:
            image (PIL.Image.Image): image to process.
            style (str): style prompt.
            num_inference_steps (int, optional): number of denoising steps. Defaults to 20.
            guidance_scale (float, optional): weight of the prompt. Defaults to 9.
            controlnet_conditioning_scale (float, optional): weight of the controlnet. Defaults to 1.0.

        Returns:
            PIL.Image.Image: stylized image.
        """
        prompt = style + ", best quality, extremely detailed"
        negative_prompt = (
            "monochrome, lowres, worst quality, low quality, text, inscription"
        )

        canny_image = self._image_to_canny(image=image)
        output = self.pipe(
            prompt,
            canny_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

        return output
