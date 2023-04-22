import random

import cv2
import einops
import numpy as np
import torch
from annotator_util import HWC3, resize_image
from canny_detector import CannyDetector
from ddim_hacked import DDIMSampler
from model_config import create_model, load_state_dict
from pytorch_lightning import seed_everything

preprocessor = None

model_name = "control_v11p_sd15_canny"
model = create_model(f"./models/{model_name}.yaml").cpu()
model.load_state_dict(
    load_state_dict("./models/v1-5-pruned.ckpt", location="cuda"), strict=False
)
model.load_state_dict(
    load_state_dict(f"./models/{model_name}.pth", location="cuda"), strict=False
)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(
    det,
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    detect_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    low_threshold,
    high_threshold,
):
    global preprocessor

    if det == "Canny":
        if not isinstance(preprocessor, CannyDetector):
            preprocessor = CannyDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == "None":
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(
                resize_image(input_image, detect_resolution),
                low_threshold,
                high_threshold,
            )
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results
