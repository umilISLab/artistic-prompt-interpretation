"""Setup diffusion model."""

import torch
from diffusers import DiffusionPipeline


@torch.no_grad()
def setup_diffusion_model(model_id: str, device: str):
    """
    Sets up and returns the diffusion pipeline.
    """
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=True,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
