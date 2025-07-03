import itertools
from typing import List, Tuple
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from daam import trace, set_seed
from diffusers import DiffusionPipeline


def calculate_iou_metrics(
    map_a: torch.Tensor,
    map_b: torch.Tensor,
    threshold: float,
    use_quantile: bool,
    device: str,
) -> Tuple[float, float, float, float, float]:
    """
    Calculates Intersection over Union (IoU) and support metrics between two heatmaps.

    Returns:
        tuple: (
            iou_score,
            support_a_norm,
            support_b_norm,
            support_intersection_norm,
            support_union_norm
        )
    """
    map_a = map_a.float().to(device)
    map_b = map_b.float().to(device)
    n = map_a.numel()

    if use_quantile:
        threshold_a = map_a.quantile(threshold)
        threshold_b = map_b.quantile(threshold)
        bin_a = map_a >= threshold_a
        bin_b = map_b >= threshold_b
    else:
        bin_a = map_a >= threshold
        bin_b = map_b >= threshold

    intersection_map = bin_a & bin_b
    union_map = bin_a | bin_b

    intersection = intersection_map.float().sum()
    union = union_map.float().sum()

    iou_score = 0.0 if union < 1e-6 else (intersection / union).item()

    support_a_norm = (bin_a.float().sum().item()) / n
    support_b_norm = (bin_b.float().sum().item()) / n
    support_intersection_norm = (intersection.item()) / n
    support_union_norm = (union.item()) / n

    return (
        iou_score,
        support_a_norm,
        support_b_norm,
        support_intersection_norm,
        support_union_norm,
    )


def calculate_iou_baseline(
    word_heatmaps: List[torch.Tensor],
    content_idx: int,
    style_idx: int,
    content_style_only: bool,
    threshold: float,
    use_quantile: bool,
    device: str,
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculates baseline IoU metrics by comparing non-content/style word pairs
    or pairs involving content/style words with other words.
    """
    iou_scores_data = []
    for wi, wj in itertools.combinations(range(len(word_heatmaps)), r=2):
        if (wi == content_idx and wj == style_idx) or (
            wi == style_idx and wj == content_idx
        ):
            continue  # Skip the direct content-style pair

        if content_style_only and not any(
            x for x in [wi, wj] if x in [content_idx, style_idx]
        ):
            continue
        iou_data = calculate_iou_metrics(
            word_heatmaps[wi], word_heatmaps[wj], threshold, use_quantile, device
        )
        iou_scores_data.append(iou_data)

    iou_values = [data[0] for data in iou_scores_data]
    sup_a_values = [data[1] for data in iou_scores_data]
    sup_b_values = [data[2] for data in iou_scores_data]
    sup_int_values = [data[3] for data in iou_scores_data]
    sup_union_values = [data[4] for data in iou_scores_data]

    return (
        np.mean(iou_values),
        np.std(iou_values),
        np.mean(sup_a_values),
        np.mean(sup_b_values),
        np.mean(sup_int_values),
        np.mean(sup_union_values),
    )


@torch.no_grad()
def save_image_with_heatmaps(
    pipe: DiffusionPipeline,
    prompt: str,
    content_word: str,
    style_word: str,
    seed: int,
    img_dir: Path,
    num_inference_steps: int = 30,
):
    """
    Generates an image and overlays heatmaps for content and style words, then saves it.
    """
    gen = set_seed(seed)
    with trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=num_inference_steps, generator=gen)
        original_image = out.images[0]
        global_heat_map = tc.compute_global_heat_map()
        content_map = global_heat_map.compute_word_heat_map(content_word)
        style_map = global_heat_map.compute_word_heat_map(style_word)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        content_map.plot_overlay(original_image, ax=axes[1])
        axes[1].set_title(f"Content: '{content_word}'")
        axes[1].axis("off")

        style_map.plot_overlay(original_image, ax=axes[2])
        axes[2].set_title(f"Style: '{style_word}'")
        axes[2].axis("off")

        plt.suptitle(f"Prompt: {prompt}", y=0.075)
        plt.tight_layout(rect=[0, 0.1, 1, 1])

        filename = f"img_s{seed}_{content_word.replace(' ','_')}_{style_word.replace(' ','_')}_overlay.png"
        fig.savefig(img_dir / filename)
        plt.close(fig)
