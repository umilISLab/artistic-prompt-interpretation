"""Main script to run the IoU experiment for content and style in diffusion models."""

import warnings
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from daam import trace, set_seed
from daam.utils import auto_autocast

import config
from data_utils import generate_prompts, get_token_indices_for_template_words
from model_utils import setup_diffusion_model
from analysis_utils import (
    calculate_iou_metrics,
    calculate_iou_baseline,
)

warnings.filterwarnings("ignore", module="daam.utils")
warnings.filterwarnings("ignore", module="huggingface_hub.file_download")


def run_experiment():
    """
    Main function to run the IoU experiment.
    """
    # 1. Setup Model
    pipe = setup_diffusion_model(config.MODEL_ID, config.DEVICE)

    # 2. Generate and Export Prompts
    prompts_info_list = generate_prompts(
        config.PROMPT_TEMPLATES, config.MEDIUMS, config.CONTENTS, config.STYLES
    )
    prompts_info_df = pd.DataFrame(prompts_info_list)
    prompts_info_df["seed"] = config.SEED_START_VALUE + prompts_info_df.index
    prompts_info_df.to_csv(config.OUTPUT_DIR / "prompts.csv", index=False)

    results = []

    # 3. Main Processing Loop
    current_seed = config.SEED_START_VALUE
    for prompt_info in tqdm(prompts_info_list, desc="Processing prompts"):
        gen = set_seed(current_seed)  # Set seed for reproducibility for this prompt

        template = prompt_info["template"]
        prompt_text = prompt_info["text"]
        content_word = prompt_info["content_word"]
        style_word = prompt_info["style_word"]
        medium_word = prompt_info["medium_word"]

        resolved_words, token_indices, content_idx, style_idx, _ = (
            get_token_indices_for_template_words(
                pipe.tokenizer,
                prompt_text,
                template,
                content_word,
                style_word,
                medium_word,
            )
        )

        with torch.no_grad(), trace(pipe) as tc:
            out = pipe(
                prompt_text,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
                generator=gen,
            )
            image = out.images[0]
            global_heat_map = tc.compute_global_heat_map()

            word_heatmaps = []
            for word, token_idx in zip(resolved_words, token_indices):
                heatmap = global_heat_map.compute_word_heat_map(
                    word=word, word_idx=token_idx
                ).expand_as(image=image)
                word_heatmaps.append(heatmap)

            content_map = word_heatmaps[content_idx]
            style_map = word_heatmaps[style_idx]

            if config.OUTPUT_ALL_IMAGES:
                # Save the generated image and heatmaps
                image.resize(config.OUTPUT_IMAGE_DIMENSIONS, Image.LANCZOS).save(
                    config.ALL_IMAGE_DIR / f"{current_seed}_og.jpg"
                )

                for m in (content_map, style_map):
                    suffix = "cont" if m is content_map else "st"
                    map_output_path = (
                        config.ALL_IMAGE_DIR / f"{current_seed}_{suffix}.jpg"
                    )
                    with auto_autocast(dtype=torch.float32):
                        im = np.array(image)
                        plt.imshow(m.squeeze().cpu().numpy(), cmap="jet")
                        im = torch.from_numpy(im).float() / 255
                        im = torch.cat((im, (1 - m.unsqueeze(-1))), dim=-1)
                        plt.imshow(im)
                        plt.axis("off")
                        plt.savefig(
                            map_output_path,
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                        map_image = Image.open(map_output_path)
                        map_image = map_image.resize(
                            config.OUTPUT_IMAGE_DIMENSIONS, Image.LANCZOS
                        )
                        map_image.save(map_output_path)
                        plt.clf()

            for t_val in config.IOU_THRESHOLDS_TO_TEST:
                for use_quantile_flag in [False, True]:
                    (
                        iou_bl_mean,
                        iou_bl_std,
                        sup_a_bl,
                        sup_b_bl,
                        sup_int_bl,
                        sup_union_bl,
                    ) = calculate_iou_baseline(
                        word_heatmaps,
                        content_idx,
                        style_idx,
                        content_style_only=False,
                        threshold=t_val,
                        use_quantile=use_quantile_flag,
                        device=config.DEVICE,
                    )

                    (
                        iou_bl_cs_mean,
                        iou_bl_cs_std,
                        sup_a_bl_cs,
                        sup_b_bl_cs,
                        sup_int_bl_cs,
                        sup_union_bl_cs,
                    ) = calculate_iou_baseline(
                        word_heatmaps,
                        content_idx,
                        style_idx,
                        content_style_only=True,
                        threshold=t_val,
                        use_quantile=use_quantile_flag,
                        device=config.DEVICE,
                    )

                    iou_score, sup_content, sup_style, sup_intersect, sup_union = (
                        calculate_iou_metrics(
                            content_map,
                            style_map,
                            t_val,
                            use_quantile_flag,
                            config.DEVICE,
                        )
                    )

                    results.append(
                        {
                            "prompt": prompt_text,
                            "template": template,
                            "content_word": content_word,
                            "style_word": style_word,
                            "seed": current_seed,
                            "use_quantile": use_quantile_flag,
                            "iou_threshold": t_val,
                            "iou_baseline_mean": iou_bl_mean,
                            "iou_baseline_std": iou_bl_std,
                            "support_a_baseline": sup_a_bl,
                            "support_b_baseline": sup_b_bl,
                            "support_intersection_baseline": sup_int_bl,
                            "support_union_baseline": sup_union_bl,
                            "iou_baseline_cs_mean": iou_bl_cs_mean,
                            "iou_baseline_cs_std": iou_bl_cs_std,
                            "support_a_baseline_cs": sup_a_bl_cs,
                            "support_b_baseline_cs": sup_b_bl_cs,
                            "support_intersection_baseline_cs": sup_int_bl_cs,
                            "support_union_baseline_cs": sup_union_bl_cs,
                            "iou_score": iou_score,
                            "support_content": sup_content,
                            "support_style": sup_style,
                            "support_intersection": sup_intersect,
                            "support_union": sup_union,
                        }
                    )

        # Memory cleanup
        if config.DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        elif config.DEVICE == "mps":
            torch.mps.empty_cache()

        current_seed += 1

    # 4. Save Results
    results_df = pd.DataFrame(results)
    csv_path = config.OUTPUT_DIR / "content_style_iou_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    run_experiment()
