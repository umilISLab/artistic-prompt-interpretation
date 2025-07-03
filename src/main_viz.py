import warnings
import pandas as pd

import config
from model_utils import setup_diffusion_model
from analysis_utils import save_image_with_heatmaps

warnings.filterwarnings("ignore", module="daam.utils")
warnings.filterwarnings("ignore", module="huggingface_hub.file_download")


def run_visualization():
    pipe = setup_diffusion_model(config.MODEL_ID, config.DEVICE)

    results_df = pd.read_csv("./output/content_style_iou_results.csv")
    results_df["delta_iou"] = results_df["iou_baseline_mean"] - results_df["iou_score"]

    for iou_threshold_for_vis, use_quantile_for_vis in zip(
        config.DEFAULT_IOU_THRESHOLD_FOR_VIS, config.DEFAULT_USE_QUANTILE_FOR_VIS
    ):
        # Filter for the specific configuration for visualization
        plot_config_df = results_df.loc[
            (results_df["iou_threshold"] == iou_threshold_for_vis)
            & (results_df["use_quantile"] == use_quantile_for_vis)
        ]

        # Group by unique image generation settings and average delta_iou
        results_to_plot = (
            plot_config_df.groupby(
                ["prompt", "content_word", "style_word", "seed"], as_index=False
            )
            .agg({"delta_iou": "mean"})
            .sort_values(by="delta_iou", ascending=False)
            .dropna()
        )

        # Top N images
        top_dir_name = (
            f"{iou_threshold_for_vis}_"
            + f"{'quantile' if use_quantile_for_vis else 'threshold'}"
            + "/top"
        )
        TOP_DIR = config.IMAGE_DIR / top_dir_name
        TOP_DIR.mkdir(parents=True, exist_ok=True)

        for _, row in results_to_plot.head(config.N_RELEVANT_IMAGES_TO_SAVE).iterrows():
            save_image_with_heatmaps(
                pipe,
                row["prompt"],
                row["content_word"],
                row["style_word"],
                int(row["seed"]),
                TOP_DIR,
            )

        # Bottom N images
        bottom_dir_name = (
            f"{iou_threshold_for_vis}_"
            + f"{'quantile' if use_quantile_for_vis else 'threshold'}"
            + "/bottom"
        )
        BOTTOM_DIR = config.IMAGE_DIR / bottom_dir_name
        BOTTOM_DIR.mkdir(parents=True, exist_ok=True)

        for _, row in results_to_plot.tail(config.N_RELEVANT_IMAGES_TO_SAVE).iterrows():
            save_image_with_heatmaps(
                pipe,
                row["prompt"],
                row["content_word"],
                row["style_word"],
                int(row["seed"]),
                BOTTOM_DIR,
            )


if __name__ == "__main__":
    run_visualization()
