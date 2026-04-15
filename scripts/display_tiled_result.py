from config.settings import (
    OUTPUT_DIR,
    ROWS,
    COLS,
    IMAGE_TO_INFER,
    ALPHA,
    ENABLE_BORDER_SMOOTHING,
    SMOOTH_BORDER_WIDTH,
    SMOOTH_BLUR_KERNEL,
    STITCHED_VIS_MAP_PATH,
    STITCHED_HEATMAP_PATH,
    STITCHED_OVERLAY_PATH,
)

from vision.visualisation.visualisation import TiledResultStitcher

def main():
    stitcher = TiledResultStitcher(
        title="Image",
        output_dir=OUTPUT_DIR,
        rows=ROWS,
        cols=COLS,
        image_to_infer=IMAGE_TO_INFER,
        alpha=ALPHA,
        enable_border_smoothing=ENABLE_BORDER_SMOOTHING,
        smooth_border_width=SMOOTH_BORDER_WIDTH,
        smooth_blur_kernel=SMOOTH_BLUR_KERNEL,
        stitched_vis_map_path=STITCHED_VIS_MAP_PATH,
        stitched_heatmap_path=STITCHED_HEATMAP_PATH,
        stitched_overlay_path=STITCHED_OVERLAY_PATH,
    )

    stitcher.run(show=True)


if __name__ == "__main__":
    main()