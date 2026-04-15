from config.settings import (
    DEFECT_BASE_IMAGE_PATH,
    DEFECT_OUTPUT_DIR,
    DEFECT_CLEAR_OUTPUT,
    STRIPES_MAX_WIDTH,
    STRIPES_N_IMAGES,
    STRIPES_LINE_COUNT_RANGE,
    STRIPES_SEED,
    BLACK_PIXELS_N_IMAGES,
    BLACK_PIXELS_FILL_PERCENT_RANGE,
    BLACK_PIXELS_SEED,
)


from vision.dataset.defect_dataset_builder import DefectDatasetBuilder


def main() -> None:
    builder = DefectDatasetBuilder(
        output_root=DEFECT_OUTPUT_DIR,
        clear_output=DEFECT_CLEAR_OUTPUT,
    )

    builder.generate_stripes_dataset(
        base_image_path=DEFECT_BASE_IMAGE_PATH,
        max_width=STRIPES_MAX_WIDTH,
        n_images=STRIPES_N_IMAGES,
        line_count_range=STRIPES_LINE_COUNT_RANGE,
        seed=STRIPES_SEED,
    )

    builder.generate_random_black_pixels_dataset(
        base_image_path=DEFECT_BASE_IMAGE_PATH,
        n_images=BLACK_PIXELS_N_IMAGES,
        fill_percent_range=BLACK_PIXELS_FILL_PERCENT_RANGE,
        seed=BLACK_PIXELS_SEED,
    )


if __name__ == "__main__":
    main()