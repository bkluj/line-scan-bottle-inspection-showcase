from vision.inference.tiled import TiledInferenceRunner
from config.settings import (
    DATASET_TILED_DIR,
    RESULTS_TILED_DIR,
    OUTPUT_DIR,
    CALIBRATION_JSON_PATH,
    IMAGE_TO_INFER,
    ALPHA,
    EPS,
    MASK_THRESHOLD,
)


def main():
    runner = TiledInferenceRunner(
        dataset_tiled_dir=DATASET_TILED_DIR,
        results_tiled_dir=RESULTS_TILED_DIR,
        output_dir=OUTPUT_DIR,
        calibration_json_path=CALIBRATION_JSON_PATH,
        image_to_infer=IMAGE_TO_INFER,
        alpha=ALPHA,
        eps=EPS,
        mask_threshold=MASK_THRESHOLD,
    )
    runner.run()


if __name__ == "__main__":
    main()