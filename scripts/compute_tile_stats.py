from vision.calibration.tile_stats import TileStatsCalibrator

from config.settings import (
    DATASET_TILED_DIR,
    RESULTS_TILED_DIR,
    CALIBRATION_JSON_PATH,
    GOOD_DIR_NAME,
    MAP_PERCENTILE_LOW,
    MAP_PERCENTILE_HIGH,
)


def main():
    calibrator = TileStatsCalibrator(
        dataset_tiled_dir=DATASET_TILED_DIR,
        results_tiled_dir=RESULTS_TILED_DIR,
        calibration_json_path=CALIBRATION_JSON_PATH,
        good_dir_name=GOOD_DIR_NAME,
        map_percentile_low=MAP_PERCENTILE_LOW,
        map_percentile_high=MAP_PERCENTILE_HIGH,
    )

    calibrator.run()


if __name__ == "__main__":
    main()