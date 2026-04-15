from config.settings import (
    PREPROCESSED_INPUT_DIR,
    PREPROCESSED_OUTPUT_DIR,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    PREPROCESS_PADDING_BGR,
    PREPROCESS_OUT_EXT,
)

from vision.dataset.preprocessed_dataset_builder import PreprocessedDatasetBuilder
from vision.preprocessing.vision_preproc import PreprocConfig


def main() -> None:
    preproc_config = PreprocConfig(
        target_height=TARGET_HEIGHT,
        target_width=TARGET_WIDTH,
        padding_bgr=PREPROCESS_PADDING_BGR,
        out_ext=PREPROCESS_OUT_EXT,
    )

    builder = PreprocessedDatasetBuilder(
        input_dir=PREPROCESSED_INPUT_DIR,
        output_dir=PREPROCESSED_OUTPUT_DIR,
        preproc_config=preproc_config,
        clear_output=True,
    )

    builder.run()


if __name__ == "__main__":
    main()