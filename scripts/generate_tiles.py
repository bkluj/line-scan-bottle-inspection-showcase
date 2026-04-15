from vision.dataset.tiles_builder import TilesBuilder
from config.settings import DATASET_NORMAL_DIR, DATASET_TILED_DIR, TILE_HEIGHT, TILE_WIDTH

builder = TilesBuilder(tile_size=(TILE_HEIGHT, TILE_WIDTH))

builder.generate_tiles_dataset(
    input_root=DATASET_NORMAL_DIR,
    output_root=DATASET_TILED_DIR
)
