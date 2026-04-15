from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

from config.settings import DATASET_TILED_DIR, RESULTS_TILED_DIR

for root in sorted(DATASET_TILED_DIR.iterdir()):
    tile_name = root.name

    datamodule = Folder(
        name="visioncontrol_full",
        root=root,
        normal_dir="good",
        abnormal_dir="defect",
        mask_dir=None,
        normal_split_ratio=0.2,
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=4,
    )

    model = Patchcore()

    engine = Engine(
        max_epochs=1,
        default_root_dir=RESULTS_TILED_DIR / tile_name,
        logger=False,
    )

    print(root)
    engine.fit(model=model, datamodule=datamodule)