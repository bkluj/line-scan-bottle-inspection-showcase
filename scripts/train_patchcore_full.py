from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

from config.settings import DATASET_NORMAL_DIR, RESULTS_NORMAL_DIR

datamodule = Folder(
    name="visioncontrol_full",
    root=DATASET_NORMAL_DIR,
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
    default_root_dir=RESULTS_NORMAL_DIR,
)

engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)