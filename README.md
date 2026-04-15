# Bottle Print Inspection with Line-Scan Vision and Anomaly Detection
### Line-Scan Vision + Anomaly Detection

> **Public repository note**  
> This is a public showcase version of the project.  
> Some source files and implementation details have been intentionally omitted or replaced with placeholders due to NDA and confidentiality restrictions.

Industrial computer vision system for inspection of cylindrical or square bottle prints with a **line-scan camera** and machine learning-based anomaly detection.

The project is designed in a modular way, allowing independent training, calibration, and visualization stages, with a possibility to trigger each of the actions from 3rd party devices which makes it easy to integrate with the hardware.

Tech stack:

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-D94F3D?style=for-the-badge&logo=pytorch&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-4C51BF?style=for-the-badge&logo=opencv&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-1F3A5F?style=for-the-badge&logo=numpy&logoColor=white)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-2C5F8A?style=for-the-badge)](#)
[![Anomalib](https://img.shields.io/badge/Anomalib-5B4FCF?style=for-the-badge&logoColor=white)](#)
[![PatchCore](https://img.shields.io/badge/PatchCore-374151?style=for-the-badge&logoColor=white)](#)
[![PySide](https://img.shields.io/badge/PySide6-2F6F4F?style=for-the-badge&logo=qt&logoColor=white)](#)

---

---
## Key features

- Tile-based processing for localized anomaly detection
- Per-tile model training for improved sensitivity to regional defects
- Percentile-based calibration for more consistent anomaly scoring
- Seam-aware stitching of tile-level anomaly maps
- Modular pipeline covering preprocessing, training, inference, and visualization
- Designed for line-scan inspection of cylindrical bottle prints
- GUI-based workflow for easier configuration and testing
- Supports synthetic defect generation for controlled experiments
- Suitable for integration with industrial inspection workflows

---

## Overview

This project implements an end-to-end industrial vision pipeline designed to detect print defects on rotating bottles in production environments.


---
## Pipeline overview

Input image → Preprocessing → Tiling → Per-tile inference → Calibration → Stitching → Final visualization

---
## Demo

A demo of the project is available [here](https://youtu.be/K-n7jh2-2q8):  
[![Watch the video](https://img.youtube.com/vi/K-n7jh2-2q8/maxresdefault.jpg)](https://youtu.be/K-n7jh2-2q8)

The demo presents the general workflow of the application and shows how inference and result visualization are triggered in the current development version.


## Project structure: 

<!-- TREE_START -->
```text
AnomalyDetection/
├── config/               # central configuration
│   ├── __init__.py
│   └── settings.py
├── data/                 # datasets, artifacts and model outputs
├── gui/                  # desktop GUI
│   ├── auth/             # authentication and local user access
│   │   ├── __init__.py
│   │   ├── auth_service.py
│   │   ├── db.py
│   │   └── login_dialog.py
│   ├── controllers/      # GUI workflow and business logic
│   │   ├── __init__.py
│   │   └── app_controller.py
│   ├── views/            # application windows and settings dialogs
│   │   ├── __init__.py
│   │   ├── main_window.py
│   │   ├── settings_dataset_dialog.py
│   │   ├── settings_defects_dialog.py
│   │   ├── settings_dialog.py
│   │   ├── settings_inference_dialog.py
│   │   └── settings_tiling_dialog.py
│   ├── widgets/          # reusable GUI widgets
│   │   ├── __init__.py
│   │   ├── image_viewer.py
│   │   └── log_panel.py
│   ├── __init__.py
│   ├── app_settings.py
│   └── main.py
├── scripts/              # thin entrypoints
│   ├── build_anomalib_dataset.py
│   ├── compute_tile_stats.py
│   ├── display_tiled_result.py
│   ├── generate_defects.py
│   ├── generate_tiles.py
│   ├── inference_run.py
│   ├── train_patchcore_full.py
│   ├── train_patchcore_tiles.py
│   └── update_readme_tree.py
├── tests/                # unit tests
│   ├── test_builder.py
│   ├── test_grouping.py
│   └── test_tiling.py
├── vision/               # core computer vision logic
│   ├── calibration/      # tile-level calibration and score normalization
│   │   ├── __init__.py
│   │   └── tile_stats.py
│   ├── dataset/          # dataset preparation and tiling builders
│   │   ├── __init__.py
│   │   ├── defect_dataset_builder.py
│   │   ├── preprocessed_dataset_builder.py
│   │   └── tiles_builder.py
│   ├── defects/          # synthetic defect generators
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── deadpixels.py
│   │   ├── stains.py
│   │   └── stripes.py
│   ├── inference/        # tile-based inference pipeline
│   │   ├── __init__.py
│   │   ├── tiled.py
│   │   └── utils.py
│   ├── preprocessing/    # image resizing, padding and tiling utilities
│   │   ├── __init__.py
│   │   ├── tiling.py
│   │   └── vision_preproc.py
│   ├── visualisation/    # stitching and result visualisation
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   └── visualisation.py
│   └── __init__.py
├── .gitignore
├── app_users.db
├── environment.yml
├── README.md
└── todo.md
```
<!-- TREE_END -->


---

## Project steps description

### 1. Configuration (`config/settings.py`)

The `config/settings.py` file contains the main, default parameters used across the project.  
It serves as a central configuration point for dataset paths, tiling, calibration, inference, and visualization.
These parameters can be modified in the GUI.

Main configuration groups:

- **Paths**
  - locations of datasets, model outputs, inference artifacts, and calibration files

- **Image size**
  - `TARGET_HEIGHT`, `TARGET_WIDTH` define the target image resolution before tiling

- **Tile layout**
  - `ROWS`, `COLS` define the image grid
  - `TILE_HEIGHT`, `TILE_WIDTH` define tile dimensions

- **Inference**
  - `IMAGE_TO_INFER` specifies the input image
  - `ALPHA` controls heatmap blending
  - `EPS` prevents division by zero during normalization

- **Calibration**
  - `MAP_PERCENTILE_LOW`, `MAP_PERCENTILE_HIGH` define the normalization range
  - `MASK_THRESHOLD` defines the anomaly threshold after calibration

- **Display / stitching**
  - `SMOOTH_TILE_BORDERS` enables border smoothing between neighboring tiles
  - `SMOOTH_BORDER_WIDTH`, `SMOOTH_BLUR_KERNEL` define smoothing behavior

---

### 2. Data preparation and tile generation

Scripts:
- `generate_defects.py`
- `build_anomalib_dataset.py`
- `generate_tiles.py`

This stage prepares the data for anomaly detection.

- `generate_defects.py`
  - creates synthetic defects for controlled experiments

- `build_anomalib_dataset.py`
  - prepares the dataset in a structure compatible with Anomalib

- `generate_tiles.py`
  - splits full images into tiles based on the configured layout
  - preserves tile size consistency
  - builds the tiled dataset used for per-tile training and inference

This step makes the pipeline suitable for high-resolution images and localized defect detection.

---

### 3. Model training

Scripts:
- `train_patchcore_tiles.py`

This stage trains anomaly detection models using normal (`good`) samples.

- `train_patchcore_tiles.py`
  - trains a separate PatchCore model for each tile
  - allows each region of the image to learn localized normal patterns
  - saves tile-specific outputs in a structured results directory

The tile-based approach improves sensitivity to local print defects in high-resolution line-scan images.

---

### 4. Calibration

Script:
- `compute_tile_stats.py`

This stage computes calibration statistics for each tile.

- extracts raw anomaly maps from normal samples
- computes percentile-based normalization ranges
- saves tile-level statistics to `tile_stats.json`
- improves consistency of anomaly scores across tiles

---

### 5. Inference

Script:
- `inference_run.py`

This stage runs tiled anomaly detection on a selected image.

- loads the corresponding trained model for each tile
- computes raw anomaly maps
- applies per-tile calibration
- saves intermediate outputs such as:
  - raw anomaly maps
  - calibrated maps
  - tile images
  - anomaly visualizations

---

### 6. Visualization and stitching

Script:
- `display_tiled_result.py`

This stage reconstructs the final full-image result.

- loads tile-level outputs
- stitches anomaly maps into a single full-resolution map
- optionally smooths tile borders
- generates final heatmap and overlay visualizations
- saves the final stitched result

