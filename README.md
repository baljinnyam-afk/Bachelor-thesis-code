# Spacecraft Attitude Estimation Using Deep Learning

Bachelor Thesis — Synthetic Dataset Generation and ResNet50-Based Quaternion Regression for LEO Spacecraft Pose Estimation

---

## Overview

This project implements a complete pipeline for estimating the attitude (orientation) of a spacecraft in Low Earth Orbit (LEO) using deep learning. It covers:

1. **Synthetic dataset generation** — Photorealistic rendered images of a spacecraft using BlenderProc with physically accurate lighting.
2. **Model training** — ResNet50-based neural network for quaternion regression, trained in two phases.
3. **Evaluation** — Error analysis, prediction visualization, and soft-label augmentation experiments.

The target spacecraft model used is **CloudSat**, a NASA Earth-observation satellite.

---

## Project Structure

```
.
├── dataset_generataion_scripts/    # BlenderProc scripts for rendering training images
├── background_textures/            # Earth, Moon, Sun, and star textures (2K & 8K)
├── CloudSat.glb                    # 3D model of the CloudSat spacecraft
├── sun_dataset_v2/                 # Generated dataset (images + CSV labels)
├── dataset_row_edit/               # CSV renumbering and dataset correction utilities
├── training_model_scripts/         # ResNet50 training scripts, notebooks, and results
├── cnn test/                       # Early prototyping notebooks (MNIST, ResNet50 baseline)
├── checkpoints/                    # Saved model weights
│   ├── phase1/                     # Phase 1 training checkpoint
│   └── phase2/                     # Phase 2 training checkpoint
├── command_conda_env.txt           # Conda environment activation commands
└── command_blender.txt             # Blender-related commands
```

---

## Dataset Generation

Synthetic images are generated using [BlenderProc](https://github.com/DLR-RM/BlenderProc) with the Cycles renderer (GPU-accelerated via CUDA).

### Scripts

| Script | Description |
|---|---|
| `dataset_generation_sat_v2.py` | Initial version — basic satellite rendering |
| `dataset_generation_sat_v3.py` | Improved camera sampling and lighting |
| `dataset_generation_sat_v4_improved.py` | Enhanced realism and texture variety |
| `dataset_generation_sat_sun_v5.py` | **Final version** — visible sun disc, lens glare, per-frame shadow diversity |
| `dataset_generation_sat_v10_sun_moon.py` | Sun + moon variant with additional celestial bodies |

### Key Features (v5)

- **Visible sun disc** — Emissive sphere at realistic angular diameter (0.53°), warm white (5778K)
- **Per-frame sun direction** — Shadows change every rendered frame for maximum diversity
- **Lens glare** — Compositor-driven streaks and fog glow for photorealistic bloom
- **Earth backgrounds** — Day/night/cloud textures at 2K and 8K resolution
- **CSV output** — Columns: `image_name, qw, qx, qy, qz, tx, ty, tz, sun_dx, sun_dy, sun_dz`

### Usage

```bash
blenderproc run dataset_generataion_scripts/dataset_generation_sat_sun_v5.py
blenderproc run dataset_generataion_scripts/dataset_generation_sat_sun_v5.py --num_images 500 --samples 48
```

---

## Model Training

The attitude estimation model is based on **ResNet50** (ImageNet pre-trained), modified for quaternion regression. The architecture follows approaches from the [SPEED benchmark](https://arxiv.org/abs/1907.04195) (Sharma et al., 2019) and [UrsoNet/SPN](https://arxiv.org/abs/2004.05076) (Proença & Gao, 2020).

### Training Pipeline

| Component | Details |
|---|---|
| Backbone | ResNet50 (ImageNet weights, frozen initial layers) |
| Output | 4-dim quaternion `(qw, qx, qy, qz)` |
| Loss | Quaternion geodesic loss + MSE |
| Optimizer | Adam with ReduceLROnPlateau scheduling |
| Framework | TensorFlow 2.x / Keras |
| Augmentation | Soft-label augmentation, image transforms |

### Two-Phase Training

- **Phase 1** — Initial training with standard augmentation
- **Phase 2** — Fine-tuning with soft-label augmentation and adjusted hyperparameters

### Files

| File | Description |
|---|---|
| `resnet50_spacecraft_attitude_tf_v2.py` | Main training script (standalone Python) |
| `Resnet_train_script.ipynb` | Jupyter notebook — initial training |
| `Resnet_train_script_v2.ipynb` | Jupyter notebook — improved training loop |
| `Resnet_train_script_v3_ursonet.ipynb` | Jupyter notebook — UrsoNet-inspired variant |

### Usage

```bash
conda activate tf_gpu
jupyter lab
# Open and run the desired notebook or script
```

---

## Environment Setup

### Requirements

- **Python** 3.8+
- **TensorFlow** 2.x with GPU support (CUDA)
- **BlenderProc** (for dataset generation)
- **Blender** 3.x+ (called by BlenderProc)
- **Conda** (environment management)

### Key Dependencies

```
tensorflow
keras
numpy
pandas
Pillow
imageio
blenderproc
```

### Quick Start

```bash
# Activate the environment
conda activate tf_gpu

# Launch JupyterLab
jupyter lab
```

---

## Results

### Prediction Results

Sample of the model's attitude predictions on test images (softmax output):

![Prediction Results](softmax_30_sar_prediction.png)

### Error Analysis

Quaternion angular error distribution across the test set:

![Error Analysis](error_analysis.png)

### Additional Plots

| File | Description |
|---|---|
| `softmax_55_sar_prediction.png` | Additional softmax prediction results |

---

## References

- Sharma, S., Beier, C., D'Amico, S. (2019). *Spacecraft Pose Estimation Dataset (SPEED)*. [[Paper]](https://arxiv.org/abs/1907.04195)
- Proença, J., Gao, Y. (2020). *Deep Learning for Spacecraft Pose Estimation from Photorealistic Rendering*. [[Paper]](https://arxiv.org/abs/2004.05076)

---

## License

This project is part of a bachelor thesis. All rights reserved.
