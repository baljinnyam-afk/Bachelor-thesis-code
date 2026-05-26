# Optimization of an Artificial Intelligence-Based Attitude Determination System for Small Satellites

**ХИЙМЭЛ ОЮУНД СУУРИЛСАН БАГА ОВРЫН ХИЙМЭЛ ДАГУУЛЫН ЧИГЛЭЛ ТОГТООХ СИСТЕМИЙН ОНОВЧЛОЛ**

Bachelor Thesis — Baljinnyam Orgilsaikhan

---

## Overview

This project implements a complete pipeline for estimating the attitude (orientation) of a spacecraft in Low Earth Orbit (LEO) using deep learning. The target spacecraft is **CloudSat**, a NASA Earth-observation satellite.

The project covers:

1. **Synthetic dataset generation** — Photorealistic rendered images using BlenderProc with physically accurate lighting.
2. **Model training** — UrsoNet architecture (ResNet50V2 backbone + Gaussian-binned Euler angle soft classification), trained in two phases.
3. **Live demo** — Browser-based inference demo using TensorFlow.js, deployed via GitHub Pages.

---

## Project Structure

```
.
├── CloudSat.glb                              # 3D model of the CloudSat spacecraft
├── Resnet_train_script_v3_ursonet.ipynb      # Main training notebook (UrsoNet)
├── dataset_generation_sat_v10_sun_moon.py     # BlenderProc sun+moon rendering variant
├── renumber_csv.py                           # CSV renumbering utility
├── renumber_images.py                        # Image renumbering utility
├── command_conda_env.txt                     # Conda environment setup
├── command_blender.txt                       # Blender commands reference
├── docs/                                     # GitHub Pages live demo
│   ├── index.html                            # Demo page
│   ├── app.js                                # Inference logic (TF.js)
│   ├── style.css                             # Styling
│   ├── demo_data.json                        # 12 test images with ground-truth labels
│   ├── images/                               # Test images for the demo
│   └── model_tfjs/                           # Converted TF.js model artifacts
│       ├── model.json                        # TF.js model graph
│       ├── group1-shard*.bin                 # Model weights
│       └── savedmodel/                       # Original Keras SavedModel export
├── softmax_30_sar_prediction.png             # Prediction visualization
├── softmax_55_sar_prediction.png             # Additional prediction results
├── error_analysis.png                        # Angular error distribution
├── training_dashboard.png                    # Training metrics dashboard
└── LICENSE
```

---

## Live Demo

A browser-based demo deployed via GitHub Pages. It loads the trained model (hosted on HuggingFace) and runs inference on test spacecraft images, displaying Euler angle predictions vs. ground truth.

### Features

- Loads TF.js model directly in the browser — no server required
- Displays Roll/Pitch/Yaw predictions alongside ground-truth values
- Shows per-angle error and overall angular error (geodesic distance)
- Visual softmax distribution debug output

### Model Hosting

The converted TF.js model is hosted on HuggingFace:

```
https://huggingface.co/amartuvshindsl/spacecraft-attitude-resnet50/resolve/main/model.json
```

---

## Model Training

The training notebook `Resnet_train_script_v3_ursonet.ipynb` implements the **UrsoNet** architecture (Proença & Gao, 2020) — orientation soft classification via Gaussian-binned Euler angles, built on a **ResNet50V2** backbone.

### Architecture

| Component | Details |
|---|---|
| Backbone | ResNet50V2 (ImageNet pre-trained, BatchNormalization frozen during fine-tuning) |
| Heads | 3 independent Dense(16, softmax) — one per Euler axis (Roll, Pitch, Yaw) |
| Output | 48 values (3 × 16 softmax bins), converted to quaternion via circular mean |
| Loss | Categorical cross-entropy per axis with Gaussian soft labels (σ = 22.5°) |
| Optimizer | Adam with learning rate scheduling and gradient clipping |
| Framework | TensorFlow 2.10 / Keras |

### Two-Phase Training

- **Phase 1 — Head warm-up** (30 epochs): Backbone frozen, only classification heads trained at lr=1e-3
- **Phase 2 — Full fine-tuning** (120 epochs): Backbone unfrozen with frozen BN, lr=1e-5 with early stopping

---

## Dataset Generation

Synthetic images are generated using [BlenderProc](https://github.com/DLR-RM/BlenderProc) with the Cycles renderer (GPU-accelerated via CUDA). The `dataset_generation_sat_v10_sun_moon.py` script is included in this repo as a reference.

### Key Features

- **Visible sun disc** — Emissive sphere at realistic angular diameter (0.53°), warm white (5778K)
- **Per-frame sun direction** — Shadows change every rendered frame for maximum diversity
- **Lens glare** — Compositor-driven streaks and fog glow for photorealistic bloom
- **Earth backgrounds** — Day/night/cloud textures at 2K and 8K resolution
- **CSV output** — Columns: `image_name, qw, qx, qy, qz, tx, ty, tz, sun_dx, sun_dy, sun_dz`

---

## Results

### Prediction Visualization

Sample softmax predictions with ground-truth comparison:

![Prediction Results](softmax_30_sar_prediction.png)

![Additional Predictions](softmax_55_sar_prediction.png)

### Error Analysis

Angular error distribution across the test set:

![Error Analysis](error_analysis.png)

### Training Dashboard

Training and validation metrics over both phases:

![Training Dashboard](training_dashboard.png)

---

## Environment Setup

### Requirements

- **Python** 3.8+
- **TensorFlow** 2.10 with GPU support (CUDA)
- **BlenderProc** (for dataset generation)
- **Blender** 3.x+ (called by BlenderProc)

### Key Dependencies

```
tensorflow==2.10
keras
numpy
pandas
Pillow
imageio
blenderproc
```

### Quick Start

```bash
conda activate tf_gpu
jupyter lab
# Open Resnet_train_script_v3_ursonet.ipynb
```

---

## References

- Sharma, S., Beier, C., D'Amico, S. (2019). *Spacecraft Pose Estimation Dataset (SPEED)*. [[Paper]](https://arxiv.org/abs/1907.04195)
- Proença, J., Gao, Y. (2020). *Deep Learning for Spacecraft Pose Estimation from Photorealistic Rendering*. [[Paper]](https://arxiv.org/abs/2004.05076)

---

## License

This project is part of a bachelor thesis. All rights reserved.
