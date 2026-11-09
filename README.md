<div align="center">

# BI-RADS: Classification & Feature Recognition

**An EfficientNet-based approach to breast imaging classification and feature identification**

Code from the [6th DIGIX Global AI Challenge: BI-RADS](https://www.saikr.com/vse/2024/DIGIX)  
**Ranked 23rd out of 240**

6-class classification &nbsp; · &nbsp; 4 feature models &nbsp; · &nbsp; PyTorch

[Overview](#overview) · [Repository](#repository) · [Setup](#setup) · [Training](#training) · [Inference](#inference)

</div>

---

## Overview

This repository shares the training and inference code developed for the BI-RADS challenge. It uses EfficientNetV2-S models for two related tasks:

| Task | Approach | Prediction |
| --- | --- | --- |
| Classification | One six-class model | A class index for each image |
| Feature identification | Four binary expert models | Boundary, calcification, direction, and shape labels |

Images are resized to 224 × 224 pixels and normalized before inference. Predictions are written to CSV files.

## Repository

```text
.
├── README.md          Project overview and usage
├── requirements.txt   Python dependencies
├── train_cla.py       Six-class model training
├── train_fea.py       Binary feature model training
└── run.py             Classification and feature inference
```

> **Availability:** This repository does not currently include the `data` and `models` modules imported by the training scripts, datasets, or trained weight files. These must be supplied separately before the corresponding workflows can run.

## Setup

Install the dependencies in a suitable Python environment:

```bash
pip install -r requirements.txt
```

### Data

Prepare image datasets and labels compatible with the project's data loaders. The original README referenced the [Mammographic Mass Data Set](https://www.kaggle.com/datasets/jimitdand/mammographic-mass-data-set-for-breast-cancer) as a public resource; check whether it provides suitable images and labels before using it with this image-based pipeline. No dataset is included here.

## Training

The training scripts contain environment-specific dataset paths. Supply the missing `data` and `models` modules and update `DATASET_PATH` and the dataset settings in each script before running them.

### 1. Train the classification model

```bash
python train_cla.py
```

### 2. Train the feature models

```bash
python train_fea.py
```

The inference workflow expects four separately trained binary models, one for each feature. Adjust the dataset and checkpoint settings for each feature when training.

## Inference

Supply the trained weights and image folders at the paths expected by `run.py`, or edit those paths in the script:

```text
model/
├── PAGE_2500_V2S_demoFinal.pt   Classification model
├── B5_925.pt                    Boundary model
├── C6_918.pt                    Calcification model
├── D8_939.pt                    Direction model
└── S4_930.pt                    Shape model

testB/
├── cla/                         Images for classification
└── fea/                         Images for feature prediction
```

Run prediction from the repository root:

```bash
python run.py
```

| Output | Contents |
| --- | --- |
| `cla_pre.csv` | Image name and predicted class index |
| `fea_pre.csv` | Image name and boundary, calcification, direction, and shape predictions |

The script reports an error if the classification weights cannot be loaded but continues running. Confirm that the weights loaded successfully before using its predictions.
