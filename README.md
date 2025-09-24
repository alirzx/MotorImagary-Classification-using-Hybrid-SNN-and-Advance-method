# Motor Imagery Classification using Hybrid SNN and Advanced Methods

![Project Banner](path_to_image_or_screenshot)

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Future Work](#future-work)
- [References](#references)

---

## Overview
This project implements a **Hybrid Spiking Neural Network (SNN) + Advanced ML pipeline** for **Motor Imagery (MI) classification** using EEG signals.  
The main goal is to accurately classify motor imagery tasks from multi-channel EEG recordings.

Key features:
- Spiking Transformer Network (SpiTranNet) for temporal-spatial EEG patterns.
- Sliding window and FBCSP feature extraction for robust representation.
- Subject-wise and cross-subject training options.
- GPU-enabled training for faster experimentation.

---

## Dataset
We use **BCI Competition IV 2a (BNCI2014_001)** dataset:
- **Subjects:** 9
- **Channels:** 22 EEG channels
- **Sampling rate:** 250 Hz
- **Tasks:** Motor imagery of left hand, right hand, both feet, tongue.

The dataset is preprocessed with:
- Bandpass filtering
- Sliding windows for temporal segmentation
- Label encoding for binary/multi-class classification

---

## Data Preparation
- Load raw EEG signals as `(trials, channels, samples)`.
- Apply **sliding windows** (configurable window length and step) to increase data augmentation.
- Normalize and standardize signals per channel.
- Split into **training** and **test** sets using stratified split or subject-wise partitioning.
- Construct **PyTorch DataLoaders** for model training.

---

## Model Architecture

### Spiking Neuron
- Custom spiking neuron cell with surrogate gradient for backpropagation.
- Models temporal dynamics of EEG signals.

### SpiTranNet
- **Conv1D layers** extract local temporal-spatial features.
- **Spiking Multi-Head Attention** captures global dependencies.
- **Feedforward network** for feature transformation.
- **Classifier** outputs predicted motor imagery class probabilities.





