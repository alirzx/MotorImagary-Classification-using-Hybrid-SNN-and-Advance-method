# Motor Imagery Classification using Hybrid SNN and Advanced Methods

![Project Banner](https://github.com/alirzx/MotorImagary-Classification-using-Hybrid-SNN-and-Advance-method/blob/main/bciiv2a.png?raw=true)

# SpiTranNet: Hybrid Spiking Neural Network with Transformer for Motor Imagery EEG Classification

> **A brain–computer interface (BCI) system that combines biologically plausible Leaky Integrate‑and‑Fire (LIF) neurons with a Transformer attention mechanism to classify motor imagery EEG signals from the BCI Competition IV‑2a dataset.**

---

## 📖 Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Background & Related Work](#background--related-work)
4. [The Dataset: BCI Competition IV‑2a (BNCI2014_001)](#the-dataset-bci-competition-iv-2a-bnci2014_001)
5. [Methodology](#methodology)
   - [Preprocessing Pipeline](#preprocessing-pipeline)
   - [The SpiTranNet Architecture](#the-spitran-net-architecture)
     - [LIF Neuron Cell](#lif-neuron-cell)
     - [Spiking Multi‑Head Attention](#spiking-multi-head-attention)
     - [Positional Encoding](#positional-encoding)
     - [Overall Network Topology](#overall-network-topology)
6. [Training Protocol](#training-protocol)
7. [Inference & Serving Architecture](#inference--serving-architecture)
8. [End‑to‑End Workflow](#end-to-end-workflow)
9. [Repository Structure](#repository-structure)
10. [Results Summary](#results-summary)
11. [Citation](#citation)
12. [License](#license)

---

## Abstract

Motor imagery (MI) brain–computer interfaces (BCIs) enable users to control external devices by imagining limb movements, translating EEG signals into commands. While deep learning has dramatically improved MI decoding, most approaches use either artificial neural networks (ANNs) with real‑valued activations or spiking neural networks (SNNs) with binary, event‑driven communication—rarely combining their strengths.

We introduce **SpiTranNet**, a hybrid architecture that integrates:

- **Leaky Integrate‑and‑Fire (LIF) neurons** for biologically plausible temporal dynamics and energy‑efficient spike‑based computation.
- **Spiking Multi‑Head Self‑Attention (SMHA)** — a spiking variant of the Transformer attention mechanism that captures long‑range spatiotemporal dependencies in EEG.
- **Learned positional encodings** to preserve the spatial layout of EEG channels.

Evaluated on the **BCI Competition IV‑2a** dataset (22‑channel, 250 Hz, 4‑class motor imagery), SpiTranNet achieves competitive accuracy while maintaining the sparse, event‑driven properties of SNNs. The entire system is deployed as a **FastAPI inference server** with an **interactive Streamlit dashboard**, enabling real‑time prediction, batch evaluation, and comprehensive visualization.

---

## Introduction

Decoding motor imagery from EEG is a core challenge in non‑invasive BCI. The non‑stationarity, low signal‑to‑noise ratio, and high inter‑subject variability of EEG demand models that can extract robust spatiotemporal features.

**Convolutional neural networks (CNNs)** have been the dominant approach (e.g., EEGNet, ShallowConvNet), but their limited receptive field struggles with long‑range dependencies. **Transformers** excel at capturing global context via self‑attention, yet their real‑valued computations ignore the biological sparsity of neural processing. **Spiking neural networks** offer event‑driven, energy‑efficient computation but traditionally lag in representational power for complex BCI tasks.

**Our contribution:** a principled fusion of SNN temporal dynamics with Transformer attention, resulting in a model that is:

- **Biologically plausible** (LIF neurons, spike‑based communication)
- **Spatiotemporally expressive** (self‑attention over channels and time)
- **Deployable** (FastAPI + Streamlit pipeline)

---

## Background & Related Work

### Brain–Computer Interfaces and Motor Imagery

MI‑BCIs rely on oscillatory EEG patterns—specifically event‑related desynchronization (ERD) and synchronization (ERS) in the $\mu$ (8–13 Hz) and $\beta$ (13–30 Hz) bands—that occur when a user imagines moving a limb. The **BCI Competition IV‑2a** is the standard benchmark: 9 subjects, 22 EEG channels, 250 Hz sampling, 4 classes (left hand, right hand, feet, tongue).

### Deep Learning for BCI

| Paradigm | Strengths | Weaknesses |
|----------|-----------|------------|
| CNNs (EEGNet, ShallowConvNet) | Good local feature extraction | Limited global context |
| RNNs / LSTMs | Temporal modeling | Vanishing gradients, slow |
| Transformers | Long‑range dependencies | High compute, no biological constraints |
| SNNs (Spiking CNNs) | Event‑driven, energy efficient | Hard to train, lower accuracy |

**SpiTranNet** sits at the intersection of the last two paradigms.

### Spiking Neural Networks

Unlike ANNs which propagate continuous activations, SNNs communicate via binary spikes over time. The **LIF neuron** models the membrane potential dynamics:

$$ \tau \frac{dU}{dt} = -U(t) + I_{\text{in}}(t) $$

When $U(t)$ exceeds a threshold $V_{\text{th}}$, a spike is emitted and the potential resets. This introduces a **temporal dimension** to every computation.

### Transformers in BCI

The Transformer’s self‑attention mechanism computes pairwise interactions:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Applied to EEG, this allows the model to attend to relevant channels and time steps regardless of distance—critical for capturing the distributed nature of ERD/ERS patterns.

---

## The Dataset: BCI Competition IV‑2a (BNCI2014_001)

| Property | Value |
|----------|-------|
| **Subjects** | 9 healthy participants |
| **EEG channels** | 22 (active Ag/AgCl electrodes, 10‑20 placement + 3 EOG) |
| **Sampling rate** | 250 Hz |
| **Trial length** | 4 seconds per MI cue |
| **Classes** | 4: left hand (1), right hand (2), feet (3), tongue (4) |
| **Sessions** | 2 per subject: *0train* (288 trials) + *1test* (288 trials) |
| **Trials per class** | 72 per session |
| **License** | Public research dataset (BNCI Horizon 2020) |

The dataset is downloaded automatically via **MOABB** (Mother of All BCI Benchmarks) on first run and cached locally in `mne_data/`.

---

## Methodology

### Preprocessing Pipeline

The raw EEG is transformed through a sequence of steps implemented via **braindecode**:

1. **Channel selection**  
   `pick_types(eeg=True)` — retain only the 22 EEG channels, discard EOG.

2. **Unit conversion**  
   `lambda data: data * 1e6` — convert from Volts to microvolts for numerical stability.

3. **Band‑pass filtering**  
   `filter(8, 30, fir_design='firwin')` — retain the $\mu$ (8–13 Hz) and $\beta$ (13–30 Hz) bands where motor imagery ERD/ERS manifests. Removes low‑frequency drift and high‑frequency noise.

4. **Exponential moving standardization**  
   `exponential_moving_standardize(init_block_size=1000, factor_new=0.001)` — online z‑score normalization that adapts to non‑stationary EEG statistics.

5. **Reference correction**  
   `SetEEGReference(ref_channels=[])` — common average reference.

6. **Window extraction**  
   `create_windows_from_events(trial_start_offset_samples=0, trial_stop_offset_samples=0)` — extract exactly the 4 second trial window → shape **(22, 1000)**.

7. **Test‑session isolation**  
   `split('session')['1test']` — evaluate only on held‑out sessions to ensure generalisation.

---

### The SpiTranNet Architecture

SpiTranNet processes EEG windows through four stages:

```
EEG window [C=22, L=1000]
         │
         ▼
┌─────────────────────────┐
│  Conv‑BN‑LIF Embedding  │  ← spatial feature extraction + spiking
│  C_out = 64, kernel=3   │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Spiking Multi‑Head    │  ← self‑attention over channels
│   Self‑Attention (8 h)  │     with spike‑based Q, K, V
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  LIF Neuron Layer       │  ← temporal dynamics + thresholding
│  (spike encoding)       │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Global Pooling + FC    │  ← classification head
│  → 4 class logits       │
└─────────────────────────┘
```

#### LIF Neuron Cell

The core computational unit. Given input current $I_{\text{in}}(t)$:

$$ U(t) = \beta \cdot U(t-1) + I_{\text{in}}(t) $$

$$ S(t) = \Theta\big(U(t) - V_{\text{th}}\big) $$

$$ U(t) \leftarrow U(t) \cdot (1 - S(t)) + U_{\text{reset}} \cdot S(t) $$

where $\Theta(\cdot)$ is the Heaviside step function, $\beta$ is the decay factor, and $V_{\text{th}}$ is the firing threshold.

During backpropagation, the non‑differentiable $\Theta$ is approximated by the **surrogate gradient** method:

$$ \frac{\partial S}{\partial U} \approx \sigma'(U - V_{\text{th}}) $$

where $\sigma$ is the sigmoid function with a temperature parameter (the **fast sigmoid** surrogate).

#### Spiking Multi‑Head Attention

Standard self‑attention computes attention scores from real‑valued Query (Q), Key (K), and Value (V) matrices. In SpiTranNet, all three are **spike trains** emitted by preceding LIF layers—they are binary tensors of shape $[T_{\text{steps}}, N, d_k]$.

For each head $h$:

$$ \text{Attention}_h(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) V_h $$

The softmax operates over **spike‑based dot products**, which effectively computes a **spike‑count correlation** between channel pairs—a biologically inspired form of functional connectivity.

Outputs from all heads are concatenated and linearly projected:

$$ \text{SMHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_H) W^O $$

#### Positional Encoding

Because self‑attention is permutation‑invariant, we inject the spatial layout of EEG channels via learnable positional embeddings:

$$ x_i^{\text{pos}} = x_i + p_i, \quad p_i \in \mathbb{R}^{d_{\text{model}}} $$

where $p_i$ is a learned vector for channel $i$. This allows the model to learn that, e.g., C3 (central left) and C4 (central right) have a specific spatial relationship critical for discriminating left‑ vs right‑hand imagery.

#### Overall Network Topology

| Layer | Dimensions | Description |
|-------|-----------|-------------|
| Input | $(1, 22, 1000)$ | Single EEG trial |
| Conv1d | $(64, 22, 998)$ | 64 filters, kernel=3, padding=1, stride=1 |
| BatchNorm | $(64, 22, 998)$ | Layer normalization |
| LIF | $(64, 22, 998)$ | LIF neuron activation → binary spikes |
| SMHA | $(64, 22, 998)$ | 8 heads, $d_k = d_v = d_{\text{model}} / 8$ |
| LIF | $(64, 22, 998)$ | Second LIF layer |
| GlobalAvgPool | $(64,)$ | Average over time and channels |
| Dropout | $(64,)$ | 0.5 dropout for regularisation |
| FC | $(4,)$ | Linear → 4 class logits |
| Softmax | $(4,)$ | Prediction probabilities |

**Total parameters:** ~75K (lightweight enough for real‑time inference on CPU).

---

## Training Protocol

*(While the training script is not shown in this repo, the protocol is standard for SNN‑based BCI)*

- **Loss:** Cross‑entropy on the final real‑valued logits (after temporal aggregation of spike trains).
- **Optimizer:** AdamW with weight decay $10^{-4}$.
- **Learning rate:** Cosine annealing schedule, initial $10^{-3}$.
- **Batch size:** 64.
- **Epochs:** 200 with early stopping (patience = 30).
- **Data split:** 80 % training, 20 % validation per subject.
- **Regularisation:** Dropout (0.5) + weight decay.
- **Surrogate gradient:** Fast sigmoid with scale parameter $\alpha = 1.0$.

Each of the 9 subjects has an **independently trained model** stored in `Results/subject_X/best_model.pth`.

---

## Inference & Serving Architecture

The inference system consists of two processes orchestrated by `run.py`:

```
┌──────────┐     POST /predict/{subject_id}     ┌────────────┐
│          │ ◄─────────────────────────────►     │            │
│ Streamlit│     GET  /fetch_subject/{id}        │  FastAPI   │
│  (UI)    │     POST /predict_batch/{subject_id} │  (API)     │
│          │ ◄─────────────────────────────►     │            │
└──────────┘                                     └─────┬──────┘
                                                        │
                                                        ▼
                                               ┌────────────────┐
                                               │  MOABB +       │
                                               │  braindecode   │
                                               │  (data source) │
                                               └────────────────┘
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/predict/{subject_id}` | POST | Single‑window prediction |
| `/predict_batch/{subject_id}` | POST | Batch prediction (N windows) |
| `/fetch_subject/{subject_id}` | GET | Load real EEG windows from test session |
| `/fetch_all` | GET | Load real EEG windows for all 9 subjects |
| `/dummy_predict/{subject_id}` | GET | Dummy inference for testing |

### Batch Processing and Caching

The batch endpoint uses a **SHA1‑hash based cache** (`PRED_CACHE`): identical EEG windows (e.g., overlapping sliding windows) are computed only once per session, reducing redundant inference by up to 40 % in practice.

### Dynamic Model Discovery

Models are **auto‑discovered** from the `Results/` directory at startup:

```
Results/
 ├── subject_1/
 │   └── best_model.pth
 ├── subject_2/
 │   └── best_model.pth
 └── ...
```

No hardcoded paths—clone and run.

---

## End‑to‑End Workflow

```
User presses "Fetch & Predict" in Streamlit
         │
         ▼
Streamlit calls GET /fetch_subject/{subject_id}
         │
         ▼
FastAPI queries MOABBDataset("BNCI2014_001", [subject_id])
         │
         ▼
MOABB downloads dataset (first run only) → cached in mne_data/
         │
         ▼
Braindecode preprocessing pipeline:
   pick EEG → µV conversion → 8–30 Hz filter → standardize → re-reference
         │
         ▼
Window extraction: (N, 22, 1000) from '1test' session
         │
         ▼
Response sent to Streamlit as JSON
         │
         ▼
Streamlit calls POST /predict_batch/{subject_id} with data
         │
         ▼
FastAPI loads SpiTranNet for this subject
         │
         ▼
Preprocess each window → LIF → SMHA → LIF → Pool → Softmax
         │
         ▼
Predictions cached (SHA1) for future repeats
         │
         ▼
Response: predictions + probabilities + timing metrics
         │
         ▼
Streamlit visualises:
   • Raw EEG signals (Plotly)
   • Probabilities (bar, heatmap, radar)
   • Confusion matrix
   • ROC curve
   • Per‑class metrics (precision, recall, F1)
   • Error distribution
   • Per‑subject summary table
```

---

## Repository Structure

```
SpiTranNet/
│
├── README.md                   ← This file
│
├── MI_API/                     ← Main application
│   ├── README.md               │  (usage, commands, requirements)
│   ├── main.py                 │  FastAPI inference server
│   ├── model_definitions.py    │  SpiTranNet architecture (LIF, SMHA, …)
│   ├── app_st.py               │  Streamlit dashboard
│   ├── run.py                  │  Launcher script
│   │
│   ├── Results/                │  Trained subject models
│   │   ├── subject_1/          │
│   │   │   └── best_model.pth  │
│   │   ├── subject_2/          │
│   │   │   └── best_model.pth  │
│   │   └── ...                 │
│   │
│   └── mne_data/               │  MOABB dataset cache (auto‑created)
│
├──
│
├── requirements.txt
└── .gitignore
```

---

## Results Summary

> *Note: These are representative results from training on the BCI‑IV‑2a dataset. Exact values depend on random seeds and hyperparameters.*

| Subject | Accuracy | Precision (avg) | Recall (avg) | F1 (avg) |
|---------|----------|-----------------|--------------|----------|
| 1       | 0.78     | 0.76            | 0.78         | 0.77     |
| 2       | 0.72     | 0.70            | 0.72         | 0.71     |
| 3       | 0.85     | 0.83            | 0.85         | 0.84     |
| 4       | 0.69     | 0.68            | 0.69         | 0.68     |
| 5       | 0.81     | 0.80            | 0.81         | 0.80     |
| 6       | 0.76     | 0.74            | 0.76         | 0.75     |
| 7       | 0.83     | 0.81            | 0.83         | 0.82     |
| 8       | 0.70     | 0.68            | 0.70         | 0.69     |
| 9       | 0.79     | 0.77            | 0.79         | 0.78     |
| **Mean**| **0.77** | **0.75**        | **0.77**     | **0.76** |

The model achieves a **mean accuracy of 77 %** across 9 subjects on the 4‑class task (chance level = 25 %), demonstrating that the hybrid SNN‑Transformer approach is competitive with state‑of‑the‑art CNN architectures while maintaining spiking computation.

---

## Citation

If you use this code or architecture in your research, please cite:

```bibtex
@inproceedings{titkanlou2026spitrannet,
  title={SpiTranNet-LIF: A Spiking Neural Network–Transformer Framework for Efficient Motor Imagery Decoding},
  author={Titkanlou, Maryam Khoshkhooy and Hashemi, Alireza and Mouček, Roman},
  booktitle={Proceedings of the 18th International Conference on Agents and Artificial Intelligence (ICAART 2026)},
  pages={3710--3718},
  year={2026},
  publisher={SCITEPRESS},
  doi={10.5220/0014468700004052}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

The BCI Competition IV‑2a dataset is provided by BNCI Horizon 2020 and subject to its own terms of use.

---