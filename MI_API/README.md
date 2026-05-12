# SpiTranNet‑LIF  
**Inference & Deployment Framework for EEG Motor Imagery Classification**

---

## 📌 Overview

This repository provides a **production‑oriented inference and deployment framework** for **SpiTranNet‑LIF**, a hybrid **Spiking Neural Network–Transformer (SNN–Transformer)** architecture designed for **EEG‑based Motor Imagery (MI) classification** in Brain–Computer Interface (BCI) systems.

The core model is based on the research paper:

> **SpiTranNet‑LIF: A Spiking Neural Network–Transformer Framework for Efficient Motor Imagery Decoding**  
> *Maryam Khoshkhooy Titkanlou, Alireza Hashemi, Roman Mouček*  
> ICAART 2026  
> DOI: `10.5220/0014468700004052`  
> License: **CC BY‑NC‑ND 4.0**

⚠️ **Important note**  
This repository **does NOT contain the full experimental training pipeline or ablation studies from the paper**.  
Instead, it focuses on:

- ✅ Model loading and inference
- ✅ API‑based prediction services
- ✅ Interactive web dashboard (Streamlit)
- ✅ Practical deployment for offline / real‑time BCI usage

---

## 🧠 Scientific Background

Motor Imagery (MI) classification is a central problem in EEG‑based BCI systems. Traditional deep learning models (CNNs, LSTMs, Transformers) offer high accuracy but suffer from **dense computation, energy inefficiency, and limited temporal sparsity**, restricting their usability in real‑time and embedded BCI environments.

**SpiTranNet‑LIF** addresses these limitations by integrating:

- **Adaptive Leaky Integrate‑and‑Fire (LIF) neurons**
- **Spiking Multi‑Head Attention (SMHA)**
- **Transformer‑based global contextual modeling**

This hybrid design enables:

- Event‑driven, temporally sparse computation
- Biologically inspired neural dynamics
- Competitive accuracy with significantly reduced inference cost

The model was evaluated on the **BCI Competition IV‑2a dataset** and achieved:

- **86.3% mean accuracy** (binary MI: left‑hand vs right‑hand)
- **Mean ROC‑AUC ≈ 0.91**
- **~594K parameters**
- **Sub‑second inference time**

---

## 🎯 Purpose of This Repository

This project is intended for:

- ✅ **Model inference and deployment**
- ✅ **Offline and local execution (no internet required)**
- ✅ **Real‑time or near real‑time MI decoding**
- ✅ **Demonstration dashboards and APIs**

It is **not**:
- ❌ A full reproduction of paper experiments
- ❌ A training framework for cross‑subject optimization
- ❌ A hyperparameter search or benchmarking suite

---

## 🗂️ Repository Structure

MI_API/
│
├── app_st.py            # Streamlit web dashboard (UI)
├── main.py              # FastAPI inference backend
├── run.py               # Unified launcher (API + Streamlit)
│
├── models/
│   └── *.pt             # Pretrained SpiTranNet-LIF model weights
│
├── notebooks/
│   └── final_model.ipynb  # Final modeling & validation notebook
│
├── utils/
│   ├── preprocessing.py
│   ├── inference.py
│   └── model_loader.py
│
├── requirements.txt
└── README.md


---

## 🧩 System Architecture

┌─────────────┐        HTTP        ┌──────────────┐
│ Streamlit   │  ─────────────▶  │ FastAPI       │
│ Frontend    │                  │ Inference API │
│ (UI)        │                  │ (SpiTranNet)  │
└─────────────┘                  └──────────────┘
       │                                  │
       │                                  ▼
       └────────────── EEG Data ─────── Model


- **FastAPI** handles model inference and prediction logic
- **Streamlit** provides an interactive visualization interface
- Both services run **locally** and communicate via `localhost`

---

## 🚀 Getting Started

### 1️⃣ Environment Setup

```bash
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux / macOS

pip install -r requirements.txt
```

---

### 2️⃣ Run the System (Recommended)

Use the unified launcher:

```bash
python run.py
```

This automatically starts:

- FastAPI → `http://localhost:8080`
- Streamlit → `http://localhost:8501`

---

### 3️⃣ Manual Execution (Optional)

```bash
# Terminal 1
python main.py

# Terminal 2
streamlit run app_st.py
```

---

## 🔌 API Endpoints (FastAPI)

| Endpoint | Method | Description |
|--------|--------|------------|
| `/dummy_predict_all` | GET | Test endpoint with dummy EEG data |
| `/predict/{subject_id}` | POST | Subject‑specific MI prediction |
| `/predict_batch/{subject_id}` | POST | Batch inference |

Swagger UI:
http://localhost:8080/docs


---

## 🖥️ Streamlit Dashboard

The Streamlit UI provides:

- EEG sample visualization
- Real‑time prediction display
- Subject‑wise inference
- Debug / testing modes with dummy data

Access at:
http://localhost:8501


---

## 📊 Dataset Reference

This project uses models trained on:

**BCI Competition IV‑2a Dataset**  
- 9 subjects  
- 4 MI classes (paper)  
- Binary classification (this repo: LH vs RH)  
- Sampling rate: 250 Hz  
- Preprocessing:  
  - Band‑pass filtering (8–30 Hz)  
  - EMA standardization  
  - Common Average Referencing (CAR)

Dataset source:
> Graz University of Technology – Laboratory of Brain‑Computer Interfaces

---

## 📚 Citation

If you use this project or the SpiTranNet‑LIF architecture in your research, please cite:

```bibtex
@inproceedings{titkanlou2026spitrannet,
  title={SpiTranNet-LIF: A Spiking Neural Network–Transformer Framework for Efficient Motor Imagery Decoding},
  author={Titkanlou, Maryam Khoshkhooy and Hashemi, Alireza and Mouček, Roman},
  booktitle={Proceedings of the 18th International Conference on Agents and Artificial Intelligence (ICAART)},
  year={2026},
  pages={3710--3718},
  publisher={SCITEPRESS}
}
```

---

## ⚖️ License & Disclaimer

- **Paper**: CC BY‑NC‑ND 4.0  
- **Code in this repository**: Provided for **research and educational purposes**

⚠️ This implementation is an **engineering and deployment interpretation** of the published model, not the official experimental codebase.

---

## 🔮 Future Work

- Multi‑class MI decoding
- Online adaptive inference
- Cross‑subject generalization
- Embedded / neuromorphic deployment
- Dockerized CPU/GPU pipelines

---

## 📬 Contact

For academic inquiries related to the original model, please refer to the **paper authors**.  
For questions about this deployment and inference framework, open an issue in this repository.

