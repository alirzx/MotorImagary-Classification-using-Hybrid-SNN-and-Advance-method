
# SpiTranNet-LIF (model infernce.)

**Inference & Deployment Framework for EEG Motor Imagery Classification**

---

## 📌 Overview

This repository provides a **lightweight deployment stack** for running EEG Motor Imagery (MI) inference using a pre-trained **SpiTranNet-LIF** model.

It includes:

* FastAPI backend for inference
* Streamlit dashboard for visualization
* Pretrained model integration
* Local real-time prediction pipeline

---

## 🗂️ Project Structure

```
MI_API/
│
├── app_st.py              # Streamlit dashboard (UI)
├── main.py               # FastAPI inference server
├── model_definitions.py  # Model architecture (SpiTranNet-LIF)
├── run.py               # Optional launcher
│
├── requirements.txt     # Dependencies
├── README.md            # Documentation
│
├── venv/                # Python environment (not tracked)
└── __pycache__/        # Cache files (auto-generated)
```

---

## 🚀 How to Run

### 1️⃣ Start FastAPI Backend

```bash
python main.py
```

API will run at:

```
http://localhost:8080
```

Swagger documentation:

```
http://localhost:8080/docs
```

---

### 2️⃣ Start Streamlit Dashboard

In a second terminal:

```bash
streamlit run app_st.py
```

UI will run at:

```
http://localhost:8501
```

---

### 3️⃣ (Optional) Run Both Together

```bash
python run.py
```

---

## 🔌 API Summary

| Endpoint                      | Method | Description                       |
| ----------------------------- | ------ | --------------------------------- |
| `/dummy_predict_all`          | GET    | Test inference with synthetic EEG |
| `/predict/{subject_id}`       | POST   | Single-subject prediction         |
| `/predict_batch/{subject_id}` | POST   | Batch EEG inference               |

---

## 🖥️ Dashboard Features (Streamlit)

* EEG signal visualization
* Real-time prediction results
* Subject-wise inference testing
* Batch evaluation view
* Model output inspection tools

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Notes

* Models are loaded locally (no external training required)
* Designed for **inference + visualization only**
* Works fully offline after setup
* Uses pre-trained SpiTranNet-LIF weights

---

## 🧪 Dataset Context

System is designed for EEG Motor Imagery data (BCI-style signals, 22-channel EEG, 250Hz sampling).
No dataset download is required at runtime if models are already trained.

---

## 📌 License

For research and educational use only.

