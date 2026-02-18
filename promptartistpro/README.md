# 🎨 Prompt Artist Pro

**Prompt Artist Pro** (formerly Prompting Techniques Engine) is a premium, high-performance expert workstation designed to bridge the gap between user intent and professional-grade AI prompting. 

Powered by **Pure ML/NLP** (no LLM calls required for classification), the engine instantly identifies your project needs across 11 distinct domains and retrieves optimized, role-based prompt templates from a massive library of 130+ expert blueprints.

---

## ✨ Key Features

### 🧠 Intent Classification (SVM Engine)
- **Zero-Latency Processing**: Uses a locally trained Support Vector Machine (LinearSVC) for instant intent detection.
- **High Accuracy**: Trained on a diverse dataset of 260+ samples covering technical, creative, and financial domains.
- **Privacy-First**: No external APIs or LLMs are used for classification—your queries stay local.

### 📚 Expert Library (133+ Templates)
Access professional prompt architectures for 11 domains:
- **UI/UX Design**: Design systems, accessibility audits, and prototypes.
- **Cybersecurity**: Threat modeling, SOC incident response, and pen-testing.
- **Data Science**: NLP pipelines, predictive modeling, and feature engineering.
- **Coding & Architecture**: Microservices, concurrent systems, and legacy refactors.
- **Finance & Data Recon**: M&A modeling, ERP reconciliation, and tax audits.
- **Marketing & Project Management**: GTM strategy, PPC optimization, and roadmap planning.
- **Creative Writing & General**: World-building, screenplay structure, and productivity hacks.

### 💎 Premium Experience
- **Glassmorphism UI**: A futuristic, translucent interface with neon glow accents.
- **Tabbed Dashboard**: Seamlessly switch between the **AI Interaction Chat** and the **Library Explorer**.
- **Interactive Fill Mode**: Click "Customize" on any template to dynamically fill `{placeholders}` with a **Live Preview** and one-click copy.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Aesthetics**: Custom Vanilla CSS (Glassmorphism & Neon Glow)
- **NLP Engine**: Scikit-Learn (SVM), NLTK, Joblib
- **Data Storage**: Local JSON (Library), CSV (Training Data)

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- Requirements installed:
  ```bash
  pip install streamlit scikit-learn nltk joblib numpy pandas
  ```

### 2. Model Training
If you want to update the classifier logic or domains:
```bash
python train_model.py
```

### 3. Launching the Engine
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
promptartistpro/
├── app.py                  # Main Streamlit Application (Glassmorphism UI)
├── prompt_dispatcher.py     # Template retrieval & placeholder logic
├── preprocessor.py          # NLTK-based text cleaning
├── train_model.py          # SVM training script
├── data/
│   ├── prompt_library.json  # 133+ Expert Templates
│   └── training_data.csv   # 266 Classification Samples
├── models/
│   └── intent_classifier.joblib # Trained SVM Pipeline
└── README.md               # You are here!
```

---

*Engineered for Precision by Sai Kishore Kodidasu.*
