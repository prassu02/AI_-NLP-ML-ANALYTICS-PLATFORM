# 🚀 AI NLP + ML Analytics Platform

An end-to-end **AI-powered analytics platform** built with Streamlit that supports **data analysis, NLP, AutoML, explainability, and report generation** — all in one application.

---

## 📌 Overview

This project is a **production-ready AI platform** that allows users to:

* Upload datasets (CSV, Excel, TXT)
* Perform automatic data cleaning
* Run NLP pipelines (TF-IDF, BERT sentiment, topic modeling)
* Build machine learning models automatically
* Visualize data interactively
* Generate model explainability using SHAP
* Download reports as PDF

---

## ⚙️ Tech Stack

* **Frontend/UI**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Visualization**: Plotly, Matplotlib
* **Machine Learning**: Scikit-learn
* **NLP**: TF-IDF, LDA, Transformers (BERT)
* **Explainability**: SHAP
* **Report Generation**: ReportLab

---

## 🚀 Features

### 📂 Data Upload

* Supports:

  * CSV
  * Excel (.xlsx)
  * TXT (auto delimiter detection)

### 🧹 Data Preprocessing

* Duplicate removal
* Missing value handling
* Automatic datatype detection
* Numeric conversion

### 🧠 NLP Capabilities

* Text cleaning
* TF-IDF vectorization
* Word frequency visualization
* BERT sentiment analysis (GPU/CPU)
* Topic modeling using LDA

### 📊 Interactive Dashboard

* Histogram
* Scatter Plot
* Bar Chart
* Pie Chart

### 🤖 AutoML

* Automatic model selection:

  * Classification → RandomForestClassifier
  * Regression → RandomForestRegressor
* Data scaling and encoding
* Model evaluation:

  * Accuracy (classification)
  * R² Score (regression)

### 🧠 Explainable AI

* SHAP summary plots for model interpretability

### 📦 Report Generation

* Download model results as PDF

---

## 📁 Project Structure

```
AI-NLP-ML-Analytics-Platform/
│── app.py
│── requirements.txt
│── README.md
```

---

## ⚡ Installation

```bash
git clone <your-repo-url>
cd AI-NLP-ML-Analytics-Platform
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 🌐 Deployment

Deploy easily using:

* Streamlit Community Cloud
* AWS / GCP / Azure (advanced)

Steps:

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy app

---

## ⚠️ Known Limitations

* GPU not available on free Streamlit Cloud (BERT runs on CPU)
* SHAP may be slow for large datasets
* Large datasets (>5000 rows) are sampled for performance

---

## 🔥 Future Enhancements

* Hyperparameter tuning (Optuna)
* Model comparison dashboard
* LLM chatbot integration
* REST API deployment
* Authentication system
* SaaS monetization features

---

## 👨‍💻 Author

**Prasanna Kumar**
AI & Data Science Enthusiast

* GitHub: https://github.com/prassu02
* LinkedIn: https://www.linkedin.com/in/k-prasanna-kumar

---

## ⭐ If You Like This Project

Give it a ⭐ on GitHub and share it!

---

## 📜 License

This project is licensed under the MIT License.
