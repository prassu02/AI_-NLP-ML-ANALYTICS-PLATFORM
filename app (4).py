# ======================================================
# 🚀 AI NLP + ML PLATFORM
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import optuna
import matplotlib.pyplot as plt
import re
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier

from transformers import pipeline

from prophet import Prophet
from stable_baselines3 import PPO
import gymnasium as gym

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from collections import Counter

# ======================================================
# GPU SETUP
# ======================================================

device = 0 if torch.cuda.is_available() else -1

if device == 0:
    st.success("🚀 GPU detected: Using CUDA for BERT")
else:
    st.warning("⚠️ GPU not found, using CPU")

# ======================================================
# CACHE BERT MODEL (VERY IMPORTANT)
# ======================================================

@st.cache_resource
def load_bert():
    return pipeline(
        "sentiment-analysis",
        device=device
    )

# ======================================================
# UI
# ======================================================

st.set_page_config(layout="wide")
st.title("🚀 AI Analytics Platform (GPU Enabled)")

# ======================================================
# FILE UPLOAD
# ======================================================

file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if file:

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    st.success("Dataset Loaded")
    st.dataframe(df.head())

# ======================================================
# CLEANING
# ======================================================

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

# ======================================================
# FEATURE ENGINEERING
# ======================================================

    num_cols = df.select_dtypes(include=np.number).columns

    for col in num_cols:
        df[f"{col}_square"] = df[col]**2
        df[f"{col}_log"] = np.log1p(np.abs(df[col]) + 1)

# ======================================================
# NLP + GPU BERT
# ======================================================

    text_cols = df.select_dtypes(include="object").columns

    if len(text_cols) > 0:

        st.subheader("🧠 Advanced NLP (GPU BERT)")

        text_col = st.selectbox("Text Column", text_cols)

        def clean_text(t):
            t = str(t).lower()
            t = re.sub(r'[^a-zA-Z ]', '', t)
            return t

        df["clean_text"] = df[text_col].apply(clean_text)

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=300)
        X_text = tfidf.fit_transform(df["clean_text"]).toarray()

        st.write("TF-IDF shape:", X_text.shape)

        # WORD FREQUENCY
        words = " ".join(df["clean_text"])
        freq = Counter(words.split())
        freq_df = pd.DataFrame(freq.items(), columns=["Word","Count"]).sort_values("Count",ascending=False)

        st.plotly_chart(px.bar(freq_df.head(20), x="Count", y="Word", orientation="h"))

        # ======================================================
        # 🚀 GPU BERT SENTIMENT
        # ======================================================

        st.markdown("### 🤖 BERT Sentiment (GPU Accelerated)")

        try:
            bert = load_bert()

            batch_size = 16
            texts = df["clean_text"].tolist()

            results = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                results.extend(bert(batch))

            st.write(results[:20])

        except Exception as e:
            st.warning(f"BERT failed: {e}")

        # ======================================================
        # TOPIC MODELING
        # ======================================================

        st.markdown("### 🧩 Topic Modeling")

        try:
            vec = CountVectorizer(stop_words="english")
            dtm = vec.fit_transform(df["clean_text"])

            lda = LatentDirichletAllocation(n_components=5)
            lda.fit(dtm)

            words = vec.get_feature_names_out()

            for i, topic in enumerate(lda.components_):
                top = [words[j] for j in topic.argsort()[-10:]]
                st.write(f"Topic {i+1}: {top}")

        except:
            st.warning("Topic modeling failed")

# ======================================================
# DASHBOARD
# ======================================================

    st.subheader("📊 Dashboard")

    chart = st.selectbox("Chart", ["Histogram","Scatter","Bar","Pie"])

    x = st.selectbox("X", df.columns)

    if chart == "Histogram":
        fig = px.histogram(df, x=x)
    elif chart == "Scatter":
        y = st.selectbox("Y", df.columns)
        fig = px.scatter(df, x=x, y=y)
    elif chart == "Bar":
        y = st.selectbox("Y", df.columns)
        fig = px.bar(df, x=x, y=y)
    else:
        fig = px.pie(df, names=x)

    st.plotly_chart(fig)

# ======================================================
# AUTOML
# ======================================================

    st.subheader("🤖 AutoML")

    target = st.selectbox("Target", df.columns)

    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]

    X = StandardScaler().fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    if y.dtype == "object":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        model = RandomForestClassifier()
        metric = "Accuracy"
    else:
        model = RandomForestRegressor()
        metric = "R2"

    model.fit(X_train,y_train)
    pred = model.predict(X_test)

    score = accuracy_score(y_test,pred) if metric=="Accuracy" else r2_score(y_test,pred)

    st.success(f"{metric}: {score}")

# ======================================================
# SHAP
# ======================================================

    st.subheader("🧠 Explainability")

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(plt.gcf())
    except:
        st.warning("SHAP failed")

# ======================================================
# PDF
# ======================================================

    st.subheader("📦 Report")

    if st.button("Generate PDF"):

        c = canvas.Canvas("report.pdf", pagesize=letter)

        c.drawString(100,750,"AI Report")
        c.drawString(100,720,f"Score: {score}")

        c.save()

        with open("report.pdf","rb") as f:
            st.download_button("Download", f, "report.pdf")

else:
    st.info("Upload dataset")
