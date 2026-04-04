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

from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from collections import Counter

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(page_title="AI Platform", layout="wide")
st.title("🚀 AI NLP Analytics Platform (GPU Enabled)")

# ======================================================
# GPU SETUP
# ======================================================

device = 0 if torch.cuda.is_available() else -1

if device == 0:
    st.success("🚀 GPU detected")
else:
    st.warning("⚠️ Running on CPU")

# ======================================================
# CACHE BERT
# ======================================================

@st.cache_resource
def load_bert():
    return pipeline("sentiment-analysis", device=device)

# ======================================================
# FILE UPLOAD
# ======================================================

file = st.file_uploader("Upload CSV or Excel or Text", type=["csv", "xlsx", "txt"])

if file:

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file) else pd.read_txt(file)

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
        df[f"{col}_square"] = df[col] ** 2
        df[f"{col}_log"] = np.log1p(np.abs(df[col]))

    # ======================================================
    # NLP SECTION
    # ======================================================

    text_cols = df.select_dtypes(include="object").columns

    if len(text_cols) > 0:

        st.subheader("🧠 NLP + BERT")

        text_col = st.selectbox("Select Text Column", text_cols)

        def clean_text(t):
            t = str(t).lower()
            return re.sub(r'[^a-zA-Z ]', '', t)

        df["clean_text"] = df[text_col].apply(clean_text)

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=300)
        X_text = tfidf.fit_transform(df["clean_text"]).toarray()

        st.write("TF-IDF Shape:", X_text.shape)

        # WORD FREQUENCY
        words = " ".join(df["clean_text"])
        freq = Counter(words.split())
        freq_df = pd.DataFrame(freq.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False)

        st.plotly_chart(px.bar(freq_df.head(20), x="Count", y="Word", orientation="h"))

        # BERT
        st.markdown("### 🤖 Sentiment Analysis")

        try:
            bert = load_bert()

            results = bert(df["clean_text"].tolist()[:50])
            st.write(results)

        except Exception as e:
            st.warning(f"BERT error: {e}")

        # LDA
        st.markdown("### 🧩 Topic Modeling")

        try:
            vec = CountVectorizer(stop_words="english")
            dtm = vec.fit_transform(df["clean_text"])

            lda = LatentDirichletAllocation(n_components=5)
            lda.fit(dtm)

            words = vec.get_feature_names_out()

            for i, topic in enumerate(lda.components_):
                top_words = [words[j] for j in topic.argsort()[-10:]]
                st.write(f"Topic {i+1}: {top_words}")

        except:
            st.warning("LDA failed")

    # ======================================================
    # DASHBOARD
    # ======================================================

    st.subheader("📊 Dashboard")

    chart = st.selectbox("Chart Type", ["Histogram", "Scatter", "Bar", "Pie"])
    x = st.selectbox("X-axis", df.columns)

    if chart == "Histogram":
        fig = px.histogram(df, x=x)
    elif chart == "Scatter":
        y = st.selectbox("Y-axis", df.columns)
        fig = px.scatter(df, x=x, y=y)
    elif chart == "Bar":
        y = st.selectbox("Y-axis", df.columns)
        fig = px.bar(df, x=x, y=y)
    else:
        fig = px.pie(df, names=x)

    st.plotly_chart(fig)

    # ======================================================
    # AUTOML
    # ======================================================

    st.subheader("🤖 AutoML")

    target = st.selectbox("Select Target", df.columns)

    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if y.dtype == "object":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        model = RandomForestClassifier()
        metric = "Accuracy"
    else:
        model = RandomForestRegressor()
        metric = "R2"

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    score = accuracy_score(y_test, pred) if metric == "Accuracy" else r2_score(y_test, pred)

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
    # PDF REPORT
    # ======================================================

    st.subheader("📦 Generate Report")

    if st.button("Generate PDF"):
        c = canvas.Canvas("report.pdf", pagesize=letter)

        c.drawString(100, 750, "AI Model Report")
        c.drawString(100, 720, f"{metric}: {score}")

        c.save()

        with open("report.pdf", "rb") as f:
            st.download_button("Download Report", f, "report.pdf")

else:
    st.info("Upload dataset to start")
