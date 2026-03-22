# ======================================================
# 🚀 AI NLP + ML ANALYTICS PLATFORM
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import optuna
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score

# ML Models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# NLP + Topic
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Semi
from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier

# Advanced NLP
from transformers import pipeline

# RL + TS
from prophet import Prophet
from stable_baselines3 import PPO
import gymnasium as gym

# ChatGPT
from openai import OpenAI

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Optional
from collections import Counter

try:
    from wordcloud import WordCloud
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

# ======================================================
# UI
# ======================================================

st.set_page_config(layout="wide")
st.title("🚀 AI Analytics Platform")

# ======================================================
# 1 FILE UPLOAD
# ======================================================

file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if file:

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    st.success("Dataset Loaded")

    st.dataframe(df.head())
    st.dataframe(df.tail())

    st.write("Shape:", df.shape)
    st.write("Total Values:", df.size)
    st.write("Statistics:", df.describe())

# ======================================================
# 2 DATA CLEANING
# ======================================================

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

# ======================================================
# 3 FEATURE ENGINEERING
# ======================================================

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        df[f"{col}_square"] = df[col]**2
        df[f"{col}_log"] = np.log1p(np.abs(df[col]) + 1)

# ======================================================
# 🔥 ADVANCED NLP BLOCK (ADDED)
# ======================================================

    text_cols = df.select_dtypes(include="object").columns
    X_text = None

    if len(text_cols) > 0:

        st.subheader("🧠 Advanced NLP")

        text_col = st.selectbox("Text Column", text_cols)

        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z ]', '', text)
            words = text.split()
            words = [w for w in words if w not in stop_words]
            return " ".join(words)

        df["clean_text"] = df[text_col].apply(clean_text)

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,2))
        X_text = tfidf.fit_transform(df["clean_text"]).toarray()

        st.write("TF-IDF Shape:", X_text.shape)

        # Word Frequency
        words = " ".join(df["clean_text"])
        freq = Counter(words.split())
        freq_df = pd.DataFrame(freq.items(), columns=["Word","Count"]).sort_values("Count",ascending=False)

        st.plotly_chart(px.bar(freq_df.head(20), x="Count", y="Word", orientation="h"))

        # WordCloud
        try:
            wc = WordCloud().generate(words)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)
        except:
            pass

        # 🔥 BERT SENTIMENT
        st.markdown("### 🤖 BERT Sentiment")

        try:
            bert = pipeline("sentiment-analysis")
            sample = df["clean_text"].head(20).tolist()
            result = bert(sample)
            st.write(result)
        except:
            st.warning("BERT not supported in environment")

        # 🔥 TOPIC MODELING (LDA)
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
            st.warning("Topic modeling failed")

# ======================================================
# 4 DASHBOARD
# ======================================================

    st.subheader("📊 Dashboard")

    chart = st.selectbox("Chart Type",
        ["Histogram","Scatter","Box","Line","Bar","Pie"])

    x = st.selectbox("X Column", df.columns)

    if chart == "Histogram":
        fig = px.histogram(df, x=x)

    elif chart == "Scatter":
        y_col = st.selectbox("Y Column", df.columns)
        fig = px.scatter(df, x=x, y=y_col)

    elif chart == "Line":
        fig = px.line(df, y=x)

    elif chart == "Box":
        fig = px.box(df, y=x)

    elif chart == "Bar":
        y_col = st.selectbox("Y Column", df.columns)
        fig = px.bar(df, x=x, y=y_col)

    elif chart == "Pie":
        fig = px.pie(df, names=x)

    st.plotly_chart(fig)

# ======================================================
# 5 AUTOML
# ======================================================

    st.subheader("🤖 AutoML")

    target = st.selectbox("Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    if X_text is not None:
        X_tab = pd.get_dummies(X.drop(columns=[text_col], errors="ignore"))
        X = np.hstack([X_tab, X_text])
    else:
        X = pd.get_dummies(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    if y.dtype == "object" or len(np.unique(y)) < 20:
        task = "classification"
    else:
        task = "regression"

    if task == "classification":

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        model = RandomForestClassifier()

        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        score = accuracy_score(y_test,pred)

    else:

        model = RandomForestRegressor()

        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        score = r2_score(y_test,pred)

    st.success(f"Score: {score}")

# ======================================================
# 6 SHAP
# ======================================================

    st.subheader("🧠 Explainable AI")

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(plt.gcf())
    except:
        st.warning("SHAP failed")

# ======================================================
# 🔥 7 DATASET CHAT (ADDED)
# ======================================================

    st.subheader("💬 Dataset Chat AI")

    api_key = st.text_input("OpenAI API Key", type="password")
    question = st.text_input("Ask about dataset")

    if api_key and question:

        client = OpenAI(api_key=api_key)

        prompt = f"""
        Dataset columns: {list(df.columns)}
        Sample: {df.head().to_dict()}
        Question: {question}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}]
            )

            st.write(response.choices[0].message.content)

        except:
            st.warning("API Error")

# ======================================================
# 8 PDF REPORT
# ======================================================

    st.subheader("📦 Generate Report")

    if st.button("Create PDF"):

        c = canvas.Canvas("report.pdf", pagesize=letter)

        c.drawString(100,750,"AI Report")
        c.drawString(100,720,f"Rows: {df.shape[0]}")
        c.drawString(100,700,f"Columns: {df.shape[1]}")
        c.drawString(100,680,f"Score: {score}")

        c.save()

        with open("report.pdf","rb") as f:
            st.download_button("Download", f, "report.pdf")

else:
    st.info("Upload dataset to start")
