# ======================================================
# 🚀 AI NLP + ML ANALYTICS PLATFORM (FULL UPGRADE)
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

from collections import Counter

# Optional NLP libs
try:
    from wordcloud import WordCloud
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
except:
    stop_words = set()

from reportlab.pdfgen import canvas

# ======================================================
# UI CONFIG
# ======================================================

st.set_page_config(layout="wide")
st.title("🚀 AI NLP + ML Analytics Platform")

# ======================================================
# FILE UPLOAD
# ======================================================

file = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx"])

if file:

    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

    st.success("Dataset Loaded")

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    st.write(df.describe())

    # ======================================================
    # DATA CLEANING
    # ======================================================

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # ======================================================
    # NLP PROCESSING
    # ======================================================

    text_cols = df.select_dtypes(include="object").columns
    X_text = None

    if len(text_cols) > 0:

        st.subheader("🧠 Advanced NLP Processing")

        text_col = st.selectbox("Select Text Column", text_cols)

        # ---------- CLEAN TEXT ----------
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z ]', '', text)
            words = text.split()
            words = [w for w in words if w not in stop_words]
            return " ".join(words)

        df["clean_text"] = df[text_col].apply(clean_text)

        st.write("Cleaned Text Sample")
        st.dataframe(df[["clean_text"]].head())

        # ---------- TF-IDF ----------
        tfidf = TfidfVectorizer(max_features=300, ngram_range=(1,2))
        X_text = tfidf.fit_transform(df["clean_text"]).toarray()

        st.write("TF-IDF Shape:", X_text.shape)

        # ---------- WORD FREQUENCY ----------
        st.markdown("### 🔤 Top Words")

        all_words = " ".join(df["clean_text"])
        word_freq = Counter(all_words.split())

        freq_df = pd.DataFrame(word_freq.items(),
                               columns=["Word","Count"]
                               ).sort_values("Count", ascending=False)

        st.dataframe(freq_df.head(20))

        fig = px.bar(freq_df.head(20),
                     x="Count",
                     y="Word",
                     orientation="h")

        st.plotly_chart(fig)

        # ---------- WORD CLOUD ----------
        st.markdown("### ☁ Word Cloud")

        try:
            wc = WordCloud(width=800, height=400).generate(all_words)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)
        except:
            st.info("WordCloud not installed")

        # ---------- SENTIMENT ----------
        st.markdown("### 😊 Sentiment Analysis")

        def sentiment(text):
            pos = ["good","great","excellent","love","happy"]
            neg = ["bad","worst","hate","poor","sad"]

            score = 0
            for w in text.split():
                if w in pos:
                    score += 1
                elif w in neg:
                    score -= 1

            return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

        df["sentiment"] = df["clean_text"].apply(sentiment)

        st.dataframe(df[["clean_text","sentiment"]].head())

        st.plotly_chart(px.pie(df, names="sentiment"))

    # ======================================================
    # DASHBOARD
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
        pie_data = df[x].value_counts().reset_index()
        pie_data.columns = [x, "count"]
        fig = px.pie(pie_data, names=x, values="count")

    st.plotly_chart(fig)

    # ======================================================
    # AUTOML
    # ======================================================

    st.subheader("🤖 AutoML")

    target = st.selectbox("Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Combine NLP + tabular
    if X_text is not None:
        X_tab = pd.get_dummies(X.drop(columns=[text_col], errors="ignore"))
        X = np.hstack([X_tab, X_text])
    else:
        X = pd.get_dummies(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

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
        st.success(f"Accuracy: {score}")

    else:

        model = RandomForestRegressor()

        model.fit(X_train,y_train)
        pred = model.predict(X_test)

        score = r2_score(y_test,pred)
        st.success(f"R2 Score: {score}")

    # ======================================================
    # EXPLAINABLE AI
    # ======================================================

    st.subheader("🧠 Explainable AI")

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        shap.summary_plot(shap_values, X_test, show=False)
        st.pyplot(plt.gcf())

    except Exception as e:
        st.warning(f"SHAP failed: {e}")

    # ======================================================
    # PDF REPORT
    # ======================================================

    st.subheader("📦 Generate Report")

    if st.button("Create PDF"):

        c = canvas.Canvas("report.pdf")

        c.drawString(100,750,"AI Analytics Report")
        c.drawString(100,720,f"Rows: {df.shape[0]}")
        c.drawString(100,700,f"Columns: {df.shape[1]}")
        c.drawString(100,680,f"Score: {score}")

        c.save()

        with open("report.pdf","rb") as f:
            st.download_button("Download Report", f, "report.pdf")

else:
    st.info("Upload dataset to start analysis")
