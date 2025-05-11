# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve

st.set_page_config(layout="wide")
st.title("🔎 LAPD Crime Data: Weapon Use Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload the LAPD dataset CSV (e.g., filtered_dataset.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded dataset with shape: {df.shape}")

    df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], errors='coerce')
    df['YEAR'] = df['DATE_OCC'].dt.year
    df['MONTH'] = df['DATE_OCC'].dt.month
    df['DAY'] = df['DATE_OCC'].dt.day
    df['HOUR'] = df['TIME_OCC'] // 100
    df['MINUTE'] = df['TIME_OCC'] % 100

    def get_time_of_day(hour):
        if 5 <= hour < 12: return 'Morning'
        elif 12 <= hour < 17: return 'Afternoon'
        elif 17 <= hour < 22: return 'Evening'
        else: return 'Night'

    df['TIME_OF_DAY'] = df['HOUR'].apply(get_time_of_day)
    df = df[df['WEAPON_DESC'] != "Nan"]

    def weapon_used(weapon_desc):
        if pd.isna(weapon_desc): return 0
        desc = str(weapon_desc).upper()
        return 0 if any(x in desc for x in ['STRONG-ARM', 'VERBAL THREAT', 'UNKNOWN WEAPON', 'NONE']) else 1

    df['WEAPON_USED_FLAG'] = df['WEAPON_DESC'].apply(weapon_used)

    features = ['AREA_NAME', 'TIME_OF_DAY', 'HOUR', 'VICT_AGE', 'VICT_SEX',
                'PREMIS_DESC', 'MONTH', 'DAY', 'CRM_CD_DESC']
    target = 'WEAPON_USED_FLAG'

    for col in ['AREA_NAME', 'VICT_SEX', 'PREMIS_DESC', 'CRM_CD_DESC', 'TIME_OF_DAY']:
        df[col] = df[col].fillna('Unknown')
    for col in ['HOUR', 'VICT_AGE', 'MONTH', 'DAY']:
        df[col] = df[col].fillna(df[col].median())

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    categorical_features = ['AREA_NAME', 'TIME_OF_DAY', 'VICT_SEX', 'PREMIS_DESC', 'CRM_CD_DESC']
    numerical_features = ['HOUR', 'VICT_AGE', 'MONTH', 'DAY']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    st.subheader("🔍 Model Evaluation: Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)

    fig1, ax1 = plt.subplots()
    ax1.plot(recall, precision, label=f'AP = {ap_score:.3f}')
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("📊 Calibration Curve")
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    fig2, ax2 = plt.subplots()
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.plot(prob_pred, prob_true, "o-")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("True Probability")
    ax2.set_title("Calibration Curve")
    st.pyplot(fig2)

    st.subheader("📈 Predicted Probability Histogram")
    fig3, ax3 = plt.subplots()
    sns.histplot(y_pred_proba, bins=50, kde=True, ax=ax3)
    ax3.axvline(0.5, color='red', linestyle='--')
    ax3.set_title("Distribution of Predicted Probabilities")
    st.pyplot(fig3)
