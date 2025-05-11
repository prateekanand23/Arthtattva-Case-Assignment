import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

st.set_page_config(layout="wide")

st.title("🔍 LAPD Crime Data Prediction Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your filtered_dataset.csv", type="csv")

# Upload EDA visualizations
eda_files = st.file_uploader("Upload EDA Visualizations (Select multiple)", type=["png", "jpg"], accept_multiple_files=True)

# Upload Predicted visualizations
pred_files = st.file_uploader("Upload Predicted Visualizations (Select multiple)", type=["png", "jpg"], accept_multiple_files=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # --- Data Preprocessing ---
    df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], errors='coerce')
    df['YEAR'] = df['DATE_OCC'].dt.year
    df['MONTH'] = df['DATE_OCC'].dt.month
    df['DAY'] = df['DATE_OCC'].dt.day
    df['HOUR'] = df['TIME_OCC'] // 100
    df['MINUTE'] = df['TIME_OCC'] % 100

    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'

    df['TIME_OF_DAY'] = df['HOUR'].apply(get_time_of_day)
    df = df[df['WEAPON_DESC'] != "Nan"]

    def weapon_used(weapon_desc):
        if pd.isna(weapon_desc) or weapon_desc is None:
            return 0
        weapon_desc = str(weapon_desc).upper()
        if any(term in weapon_desc for term in ['STRONG-ARM', 'VERBAL THREAT', 'UNKNOWN WEAPON', 'NONE', 'PHYSICAL PRESENCE']):
            return 0
        else:
            return 1

    df['WEAPON_USED_FLAG'] = df['WEAPON_DESC'].apply(weapon_used)

    features = ['AREA_NAME', 'TIME_OF_DAY', 'HOUR', 'VICT_AGE', 'VICT_SEX',
                'PREMIS_DESC', 'MONTH', 'DAY', 'CRM_CD_DESC']
    target = 'WEAPON_USED_FLAG'

    for col in ['AREA_NAME', 'VICT_SEX', 'PREMIS_DESC', 'CRM_CD_DESC', 'TIME_OF_DAY']:
        df[col] = df[col].fillna('Unknown')
    for col in ['HOUR', 'VICT_AGE', 'MONTH', 'DAY']:
        df[col] = df[col].fillna(df[col].median())

    categorical_features = ['AREA_NAME', 'TIME_OF_DAY', 'VICT_SEX', 'PREMIS_DESC', 'CRM_CD_DESC']
    numerical_features = ['HOUR', 'VICT_AGE', 'MONTH', 'DAY']

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numerical_features),
        ('cat', cat_pipe, categorical_features)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    model.fit(X_train, y_train)
    df['predicted_probability'] = model.predict_proba(X[features])[:, 1]

    st.success("✅ Model trained and predictions made.")
    import streamlit as st
import numpy as np
import pandas as pd

# Example: Assuming you've already loaded the trained model and the input features into `X_test`
# predicted_probs = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1 (weapon used)

st.sidebar.header("Prediction Settings")
threshold = st.sidebar.slider("Probability Threshold", 0.0, 1.0, 0.5, 0.01)

# Convert probabilities to binary predictions using the threshold
predicted_labels = (predicted_probs >= threshold).astype(int)

# Display some results
st.write(f"Threshold: {threshold}")
st.write("Example predictions:")
st.write(pd.DataFrame({
    "Predicted Probability": predicted_probs[:10],
    "Predicted Label": predicted_labels[:10]
}))


    # ---------- Show EDA Visualizations ----------
    if eda_files:
        st.header("📊 EDA Visualizations")
        for file in eda_files:
            st.image(file, use_container_width =True, caption=file.name)

    # ---------- Show Predicted Visualizations ----------
    if pred_files:
        st.header("📈 Predicted Visualizations")
        for file in pred_files:
            st.image(file, use_container_width =True, caption=file.name)
else:
    st.info("Please upload the dataset to begin.")
