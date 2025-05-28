# Import dependencies with error handling
import streamlit as st
import pandas as pd
import numpy as np
import os

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError as e:
    st.error(f"Plotting libraries not available: {e}")
    st.error("Please install: pip install matplotlib seaborn")
    PLOTTING_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.calibration import calibration_curve
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError as e:
    st.error(f"Scikit-learn not available: {e}")
    st.error("Please install: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

# Page layout
st.set_page_config(layout="wide", page_title="LAPD Weapon Use Predictions")

st.title("🔍 LAPD Crime Data - Weapon Use Prediction Dashboard")

# Check if all dependencies are available
if not (PLOTTING_AVAILABLE and SKLEARN_AVAILABLE):
    st.error("❌ Missing required dependencies. Please install them to continue.")
    st.code("""
# Install required packages:
pip install streamlit pandas numpy matplotlib seaborn scikit-learn

# Or use requirements.txt:
pip install -r requirements.txt
    """)
    st.stop()

# Upload dataset with size limit
st.info("📝 **File Size Limits:** CSV max 200MB, Images max 50MB total")
uploaded_file = st.file_uploader("📁 Upload filtered_dataset.csv", type="csv")
eda_files = st.file_uploader("📂 Upload EDA visualizations (PNG)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Check file size
    file_size = uploaded_file.size
    if file_size > 200 * 1024 * 1024:  # 200MB limit
        st.error(f"❌ File too large ({file_size/1024/1024:.1f}MB). Please use a file smaller than 200MB.")
        st.info("💡 Try sampling your data or filtering to recent years only")
        st.stop()
    
    try:
        # Load data with error handling
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        
        # Sample large datasets
        if len(df) > 100000:
            df = df.sample(n=50000, random_state=42)
            st.warning(f"⚠️ Dataset was large, showing analysis on {len(df)} sampled rows for better performance")
        
        # Show basic info about the dataset
        with st.expander("📋 Dataset Info"):
            st.write("**Column Names:**", list(df.columns))
            st.write("**Missing Values:**")
            st.write(df.isnull().sum())
            st.write("**Sample Data:**")
            st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"❌ Error loading CSV file: {str(e)}")
        st.stop()

    try:
        # Preprocessing with error handling
        st.subheader("🔄 Data Preprocessing")
        
        # Check if required columns exist
        required_columns = ['DATE_OCC', 'TIME_OCC', 'WEAPON_DESC', 'AREA_NAME', 'VICT_AGE', 'VICT_SEX', 'PREMIS_DESC', 'CRM_CD_DESC']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("Available columns: " + ", ".join(df.columns.tolist()))
            st.stop()
        
        # Data preprocessing
        df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], errors='coerce')
        df['YEAR'] = df['DATE_OCC'].dt.year
        df['MONTH'] = df['DATE_OCC'].dt.month
        df['DAY'] = df['DATE_OCC'].dt.day
        df['HOUR'] = df['TIME_OCC'] // 100
        df['MINUTE'] = df['TIME_OCC'] % 100

        def get_time_of_day(hour):
            if pd.isna(hour):
                return 'Unknown'
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'

        df['TIME_OF_DAY'] = df['HOUR'].apply(get_time_of_day)
        
        # Filter out NaN weapon descriptions
        initial_rows = len(df)
        df = df[df['WEAPON_DESC'] != "Nan"]
        df = df.dropna(subset=['WEAPON_DESC'])
        final_rows = len(df)
        
        st.info(f"Filtered out {initial_rows - final_rows} rows with missing weapon descriptions")

        def weapon_used(weapon_desc):
            if pd.isna(weapon_desc):
                return 0
            weapon_desc = str(weapon_desc).upper()
            if any(term in weapon_desc for term in ['STRONG-ARM', 'VERBAL THREAT', 'UNKNOWN WEAPON', 'NONE', 'PHYSICAL PRESENCE']):
                return 0
            else:
                return 1

        df['WEAPON_USED_FLAG'] = df['WEAPON_DESC'].apply(weapon_used)
        
        # Show weapon use distribution
        weapon_dist = df['WEAPON_USED_FLAG'].value_counts()
        st.write("**Weapon Use Distribution:**")
        st.write(f"- No Weapon: {weapon_dist.get(0, 0)} ({weapon_dist.get(0, 0)/len(df)*100:.1f}%)")
        st.write(f"- Weapon Used: {weapon_dist.get(1, 0)} ({weapon_dist.get(1, 0)/len(df)*100:.1f}%)")

        # Feature engineering
        features = ['AREA_NAME', 'TIME_OF_DAY', 'HOUR', 'VICT_AGE', 'VICT_SEX', 'PREMIS_DESC', 'MONTH', 'DAY', 'CRM_CD_DESC']
        target = 'WEAPON_USED_FLAG'

        # Handle missing values
        for col in ['AREA_NAME', 'VICT_SEX', 'PREMIS_DESC', 'CRM_CD_DESC', 'TIME_OF_DAY']:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')

        for col in ['HOUR', 'VICT_AGE', 'MONTH', 'DAY']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        categorical_features = ['AREA_NAME', 'TIME_OF_DAY', 'VICT_SEX', 'PREMIS_DESC', 'CRM_CD_DESC']
        numerical_features = ['HOUR', 'VICT_AGE', 'MONTH', 'DAY']

        # Prepare data for modeling
        X = df[features]
        y = df[target]
        
        if len(X) == 0:
            st.error("❌ No data available for modeling after preprocessing")
            st.stop()
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        st.success(f"✅ Data split - Training: {len(X_train)}, Testing: {len(X_test)}")

    except Exception as e:
        st.error(f"❌ Error during preprocessing: {str(e)}")
        st.stop()

    try:
        # Model training
        st.subheader("🤖 Model Training")
        
        with st.spinner("Training model..."):
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            numerical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            preprocessor = ColumnTransformer([
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42))
            ])
            model.fit(X_train, y_train)
            
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
            
        st.success("✅ Model trained successfully!")
        
        # Model performance metrics
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"**Model Accuracy:** {accuracy:.3f}")

    except Exception as e:
        st.error(f"❌ Error during model training: {str(e)}")
        st.stop()

    if PLOTTING_AVAILABLE:
        try:
            st.subheader("📊 Prediction Visualizations")

            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC Curve
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, color='darkorange', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--', label='Random Classifier')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                # Precision-recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                average_precision = average_precision_score(y_test, y_pred_proba)

                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.plot(recall, precision, linewidth=2, label=f'Average Precision = {average_precision:.3f}')
                ax2.set_title('Precision-Recall Curve')
                ax2.set_xlabel('Recall')
                ax2.set_ylabel('Precision')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                st.pyplot(fig2)
                plt.close(fig2)

            # Calibration curve
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

            fig3, ax3 = plt.subplots(figsize=(8, 6))
            ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax3.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model Calibration')
            ax3.set_title('Calibration Curve')
            ax3.set_xlabel('Mean Predicted Probability')
            ax3.set_ylabel('Fraction of Positives')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            st.pyplot(fig3)
            plt.close(fig3)

        except Exception as e:
            st.error(f"❌ Error creating visualizations: {str(e)}")

        # Show uploaded EDA visualizations
        if eda_files:
            st.subheader("📁 Uploaded EDA Visualizations")
            for i, image_file in enumerate(eda_files):
                try:
                    st.image(image_file, caption=image_file.name, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Error displaying {image_file.name}: {str(e)}")

else:
    st.info("👆 Upload your `filtered_dataset.csv` file to get started.")
    
    # Instructions
    st.markdown("""
    ### 📋 Instructions:
    1. **Upload your CSV file** containing LAPD crime data
    2. **Upload EDA visualizations** (optional PNG files from your Colab analysis)
    3. The dashboard will automatically:
       - Preprocess the data
       - Train a logistic regression model
       - Generate prediction visualizations
       - Display your uploaded EDA charts
    
    ### 📊 Required CSV Columns:
    - `DATE_OCC`: Date of occurrence
    - `TIME_OCC`: Time of occurrence  
    - `WEAPON_DESC`: Weapon description
    - `AREA_NAME`: Area name
    - `VICT_AGE`: Victim age
    - `VICT_SEX`: Victim gender
    - `PREMIS_DESC`: Premise description
    - `CRM_CD_DESC`: Crime code description
    """)
