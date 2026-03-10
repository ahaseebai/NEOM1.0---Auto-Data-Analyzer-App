import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from datetime import datetime
import json
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart DROP AI", page_icon="🧠", layout="wide")

# ---------------- GEMINI API SETUP ----------------
GEMINI_API_KEY = "AIzaSyAAh6CPj15D59OKJGi8GFpF2aBIT-XfnJo"

try:
    genai.configure(api_key=GEMINI_API_KEY)
    
    # List available models
    available_models = []
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            available_models.append(model.name)
    
    # Select appropriate model
    GEMINI_MODEL_NAME = None
    if 'models/gemini-1.5-pro' in available_models:
        GEMINI_MODEL_NAME = 'models/gemini-1.5-pro'
    elif 'models/gemini-1.0-pro' in available_models:
        GEMINI_MODEL_NAME = 'models/gemini-1.0-pro'
    elif 'models/gemini-pro' in available_models:
        GEMINI_MODEL_NAME = 'models/gemini-pro'
    else:
        GEMINI_MODEL_NAME = available_models[0] if available_models else None
    
    if GEMINI_MODEL_NAME:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
        st.sidebar.error("No suitable Gemini model found")
        
except Exception as e:
    GEMINI_AVAILABLE = False
    st.sidebar.warning(f"Gemini API not available: {str(e)}")

# ---------------- HEADER ----------------
col1, col2 = st.columns([1, 6])
with col1:
    st.image("neom1.0.png", width=1500)
with col2:
    st.markdown("## Smart DROP – AI Data Analyzer")
    st.caption("From Data → Intelligence → Prediction")

st.markdown("---")

# ---------------- SIDEBAR ----------------
# st.sidebar.title("Navigation")
st.sidebar.image("neom1.0.png", width=300)
menu = st.sidebar.radio(
    "Go to",
    ["Home", "Upload Data", "Explore", "Handle Data", "Data Cleaning", "Feature Encoding", 
     "Data Correlation", "Advanced Visualization", "AI Agent", "Gemini AI", "AI Insights", 
     "ML Lab", "Chatbot", "Download", "About"]
)

# Add Gemini status to sidebar
if GEMINI_AVAILABLE:
    st.sidebar.success(f"Gemini AI Connected ({GEMINI_MODEL_NAME.split('/')[-1]})")
else:
    st.sidebar.error("Gemini AI Disabled")

# ---------------- HOME ----------------
if menu == "Home":
    st.title("Welcome to Smart DROP")
    st.markdown("""
    ### AI Powered Data Intelligence Platform
    **New Features:**
    - Advanced Data Cleaning
    - Feature Encoding (One-Hot, Label Encoding)
    - Enhanced Data Correlation Analysis
    - **Advanced Visualization with Seaborn & Plotly**
    - Gemini AI Integration
    - Smart Insights
    - Machine Learning
    - Data Export

    **** 
    """)

# ---------------- SESSION DATA ----------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'encoded_df' not in st.session_state:
    st.session_state.encoded_df = None
if 'gemini_history' not in st.session_state:
    st.session_state.gemini_history = []

# ---------------- UPLOAD ----------------
elif menu == "Upload Data":
    st.title("Upload Dataset")
    
    # Multiple upload options
    upload_option = st.radio("Upload Method", ["CSV File", "Excel File", "Paste Data", "Sample Dataset"])
    
    if upload_option == "CSV File":
        file = st.file_uploader("Upload CSV File", type=['csv'])
        if file:
            df = pd.read_csv(file)
            st.session_state.df = df
            st.session_state.cleaned_df = df.copy()
            st.session_state.encoded_df = None
            st.success("CSV Dataset Uploaded Successfully")
            
    elif upload_option == "Excel File":
        file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
        if file:
            df = pd.read_excel(file)
            st.session_state.df = df
            st.session_state.cleaned_df = df.copy()
            st.session_state.encoded_df = None
            st.success("Excel Dataset Uploaded Successfully")
            
    elif upload_option == "Paste Data":
        data_text = st.text_area("Paste your data (CSV format):", height=200)
        if data_text:
            try:
                df = pd.read_csv(StringIO(data_text))
                st.session_state.df = df
                st.session_state.cleaned_df = df.copy()
                st.session_state.encoded_df = None
                st.success("Data Pasted Successfully")
            except Exception as e:
                st.error(f"Error parsing data: {e}")
    
    elif upload_option == "Sample Dataset":
        sample_option = st.selectbox("Choose Sample Dataset", 
                                   ["Iris", "Titanic", "Boston Housing", "Diabetes", "Wine Quality"])
        
        if st.button("Load Sample Dataset"):
            try:
                if sample_option == "Iris":
                    from sklearn.datasets import load_iris
                    data = load_iris()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                elif sample_option == "Titanic":
                    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
                elif sample_option == "Boston Housing":
                    from sklearn.datasets import load_boston
                    data = load_boston()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['PRICE'] = data.target
                elif sample_option == "Diabetes":
                    from sklearn.datasets import load_diabetes
                    data = load_diabetes()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['target'] = data.target
                elif sample_option == "Wine Quality":
                    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
                
                st.session_state.df = df
                st.session_state.cleaned_df = df.copy()
                st.session_state.encoded_df = None
                st.success(f"{sample_option} Sample Dataset Loaded")
            except Exception as e:
                st.error(f"Error loading sample: {e}")
    
    # Display uploaded data
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("Dataset Preview")
        
        tab1, tab2, tab3 = st.tabs(["Head", "Tail", "Info"])
        
        with tab1:
            st.dataframe(df.head())
        with tab2:
            st.dataframe(df.tail())
        with tab3:
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

# ---------------- EXPLORE ----------------
elif menu == "Explore":
    df = st.session_state.df
    if df is not None:
        st.title("Data Exploration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Memory", f"{df.memory_usage().sum() / 1024:.1f} KB")
        
        st.subheader("Summary Statistics")
        st.write(df.describe())
        
        st.subheader("Data Types")
        dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
        st.dataframe(dtype_df)
        
        # Unique values for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            st.subheader("Categorical Columns Analysis")
            for col in categorical_cols[:5]:  # Show first 5 columns
                unique_vals = df[col].nunique()
                st.write(f"**{col}**: {unique_vals} unique values")
                if unique_vals <= 20:
                    st.write(f"Values: {df[col].unique().tolist()}")
    else:
        st.warning("Upload data first")

# ---------------- HANDLE DATA ----------------
elif menu == "Handle Data":
    df = st.session_state.df
    if df is not None:
        st.title("Data Handling")
        tab1, tab2, tab3 = st.tabs(["Missing Values", "Duplicates", "Outliers"])

        with tab1:
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            missing_df['Percentage'] = (missing_df['Missing Count'] / len(df)) * 100
            st.dataframe(missing_df)
            
            # Visualize missing values
            if missing_df['Missing Count'].sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                missing_data = missing_df[missing_df['Missing Count'] > 0]
                ax.bar(missing_data['Column'], missing_data['Percentage'])
                ax.set_xlabel('Columns')
                ax.set_ylabel('Missing Percentage (%)')
                ax.set_title('Missing Values by Column')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
        
        with tab2:
            dup = df.duplicated().sum()
            st.write(f"**Total Duplicates:** {dup}")
            if dup > 0:
                st.write(f"**Duplicate Percentage:** {(dup/len(df))*100:.2f}%")
                st.dataframe(df[df.duplicated()].head())
        
        with tab3:
            num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
            if num_cols:
                col = st.selectbox("Select Column", num_cols)
                
                # Calculate outliers using IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lb = Q1 - 1.5*IQR
                ub = Q3 + 1.5*IQR
                out = df[(df[col] < lb) | (df[col] > ub)]
                
                st.write(f"**Outliers Count:** {out.shape[0]}")
                st.write(f"**Outlier Percentage:** {(out.shape[0]/len(df))*100:.2f}%")
                
                # Visualize outliers
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Box plot
                sns.boxplot(y=df[col], ax=ax1)
                ax1.set_title(f'Box Plot of {col}')
                
                # Histogram with outlier boundaries
                sns.histplot(df[col], kde=True, ax=ax2)
                ax2.axvline(lb, color='red', linestyle='--', label=f'Lower Bound: {lb:.2f}')
                ax2.axvline(ub, color='red', linestyle='--', label=f'Upper Bound: {ub:.2f}')
                ax2.set_title(f'Distribution of {col} with Outlier Boundaries')
                ax2.legend()
                
                plt.tight_layout()
                st.pyplot(fig)
                
                if out.shape[0] > 0:
                    st.dataframe(out)
    else:
        st.warning("Upload data first")

# ---------------- DATA CLEANING ----------------
elif menu == "Data Cleaning":
    st.title("Advanced Data Cleaning")
    
    df = st.session_state.df
    if df is not None:
        # Initialize cleaned_df if not exists
        if st.session_state.cleaned_df is None:
            st.session_state.cleaned_df = df.copy()
        
        cleaned_df = st.session_state.cleaned_df
        
        st.subheader("Current Dataset Preview")
        st.dataframe(cleaned_df.head())
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Text Data Cleaning")
            
            # Select column for text cleaning
            text_cols = cleaned_df.select_dtypes(include=['object']).columns.tolist()
            if text_cols:
                selected_text_col = st.selectbox("Select Text Column to Clean", text_cols)
                
                cleaning_options = st.multiselect(
                    "Select Cleaning Operations",
                    [
                        "Remove Special Characters",
                        "Remove Extra Spaces",
                        "Convert to Lowercase",
                        "Remove Numbers",
                        "Remove Email Addresses",
                        "Remove URLs"
                    ]
                )
                
                if st.button("Apply Text Cleaning"):
                    if selected_text_col:
                        original_series = cleaned_df[selected_text_col].copy()
                        
                        for operation in cleaning_options:
                            if operation == "Remove Special Characters":
                                # Keep only alphanumeric and spaces
                                cleaned_df[selected_text_col] = cleaned_df[selected_text_col].astype(str).apply(
                                    lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x)
                                )
                            elif operation == "Remove Extra Spaces":
                                cleaned_df[selected_text_col] = cleaned_df[selected_text_col].astype(str).apply(
                                    lambda x: ' '.join(x.split())
                                )
                            elif operation == "Convert to Lowercase":
                                cleaned_df[selected_text_col] = cleaned_df[selected_text_col].astype(str).str.lower()
                            elif operation == "Remove Numbers":
                                cleaned_df[selected_text_col] = cleaned_df[selected_text_col].astype(str).apply(
                                    lambda x: re.sub(r'\d+', '', x)
                                )
                            elif operation == "Remove Email Addresses":
                                cleaned_df[selected_text_col] = cleaned_df[selected_text_col].astype(str).apply(
                                    lambda x: re.sub(r'\S+@\S+', '', x)
                                )
                            elif operation == "Remove URLs":
                                cleaned_df[selected_text_col] = cleaned_df[selected_text_col].astype(str).apply(
                                    lambda x: re.sub(r'https?://\S+|www\.\S+', '', x)
                                )
                        
                        st.session_state.cleaned_df = cleaned_df
                        st.success(f"Applied {len(cleaning_options)} cleaning operations to '{selected_text_col}'")
                        
                        # Show before/after comparison
                        st.subheader("Before vs After Cleaning")
                        comparison_df = pd.DataFrame({
                            'Before': original_series.head(10),
                            'After': cleaned_df[selected_text_col].head(10)
                        })
                        st.dataframe(comparison_df)
        
        with col2:
            st.subheader("Numeric Data Cleaning")
            
            # Select column for numeric cleaning
            num_cols = cleaned_df.select_dtypes(include=['int64','float64']).columns.tolist()
            if num_cols:
                selected_num_col = st.selectbox("Select Numeric Column to Clean", num_cols)
                
                numeric_operations = st.multiselect(
                    "Select Numeric Operations",
                    [
                        "Remove Outliers (IQR Method)",
                        "Replace with Mean",
                        "Replace with Median",
                        "Log Transform",
                        "Min-Max Normalize",
                        "Standardize (Z-score)"
                    ]
                )
                
                if st.button("Apply Numeric Cleaning"):
                    if selected_num_col:
                        original_values = cleaned_df[selected_num_col].copy()
                        
                        for operation in numeric_operations:
                            if operation == "Remove Outliers (IQR Method)":
                                Q1 = cleaned_df[selected_num_col].quantile(0.25)
                                Q3 = cleaned_df[selected_num_col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                # Cap outliers instead of removing rows
                                cleaned_df[selected_num_col] = cleaned_df[selected_num_col].clip(lower_bound, upper_bound)
                            
                            elif operation == "Replace with Mean":
                                mean_val = cleaned_df[selected_num_col].mean()
                                cleaned_df[selected_num_col] = cleaned_df[selected_num_col].fillna(mean_val)
                            
                            elif operation == "Replace with Median":
                                median_val = cleaned_df[selected_num_col].median()
                                cleaned_df[selected_num_col] = cleaned_df[selected_num_col].fillna(median_val)
                            
                            elif operation == "Log Transform":
                                # Add 1 to handle zeros
                                cleaned_df[selected_num_col] = np.log1p(cleaned_df[selected_num_col])
                            
                            elif operation == "Min-Max Normalize":
                                min_val = cleaned_df[selected_num_col].min()
                                max_val = cleaned_df[selected_num_col].max()
                                if max_val > min_val:
                                    cleaned_df[selected_num_col] = (cleaned_df[selected_num_col] - min_val) / (max_val - min_val)
                            
                            elif operation == "Standardize (Z-score)":
                                mean_val = cleaned_df[selected_num_col].mean()
                                std_val = cleaned_df[selected_num_col].std()
                                if std_val > 0:
                                    cleaned_df[selected_num_col] = (cleaned_df[selected_num_col] - mean_val) / std_val
                        
                        st.session_state.cleaned_df = cleaned_df
                        st.success(f"Applied {len(numeric_operations)} operations to '{selected_num_col}'")
                        
                        # Show statistics before/after
                        st.subheader("Statistics Comparison")
                        stats_df = pd.DataFrame({
                            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                            'Before': [
                                original_values.mean(),
                                original_values.median(),
                                original_values.std(),
                                original_values.min(),
                                original_values.max()
                            ],
                            'After': [
                                cleaned_df[selected_num_col].mean(),
                                cleaned_df[selected_num_col].median(),
                                cleaned_df[selected_num_col].std(),
                                cleaned_df[selected_num_col].min(),
                                cleaned_df[selected_num_col].max()
                            ]
                        })
                        st.dataframe(stats_df)
        
        st.markdown("---")
        st.subheader("General Data Operations")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("Remove All Duplicates"):
                before_rows = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                st.session_state.cleaned_df = cleaned_df
                after_rows = len(cleaned_df)
                st.success(f"Removed {before_rows - after_rows} duplicate rows")
        
        with col4:
            if st.button("Reset to Original Data"):
                st.session_state.cleaned_df = df.copy()
                st.success("Data reset to original")
        
        st.markdown("---")
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(st.session_state.cleaned_df.head())
        
        # Download cleaned data
        csv = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Cleaned Data",
            csv,
            "cleaned_data.csv",
            "text/csv",
            key='download_cleaned'
        )
        
    else:
        st.warning("Upload data first")

# ---------------- FEATURE ENCODING ----------------
elif menu == "Feature Encoding":
    st.title("Feature Encoding & Transformation")
    
    # Use cleaned_df if available, otherwise use original df
    current_df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.df
    
    if current_df is not None:
        st.subheader("Current Dataset")
        st.dataframe(current_df.head())
        
        st.markdown("---")
        
        # Select columns for encoding
        categorical_cols = current_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            st.subheader("Categorical Column Encoding")
            
            col1, col2 = st.columns(2)
            
            with col1:
                encoding_type = st.selectbox(
                    "Select Encoding Method",
                    ["One-Hot Encoding", "Label Encoding", "Binary Encoding", "Frequency Encoding", "Target Encoding"]
                )
                
                selected_cat_cols = st.multiselect(
                    "Select Categorical Columns to Encode",
                    categorical_cols,
                    default=categorical_cols[:min(3, len(categorical_cols))]
                )
            
            with col2:
                st.info("""
                **Encoding Methods:**
                - **One-Hot Encoding**: Creates binary columns for each category
                - **Label Encoding**: Converts categories to numeric labels (0, 1, 2...)
                - **Binary Encoding**: Compact representation of categories
                - **Frequency Encoding**: Replaces categories with their frequencies
                - **Target Encoding**: Uses target variable statistics
                """)
            
            if selected_cat_cols:
                if st.button(f"Apply {encoding_type}"):
                    encoded_df = current_df.copy()
                    encoding_report = []
                    
                    for col in selected_cat_cols:
                        if encoding_type == "One-Hot Encoding":
                            # One-Hot Encoding
                            dummies = pd.get_dummies(encoded_df[col], prefix=col, drop_first=True)
                            encoded_df = pd.concat([encoded_df, dummies], axis=1)
                            encoded_df = encoded_df.drop(columns=[col])
                            encoding_report.append(f" {col}: One-Hot encoded → {len(dummies.columns)} new columns")
                            
                        elif encoding_type == "Label Encoding":
                            # Label Encoding
                            le = LabelEncoder()
                            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                            encoding_report.append(f" {col}: Label encoded → {len(le.classes_)} unique labels")
                            
                        elif encoding_type == "Binary Encoding":
                            # Binary Encoding (simplified)
                            encoded_df[col] = pd.factorize(encoded_df[col])[0]
                            # Convert to binary if needed
                            unique_count = len(encoded_df[col].unique())
                            if unique_count <= 2:
                                encoding_report.append(f" {col}: Binary encoded (0/1)")
                            else:
                                encoding_report.append(f"⚠ {col}: Converted to numeric (not strictly binary)")
                                
                        elif encoding_type == "Frequency Encoding":
                            # Frequency Encoding
                            freq_map = encoded_df[col].value_counts(normalize=True).to_dict()
                            encoded_df[f"{col}_freq"] = encoded_df[col].map(freq_map)
                            encoding_report.append(f" {col}: Frequency encoded")
                            
                        elif encoding_type == "Target Encoding":
                            # Target Encoding (if target exists)
                            numeric_cols = encoded_df.select_dtypes(include=['int64','float64']).columns.tolist()
                            if len(numeric_cols) > 0:
                                # Use first numeric column as target for demo
                                target_col = st.selectbox("Select Target Column for Encoding", numeric_cols)
                                if target_col:
                                    target_mean = encoded_df.groupby(col)[target_col].mean().to_dict()
                                    encoded_df[f"{col}_target_encoded"] = encoded_df[col].map(target_mean)
                                    encoding_report.append(f" {col}: Target encoded using {target_col}")
                            
                        else:
                            # Default: Label Encoding
                            le = LabelEncoder()
                            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                            encoding_report.append(f" {col}: Default label encoded")
                    
                    st.session_state.encoded_df = encoded_df
                    
                    # Show encoding report
                    st.subheader("Encoding Report")
                    for report in encoding_report:
                        st.write(report)
                    
                    # Show before/after comparison
                    st.subheader("Before vs After Encoding")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Original Columns:**")
                        st.write(list(current_df.columns))
                        st.write(f"**Shape:** {current_df.shape}")
                        
                    with col2:
                        st.write("**Encoded Columns:**")
                        st.write(list(encoded_df.columns))
                        st.write(f"**Shape:** {encoded_df.shape}")
                    
                    st.subheader("Encoded Dataset Preview")
                    st.dataframe(encoded_df.head())
                    
                    # Download encoded data
                    csv = encoded_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Encoded Data",
                        csv,
                        "encoded_data.csv",
                        "text/csv",
                        key='download_encoded'
                    )
        else:
            st.info("No categorical columns found for encoding.")
        
        st.markdown("---")
        
        # Numeric Transformation
        st.subheader("Numeric Feature Transformation")
        
        numeric_cols = current_df.select_dtypes(include=['int64','float64']).columns.tolist()
        
        if numeric_cols:
            selected_num_cols = st.multiselect(
                "Select Numeric Columns to Transform",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            transform_method = st.selectbox(
                "Select Transformation Method",
                ["Standardization", "Normalization", "Log Transform", "Square Root", "Power Transform", "Robust Scaling"]
            )
            
            if selected_num_cols and st.button(f"Apply {transform_method}"):
                transformed_df = current_df.copy() if st.session_state.encoded_df is None else st.session_state.encoded_df.copy()
                
                for col in selected_num_cols:
                    if transform_method == "Standardization":
                        # Z-score standardization
                        mean_val = transformed_df[col].mean()
                        std_val = transformed_df[col].std()
                        if std_val > 0:
                            transformed_df[f"{col}_standardized"] = (transformed_df[col] - mean_val) / std_val
                            
                    elif transform_method == "Normalization":
                        # Min-Max normalization
                        min_val = transformed_df[col].min()
                        max_val = transformed_df[col].max()
                        if max_val > min_val:
                            transformed_df[f"{col}_normalized"] = (transformed_df[col] - min_val) / (max_val - min_val)
                            
                    elif transform_method == "Log Transform":
                        # Log transformation (add 1 to handle zeros)
                        transformed_df[f"{col}_log"] = np.log1p(transformed_df[col])
                        
                    elif transform_method == "Square Root":
                        # Square root transformation
                        transformed_df[f"{col}_sqrt"] = np.sqrt(transformed_df[col].abs())
                        
                    elif transform_method == "Power Transform":
                        # Box-Cox like transform (simplified)
                        transformed_df[f"{col}_power"] = np.power(transformed_df[col] + 1, 0.5)
                        
                    elif transform_method == "Robust Scaling":
                        # Robust scaling using IQR
                        Q1 = transformed_df[col].quantile(0.25)
                        Q3 = transformed_df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:
                            transformed_df[f"{col}_robust"] = (transformed_df[col] - transformed_df[col].median()) / IQR
                
                st.session_state.encoded_df = transformed_df
                st.success(f"Applied {transform_method} to selected columns")
                
                # Show transformation preview
                st.subheader("Transformation Preview")
                preview_cols = [f"{col}_standardized" for col in selected_num_cols if f"{col}_standardized" in transformed_df.columns]
                if not preview_cols:
                    preview_cols = [f"{col}_normalized" for col in selected_num_cols if f"{col}_normalized" in transformed_df.columns]
                
                if preview_cols:
                    st.dataframe(transformed_df[preview_cols].head())
        else:
            st.info("No numeric columns found for transformation.")
    
    else:
        st.warning("Upload data first")

# ---------------- DATA CORRELATION ----------------
elif menu == "Data Correlation":
    st.title("Data Correlation Analysis")
    
    df = st.session_state.df
    if df is not None:
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            st.subheader("Select Variables for Correlation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Auto-select first numeric column as dependent if available
                default_target = numeric_cols[0] if numeric_cols else None
                dependent_var = st.selectbox(
                    "Select Dependent Variable (Target)",
                    numeric_cols,
                    index=0 if default_target else 0,
                    key="corr_dependent"
                )
            
            with col2:
                # Remove dependent variable from independent options
                independent_options = [col for col in numeric_cols if col != dependent_var]
                
                # Select all numeric columns except target by default
                default_independent = independent_options[:min(10, len(independent_options))]
                independent_vars = st.multiselect(
                    "Select Independent Variables (Features)",
                    independent_options,
                    default=default_independent,
                    key="corr_independent"
                )
            
            if dependent_var and independent_vars:
                st.markdown("---")
                
                # Calculate correlation matrix
                corr_data = df[independent_vars + [dependent_var]]
                corr_matrix = corr_data.corr()
                
                # 1. HEATMAP
                st.subheader("1. Correlation Heatmap")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, 
                           annot=True, 
                           cmap='coolwarm', 
                           center=0, 
                           ax=ax,
                           fmt='.2f',
                           linewidths=0.5,
                           cbar_kws={'shrink': 0.8})
                ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
                st.pyplot(fig)
                
                # 2. CORRELATION WITH TARGET
                st.subheader("2. Correlation with Target Variable")
                
                corr_with_target = corr_matrix[dependent_var].sort_values(ascending=False)
                corr_df = pd.DataFrame({
                    'Variable': corr_with_target.index,
                    'Correlation': corr_with_target.values,
                    'Strength': pd.cut(
                        abs(corr_with_target.values),
                        bins=[0, 0.3, 0.5, 0.7, 1],
                        labels=['Very Weak', 'Weak', 'Moderate', 'Strong']
                    ),
                    'Direction': ['Positive' if x >= 0 else 'Negative' for x in corr_with_target.values]
                })
                
                # Display correlation table
                st.dataframe(corr_df.style.background_gradient(subset=['Correlation'], cmap='RdYlGn'))
                
                # Visualize correlation with target
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                colors = ['green' if x >= 0 else 'red' for x in corr_with_target.values]
                bars = ax2.bar(corr_with_target.index, corr_with_target.values, color=colors)
                ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax2.axhline(y=0.5, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
                ax2.axhline(y=-0.5, color='blue', linestyle='--', linewidth=0.5, alpha=0.5)
                ax2.set_xlabel('Variables')
                ax2.set_ylabel('Correlation Coefficient')
                ax2.set_title(f'Correlation with Target: {dependent_var}')
                ax2.set_xticklabels(corr_with_target.index, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=9)
                
                st.pyplot(fig2)
                
                # 3. SCATTER PLOTS
                st.subheader("3. Scatter Plots vs Target")
                
                # Limit to first 6 variables for better visualization
                plot_vars = independent_vars[:6]
                
                if plot_vars:
                    # Determine grid size
                    n_cols = 3
                    n_rows = (len(plot_vars) + n_cols - 1) // n_cols
                    
                    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                    axes = axes.flatten() if n_rows > 1 else [axes]
                    
                    for idx, var in enumerate(plot_vars):
                        ax = axes[idx]
                        ax.scatter(df[var], df[dependent_var], alpha=0.5, s=20)
                        
                        # Add regression line
                        z = np.polyfit(df[var], df[dependent_var], 1)
                        p = np.poly1d(z)
                        ax.plot(df[var], p(df[var]), "r--", alpha=0.8)
                        
                        corr_value = corr_matrix.loc[var, dependent_var]
                        ax.set_xlabel(var)
                        ax.set_ylabel(dependent_var)
                        ax.set_title(f"{var} vs {dependent_var}\nCorr: {corr_value:.3f}")
                        ax.grid(True, alpha=0.3)
                    
                    # Hide empty subplots
                    for idx in range(len(plot_vars), len(axes)):
                        axes[idx].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                
                # 4. PAIR PLOT (for up to 5 variables)
                if len(independent_vars) <= 5:
                    st.subheader("4. Pair Plot Analysis")
                    
                    pair_data = df[independent_vars[:5] + [dependent_var]]
                    
                    # Create pair plot with seaborn
                    pair_fig = sns.pairplot(pair_data, 
                                           diag_kind='kde',
                                           plot_kws={'alpha': 0.6, 's': 20},
                                           diag_kws={'fill': True})
                    pair_fig.fig.suptitle('Pair Plot of Selected Variables', y=1.02, fontsize=14, fontweight='bold')
                    st.pyplot(pair_fig)
                
                # 5. STATISTICAL INSIGHTS
                st.subheader("5. Statistical Insights")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    highest_pos = corr_with_target[corr_with_target < 1].max()
                    highest_pos_var = corr_with_target[corr_with_target < 1].idxmax()
                    st.metric(
                        "Highest Positive",
                        f"{highest_pos:.3f}",
                        highest_pos_var
                    )
                
                with col2:
                    lowest_neg = corr_with_target[corr_with_target < 1].min()
                    lowest_neg_var = corr_with_target[corr_with_target < 1].idxmin()
                    st.metric(
                        "Highest Negative",
                        f"{lowest_neg:.3f}",
                        lowest_neg_var
                    )
                
                with col3:
                    avg_corr = corr_with_target[corr_with_target < 1].abs().mean()
                    st.metric("Avg Absolute Correlation", f"{avg_corr:.3f}")
                
                with col4:
                    strong_vars = sum(abs(corr_with_target[corr_with_target < 1]) > 0.7)
                    st.metric("Strong Predictors (>0.7)", strong_vars)
                
                # 6. AI RECOMMENDATIONS
                st.subheader("6. AI Recommendations")
                
                strong_pos = corr_with_target[(corr_with_target > 0.7) & (corr_with_target < 1)]
                strong_neg = corr_with_target[corr_with_target < -0.7]
                weak_vars = corr_with_target[(abs(corr_with_target) < 0.3) & (corr_with_target < 1)]
                moderate_vars = corr_with_target[(abs(corr_with_target) >= 0.3) & (abs(corr_with_target) <= 0.7) & (corr_with_target < 1)]
                
                if not strong_pos.empty:
                    st.success(f"** Strong Positive Predictors:** {', '.join(strong_pos.index[:3])}")
                    st.info("These variables are highly correlated with your target. Consider using them as primary features.")
                
                if not strong_neg.empty:
                    st.success(f"** Strong Negative Predictors:** {', '.join(strong_neg.index[:3])}")
                    st.info("These variables have strong inverse relationships with your target. They can be powerful predictors.")
                
                if not moderate_vars.empty:
                    st.warning(f"** Moderate Predictors:** {len(moderate_vars)} variables")
                    st.info("These variables have moderate correlation. Consider feature engineering to improve their predictive power.")
                
                if not weak_vars.empty:
                    st.error(f"** Weak Predictors:** {len(weak_vars)} variables")
                    st.info("These variables show weak correlation. You might consider removing them from your model to reduce complexity.")
                
                # 7. EXPORT CORRELATION MATRIX
                st.subheader("7. Export Correlation Data")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export correlation matrix as CSV
                    csv = corr_matrix.to_csv().encode('utf-8')
                    st.download_button(
                        "📥 Download Correlation Matrix",
                        csv,
                        "correlation_matrix.csv",
                        "text/csv",
                        key='download_corr_matrix'
                    )
                
                with col2:
                    # Export target correlations
                    target_corr_csv = corr_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Target Correlations",
                        target_corr_csv,
                        "target_correlations.csv",
                        "text/csv",
                        key='download_target_corr'
                    )
                
        else:
            st.warning(" Need at least 2 numeric columns for correlation analysis.")
            st.info(f"Current numeric columns: {numeric_cols}")
            
            # Show available columns
            st.subheader("Available Columns")
            all_cols = df.columns.tolist()
            col_types = df.dtypes.tolist()
            
            cols_df = pd.DataFrame({
                'Column': all_cols,
                'Type': col_types,
                'Is Numeric': [str(dtype) in ['int64', 'float64'] for dtype in col_types]
            })
            st.dataframe(cols_df)
    
    else:
        st.warning("Upload data first")

# ---------------- ADVANCED VISUALIZATION ----------------
elif menu == "Advanced Visualization":
    st.title(" Advanced Visualization Dashboard")
    
    df = st.session_state.df
    if df is not None:
        # Sidebar for visualization controls
        st.sidebar.header("Visualization Settings")
        
        # Select plot type
        plot_type = st.sidebar.selectbox(
            "Select Plot Type",
            [
                "Distribution Plots",
                "Relationship Plots",
                "Categorical Plots",
                "Time Series Plots",
                "Multivariate Analysis",
                "Custom Plot"
            ]
        )
        
        # Select columns based on data type
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if plot_type == "Distribution Plots":
            st.subheader("Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                dist_col = st.selectbox("Select Column", numeric_cols, key="dist_col")
                dist_type = st.selectbox("Select Distribution Plot", 
                                        ["Histogram", "KDE Plot", "Box Plot", "Violin Plot", "ECDF Plot"])
            
            with col2:
                if dist_type == "Histogram":
                    bins = st.slider("Number of Bins", 5, 100, 30)
                    kde = st.checkbox("Add KDE", value=True)
                elif dist_type == "Box Plot":
                    hue_col = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                    hue_col = None if hue_col == "None" else hue_col
        
            if dist_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if dist_type == "Histogram":
                    sns.histplot(data=df, x=dist_col, bins=bins, kde=kde, ax=ax)
                    ax.set_title(f'Distribution of {dist_col}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(dist_col)
                    ax.set_ylabel('Frequency')
                    
                elif dist_type == "KDE Plot":
                    sns.kdeplot(data=df, x=dist_col, fill=True, ax=ax)
                    ax.set_title(f'KDE Plot of {dist_col}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(dist_col)
                    ax.set_ylabel('Density')
                    
                elif dist_type == "Box Plot":
                    if hue_col:
                        sns.boxplot(data=df, x=hue_col, y=dist_col, ax=ax)
                        ax.set_title(f'Box Plot of {dist_col} by {hue_col}', fontsize=14, fontweight='bold')
                    else:
                        sns.boxplot(data=df, y=dist_col, ax=ax)
                        ax.set_title(f'Box Plot of {dist_col}', fontsize=14, fontweight='bold')
                    ax.set_ylabel(dist_col)
                    
                elif dist_type == "Violin Plot":
                    if hue_col:
                        sns.violinplot(data=df, x=hue_col, y=dist_col, ax=ax)
                        ax.set_title(f'Violin Plot of {dist_col} by {hue_col}', fontsize=14, fontweight='bold')
                    else:
                        sns.violinplot(data=df, y=dist_col, ax=ax)
                        ax.set_title(f'Violin Plot of {dist_col}', fontsize=14, fontweight='bold')
                    ax.set_ylabel(dist_col)
                    
                elif dist_type == "ECDF Plot":
                    sns.ecdfplot(data=df, x=dist_col, ax=ax)
                    ax.set_title(f'ECDF Plot of {dist_col}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(dist_col)
                    ax.set_ylabel('Proportion')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistics
                st.subheader("Distribution Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{df[dist_col].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[dist_col].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{df[dist_col].std():.2f}")
                with col4:
                    st.metric("Skewness", f"{df[dist_col].skew():.2f}")
        
        elif plot_type == "Relationship Plots":
            st.subheader("Relationship Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox("X-axis Variable", numeric_cols, key="rel_x")
            with col2:
                y_col = st.selectbox("Y-axis Variable", numeric_cols, key="rel_y")
            with col3:
                rel_type = st.selectbox("Plot Type", ["Scatter Plot", "Line Plot", "Hexbin Plot", "Regression Plot"])
            
            if x_col and y_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if rel_type == "Scatter Plot":
                    hue_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols, key="scatter_hue")
                    hue_data = None if hue_col == "None" else df[hue_col]
                    size_col = st.selectbox("Size by (optional)", ["None"] + numeric_cols, key="scatter_size")
                    size_data = None if size_col == "None" else df[size_col]
                    
                    scatter = ax.scatter(df[x_col], df[y_col], 
                                       c=hue_data if hue_data is not None else 'blue',
                                       s=size_data if size_data is not None else 20,
                                       alpha=0.6, cmap='viridis' if hue_data is not None else None)
                    
                    if hue_data is not None:
                        plt.colorbar(scatter, ax=ax, label=hue_col)
                    
                    ax.set_title(f'{x_col} vs {y_col}', fontsize=14, fontweight='bold')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.grid(True, alpha=0.3)
                    
                elif rel_type == "Line Plot":
                    sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f'{x_col} vs {y_col} - Line Plot', fontsize=14, fontweight='bold')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.grid(True, alpha=0.3)
                    
                elif rel_type == "Hexbin Plot":
                    hb = ax.hexbin(df[x_col], df[y_col], gridsize=50, cmap='viridis')
                    plt.colorbar(hb, ax=ax, label='Count')
                    ax.set_title(f'{x_col} vs {y_col} - Hexbin Plot', fontsize=14, fontweight='bold')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    
                elif rel_type == "Regression Plot":
                    hue_col = st.selectbox("Group by (optional)", ["None"] + categorical_cols, key="reg_hue")
                    hue_data = None if hue_col == "None" else hue_col
                    
                    sns.regplot(data=df, x=x_col, y=y_col, 
                               scatter_kws={'alpha': 0.5}, 
                               line_kws={'color': 'red'}, 
                               ax=ax)
                    
                    if hue_data:
                        # Add different colors for groups
                        unique_groups = df[hue_col].unique()
                        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
                        
                        for group, color in zip(unique_groups, colors):
                            group_data = df[df[hue_col] == group]
                            sns.regplot(data=group_data, x=x_col, y=y_col, 
                                       scatter_kws={'alpha': 0.5, 'color': color},
                                       line_kws={'color': color, 'alpha': 0.7},
                                       label=group, ax=ax)
                        ax.legend(title=hue_col)
                    
                    ax.set_title(f'{x_col} vs {y_col} - Regression Plot', fontsize=14, fontweight='bold')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        elif plot_type == "Categorical Plots":
            st.subheader("Categorical Data Analysis")
            
            if categorical_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    cat_col = st.selectbox("Select Categorical Column", categorical_cols, key="cat_col")
                with col2:
                    num_col = st.selectbox("Select Numeric Column", numeric_cols, key="cat_num_col")
                
                plot_choice = st.selectbox("Select Plot Type",
                                         ["Bar Plot", "Count Plot", "Box Plot", "Violin Plot", 
                                          "Swarm Plot", "Point Plot", "Strip Plot"])
                
                if cat_col and num_col:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    if plot_choice == "Bar Plot":
                        sns.barplot(data=df, x=cat_col, y=num_col, ax=ax, ci=95)
                        ax.set_title(f'Average {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                        
                    elif plot_choice == "Count Plot":
                        sns.countplot(data=df, x=cat_col, ax=ax)
                        ax.set_title(f'Count of {cat_col}', fontsize=14, fontweight='bold')
                        
                    elif plot_choice == "Box Plot":
                        sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
                        ax.set_title(f'Distribution of {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                        
                    elif plot_choice == "Violin Plot":
                        sns.violinplot(data=df, x=cat_col, y=num_col, ax=ax)
                        ax.set_title(f'Violin Plot of {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                        
                    elif plot_choice == "Swarm Plot":
                        sns.swarmplot(data=df, x=cat_col, y=num_col, ax=ax, size=3)
                        ax.set_title(f'Swarm Plot of {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                        
                    elif plot_choice == "Point Plot":
                        sns.pointplot(data=df, x=cat_col, y=num_col, ax=ax, ci=95)
                        ax.set_title(f'Point Plot of {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                        
                    elif plot_choice == "Strip Plot":
                        sns.stripplot(data=df, x=cat_col, y=num_col, ax=ax, jitter=True, alpha=0.5)
                        ax.set_title(f'Strip Plot of {num_col} by {cat_col}', fontsize=14, fontweight='bold')
                    
                    ax.set_xlabel(cat_col)
                    ax.set_ylabel(num_col)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Categorical statistics
                    st.subheader("Category Statistics")
                    stats_df = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
                    st.dataframe(stats_df.style.background_gradient(subset=['mean'], cmap='YlOrRd'))
                    
            else:
                st.warning("No categorical columns found in the dataset.")
        
        elif plot_type == "Time Series Plots":
            st.subheader("Time Series Analysis")
            
            # Check for date columns
            date_cols = []
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() or 'month' in col.lower():
                    date_cols.append(col)
            
            if date_cols:
                date_col = st.selectbox("Select Date/Time Column", date_cols)
                value_col = st.selectbox("Select Value Column", numeric_cols)
                
                try:
                    # Try to convert to datetime
                    df[date_col] = pd.to_datetime(df[date_col])
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(data=df, x=date_col, y=value_col, ax=ax)
                    ax.set_title(f'{value_col} over Time', fontsize=14, fontweight='bold')
                    ax.set_xlabel(date_col)
                    ax.set_ylabel(value_col)
                    ax.grid(True, alpha=0.3)
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error processing time series: {str(e)}")
            else:
                st.warning("No date/time columns detected. Try converting columns to datetime format.")
        
        elif plot_type == "Multivariate Analysis":
            st.subheader("Multivariate Analysis")
            
            if len(numeric_cols) >= 3:
                selected_cols = st.multiselect("Select 3-5 Numeric Columns", 
                                              numeric_cols,
                                              default=numeric_cols[:3])
                
                if len(selected_cols) >= 3:
                    # Pair plot
                    st.subheader("Pair Plot")
                    pair_fig = sns.pairplot(df[selected_cols], 
                                           diag_kind='kde',
                                           plot_kws={'alpha': 0.6, 's': 20},
                                           diag_kws={'fill': True})
                    pair_fig.fig.suptitle('Pair Plot of Selected Variables', y=1.02, fontsize=14, fontweight='bold')
                    st.pyplot(pair_fig)
                    
                    # Correlation heatmap
                    st.subheader("Correlation Heatmap")
                    corr_matrix = df[selected_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                               square=True, linewidths=0.5, ax=ax)
                    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
                    st.pyplot(fig)
        
        elif plot_type == "Custom Plot":
            st.subheader("Custom Plot Builder")
            
            # Interactive plot builder
            x_axis = st.selectbox("Select X-axis", df.columns)
            y_axis = st.selectbox("Select Y-axis", df.columns)
            
            plot_kind = st.selectbox("Select Plot Kind",
                                   ["scatter", "line", "bar", "hist", "box", "kde", "area"])
            
            if st.button("Generate Custom Plot"):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                try:
                    if plot_kind == "scatter":
                        df.plot.scatter(x=x_axis, y=y_axis, ax=ax, alpha=0.6)
                    elif plot_kind == "line":
                        df.plot.line(x=x_axis, y=y_axis, ax=ax)
                    elif plot_kind == "bar":
                        if df[x_axis].nunique() < 20:  # Limit for bar plots
                            df.groupby(x_axis)[y_axis].mean().plot.bar(ax=ax)
                        else:
                            st.warning("Too many unique values for bar plot. Try different columns.")
                    elif plot_kind == "hist":
                        df[y_axis].plot.hist(ax=ax, bins=30, alpha=0.7)
                    elif plot_kind == "box":
                        df[[x_axis, y_axis]].boxplot(by=x_axis, ax=ax)
                    elif plot_kind == "kde":
                        df[y_axis].plot.kde(ax=ax)
                    elif plot_kind == "area":
                        if df[x_axis].nunique() < 50:
                            df.groupby(x_axis)[y_axis].sum().plot.area(ax=ax, alpha=0.7)
                        else:
                            st.warning("Too many unique values for area plot.")
                    
                    ax.set_title(f'{plot_kind.title()} Plot: {x_axis} vs {y_axis}', 
                                fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error creating plot: {str(e)}")
        
        # Export option
        st.sidebar.markdown("---")
        if st.sidebar.button("Export All Plots as PNG"):
            st.info("Export functionality would save all generated plots as PNG files.")
        
    else:
        st.warning("Upload data first")

# ---------------- GEMINI AI ----------------
elif menu == "Gemini AI":
    st.title("Gemini AI Assistant")
    st.caption("Powered by Google Gemini")
    
    if not GEMINI_AVAILABLE:
        st.error("Gemini API is not configured. Please check your API key.")
        st.info("""
        To enable Gemini AI:
        1. Get API key from: https://makersuite.google.com/app/apikey
        2. Make sure billing is enabled
        3. Restart the application
        
        **Alternative: Use OpenRouter API (Free)**
        You can also use free AI APIs from OpenRouter:
        - Sign up at: https://openrouter.ai
        - Get API key
        - Select any model (Claude, GPT, etc.)
        """)
        
        # Alternative API option
        st.subheader("Alternative AI API Setup")
        alt_api_key = st.text_input("Enter OpenRouter API Key (optional):", type="password")
        alt_api_url = st.selectbox("Select API Provider", 
                                  ["OpenRouter", "Groq", "Together AI"])
        
        if alt_api_key and st.button("Connect Alternative AI"):
            # You can implement alternative API here
            st.success("Alternative AI connected!")
            
    else:
        df = st.session_state.df
        if df is not None:
            st.success("Dataset available for analysis")
            
            # Show dataset summary
            with st.expander("Dataset Summary"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Size", f"{df.memory_usage().sum() / 1024:.1f} KB")
            
            # Gemini AI Analysis Options
            st.subheader("AI Analysis Options")
            analysis_option = st.selectbox(
                "Select Analysis Type",
                [
                    "Quick Dataset Overview",
                    "Detailed Data Quality Report",
                    "Feature Analysis & Selection",
                    "ML Model Recommendations",
                    "Business Insights Extraction",
                    "Custom Data Query",
                    "Generate Data Visualizations",
                    "Predictive Analysis"
                ]
            )
            
            # Initialize prompt variable
            prompt = None
            
            if analysis_option == "Quick Dataset Overview":
                prompt = f"""
                Provide a concise overview of this dataset:
                
                Dataset Information:
                - Shape: {df.shape[0]} rows, {df.shape[1]} columns
                - Columns: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}
                - Data Types: {dict(df.dtypes)}
                - Missing Values Total: {df.isnull().sum().sum()}
                
                Please analyze and provide:
                1. Dataset type and potential use cases
                2. Key columns and their significance
                3. Initial data quality assessment
                4. Quick recommendations for next steps
                
                Keep response concise and actionable.
                """
                
            elif analysis_option == "Detailed Data Quality Report":
                prompt = f"""
                Perform comprehensive data quality analysis:
                
                Dataset Statistics:
                - Total Rows: {df.shape[0]}
                - Total Columns: {df.shape[1]}
                - Missing Values per Column: {df.isnull().sum().to_dict()}
                - Duplicate Rows: {df.duplicated().sum()}
                - Column Data Types: {dict(df.dtypes)}
                
                Please provide:
                1. Data completeness score (0-100%)
                2. Data consistency assessment
                3. Column-wise quality issues
                4. Impact on analysis/modeling
                5. Prioritized cleaning recommendations
                """
                
            elif analysis_option == "Feature Analysis & Selection":
                numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
                if numeric_cols:
                    target_col = st.selectbox("Select Target Variable", numeric_cols, key="feature_target")
                    
                    prompt = f"""
                    Analyze features for predicting '{target_col}':
                    
                    Available Features: {len([c for c in numeric_cols if c != target_col])}
                    Sample Correlation Matrix (first 5 features):
                    {df[numeric_cols[:min(6, len(numeric_cols))]].corr().to_string()}
                    
                    Tasks:
                    1. Identify top 5 most important features for {target_col}
                    2. Suggest feature engineering opportunities
                    3. Recommend feature selection strategy
                    4. Highlight multicollinearity risks
                    5. Provide feature transformation suggestions
                    """
                
            elif analysis_option == "ML Model Recommendations":
                prompt = f"""
                Recommend machine learning approaches:
                
                Dataset Characteristics:
                - Size: {df.shape[0]} samples, {df.shape[1]} features
                - Feature Types: {len(df.select_dtypes(include=['int64','float64']).columns)} numeric, 
                  {len(df.select_dtypes(include=['object']).columns)} categorical
                - Target Variable Possibilities: {df.select_dtypes(include=['int64','float64']).columns.tolist()}
                
                Please recommend:
                1. Suitable ML algorithms (regression/classification/clustering)
                2. Model comparison matrix
                3. Expected performance metrics
                4. Required preprocessing steps
                5. Hyperparameter tuning strategy
                """
                
            elif analysis_option == "Business Insights Extraction":
                prompt = f"""
                Extract business insights from this data:
                
                Data Sample (first 3 rows):
                {df.head(3).to_string()}
                
                Column Categories:
                - Numeric: {df.select_dtypes(include=['int64','float64']).columns.tolist()}
                - Categorical: {df.select_dtypes(include=['object']).columns.tolist()}
                
                Extract:
                1. Key trends and patterns
                2. Business opportunities
                3. Risk factors
                4. Performance indicators
                5. Actionable recommendations
                """
                
            elif analysis_option == "Generate Data Visualizations":
                prompt = f"""
                Suggest optimal visualizations for this dataset:
                
                Dataset Details:
                - Columns: {df.columns.tolist()}
                - Numeric Columns: {df.select_dtypes(include=['int64','float64']).columns.tolist()}
                - Categorical Columns: {df.select_dtypes(include=['object']).columns.tolist()}
                
                Suggest:
                1. Distribution plots needed
                2. Relationship visualizations
                3. Time series plots (if applicable)
                4. Comparative charts
                5. Dashboard layout recommendations
                """
                
            elif analysis_option == "Predictive Analysis":
                if len(df.select_dtypes(include=['int64','float64']).columns) >= 2:
                    prompt = f"""
                    Perform predictive analysis:
                    
                    Dataset Stats:
                    - Numeric Features: {df.select_dtypes(include=['int64','float64']).columns.tolist()}
                    - Correlation Overview: Available
                    - Sample Size: {df.shape[0]}
                    
                    Tasks:
                    1. Predictability assessment
                    2. Suitable prediction models
                    3. Expected accuracy range
                    4. Data requirements for good predictions
                    5. Common pitfalls to avoid
                    """
                    
            else:  # Custom Query
                custom_prompt = st.text_area("Enter your custom query about the dataset:", 
                                           height=100,
                                           placeholder="E.g., 'What patterns do you see in sales data?' or 'How should I prepare this data for ML?'")
                if custom_prompt:
                    prompt = f"""
                    Dataset Context:
                    - Shape: {df.shape}
                    - Columns: {df.columns.tolist()[:15]}{'...' if len(df.columns) > 15 else ''}
                    - Data Types: Mixed
                    
                    User Question: {custom_prompt}
                    
                    Please provide a detailed, practical answer based on the dataset structure.
                    """
            
            # Safety settings
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            # Configure generation
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            if prompt and st.button("Run AI Analysis", type="primary"):
                with st.spinner(" AI is analyzing your data..."):
                    try:
                        # Create model with safety settings
                        model = genai.GenerativeModel(
                            model_name=GEMINI_MODEL_NAME,
                            generation_config=generation_config,
                            safety_settings=safety_settings
                        )
                        
                        # Generate response
                        response = model.generate_content(prompt)
                        
                        # Display response
                        st.subheader(" AI Analysis Results")
                        st.markdown("---")
                        
                        # Format and display response
                        if response.text:
                            # Clean and format the response
                            formatted_response = response.text.replace('•', '•').replace('**', '**')
                            
                            # Display in a nice container
                            with st.container():
                                st.markdown(formatted_response)
                            
                            # Save to history
                            st.session_state.gemini_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'analysis_type': analysis_option,
                                'prompt_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                                'response': response.text
                            })
                            
                            # Analysis metrics
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Response Length", f"{len(response.text)} chars")
                            with col2:
                                st.metric("Analysis Type", analysis_option)
                            with col3:
                                st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))
                            
                            # Download option
                            analysis_text = f"Smart DROP AI Analysis Report\n"
                            analysis_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                            analysis_text += f"Analysis Type: {analysis_option}\n"
                            analysis_text += f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns\n"
                            analysis_text += "="*60 + "\n\n"
                            analysis_text += response.text
                            
                            st.download_button(
                                "📥 Download Analysis Report",
                                analysis_text,
                                f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                "text/plain"
                            )
                            
                        else:
                            st.warning("AI returned an empty response. Please try again.")
                            
                    except Exception as e:
                        st.error(f"AI Analysis Error: {str(e)}")
                        st.info("""
                        Troubleshooting Tips:
                        1. Check your API key validity
                        2. Ensure billing is enabled in Google AI Studio
                        3. Try a simpler query
                        4. Check API rate limits
                        """)
                        
                        # Fallback to local analysis if API fails
                        if "API" in str(e) or "key" in str(e).lower():
                            st.warning("Using local analysis as fallback...")
            
            # Show analysis history
            if st.session_state.gemini_history:
                with st.expander("View Analysis History"):
                    for i, item in enumerate(reversed(st.session_state.gemini_history[-5:]), 1):
                        st.markdown(f"**{i}. {item['analysis_type']}** - {item['timestamp']}")
                        with st.expander(f"View response {i}"):
                            st.markdown(item['response'][:500] + "..." if len(item['response']) > 500 else item['response'])
        else:
            st.warning("📁 Please upload data first to use AI analysis")

# ---------------- AI INSIGHTS ----------------
elif menu == "AI Insights":
    df = st.session_state.df
    if df is not None:
        st.title("AI Insights")
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.write("Numeric Columns:", df.select_dtypes(include=['int64','float64']).columns.tolist())
    else:
        st.warning("Upload data first")

# ---------------- ML LAB ----------------
elif menu == "ML Lab":
    df = st.session_state.df
    if df is not None:
        st.title("Machine Learning Lab")
        num_cols = df.select_dtypes(include=['int64','float64','object']).columns.tolist()
        if len(num_cols) >= 2:
            target = st.selectbox("Select Target", num_cols)
            features = st.multiselect("Select Features", [c for c in num_cols if c != target])

            if features:
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                st.subheader("Model Performance")
                st.write("R2 Score:", r2_score(y_test, preds))
                st.write("MSE:", mean_squared_error(y_test, preds))

                st.subheader("Predictions Preview")
                st.dataframe(pd.DataFrame({"Actual": y_test.values, "Predicted": preds}))
    else:
        st.warning("Upload data first")

# ---------------- CHATBOT ----------------
elif menu == "Chatbot":
    st.title("Neom1.0 AI Assistant")
    st.caption("Your ML Engineer & App Guide")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    df = st.session_state.df

    def neom_bot(user_msg, df=None):
        msg = user_msg.lower()

        # App Explanation
        if "what is this app" in msg or "explain neom" in msg:
            return "Neom1.0 is an AI-driven system that simulates how Machine Learning engineers analyze data, detect issues, and build ML models."

        # How to use
        if "how to use" in msg or "guide" in msg:
            return "Upload a dataset first, then explore data, clean it, visualize insights, run the AI Agent, or train ML models."

        # Dataset awareness
        if df is not None:
            if "rows" in msg or "size" in msg:
                return f"Your dataset has {df.shape[0]} rows and {df.shape[1]} columns."

            if "missing" in msg:
                miss = df.isnull().sum().sum()
                return f"Your dataset contains {miss} missing values."

            if "duplicates" in msg:
                dup = df.duplicated().sum()
                return f"Your dataset has {dup} duplicate rows."

            if "columns" in msg:
                return f"Dataset columns: {list(df.columns)}"

            if "ml" in msg or "model" in msg:
                return "You can train a Machine Learning model in the ML Lab section by selecting a target and feature columns."

            if "recommend" in msg or "what next" in msg:
                return "I recommend running the AI Agent or exploring summary statistics before building an ML model."

        # General ML Help
        if "what is machine learning" in msg:
            return "Machine Learning allows systems to learn patterns from data and make predictions without being explicitly programmed."

        return "I'm your Neom1.0 assistant. Ask about the app, your dataset, or Machine Learning."

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    # Input box
    user_input = st.chat_input("Ask Neom1.0 Assistant...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        bot_response = neom_bot(user_input, df)

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

        st.rerun()

# ---------------- DOWNLOAD ----------------
elif menu == "Download":
    df = st.session_state.df
    if df is not None:
        st.title("Download Data")
        
        # Select which version to download
        download_option = st.radio(
            "Select Dataset Version to Download",
            ["Original Data", "Cleaned Data", "Encoded Data"],
            help="Choose which processed version of data to download"
        )
        
        if download_option == "Original Data":
            csv = df.to_csv(index=False).encode('utf-8')
            filename = "original_data.csv"
        elif download_option == "Cleaned Data" and st.session_state.cleaned_df is not None:
            csv = st.session_state.cleaned_df.to_csv(index=False).encode('utf-8')
            filename = "cleaned_data.csv"
        elif download_option == "Encoded Data" and st.session_state.encoded_df is not None:
            csv = st.session_state.encoded_df.to_csv(index=False).encode('utf-8')
            filename = "encoded_data.csv"
        else:
            st.warning(f"{download_option} not available. Using original data.")
            csv = df.to_csv(index=False).encode('utf-8')
            filename = "data.csv"
        
        st.download_button(f"Download {download_option}", csv, filename, "text/csv")
    else:
        st.warning("Upload data first")

# ---------------- AI AGENT ----------------
elif menu == "AI Agent":
    df = st.session_state.df

    if df is not None:
        st.title("Neom1.0 – Autonomous AI Agent")

        class NeomAgent:
            def __init__(self, dataframe):
                self.df = dataframe

            def scan_data(self):
                return {
                    "Rows": self.df.shape[0],
                    "Columns": self.df.shape[1],
                    "Missing Values": self.df.isnull().sum().sum(),
                    "Duplicate Rows": self.df.duplicated().sum(),
                    "Numeric Columns": self.df.select_dtypes(include=['int64','float64']).columns.tolist()
                }

            def detect_patterns(self):
                insights = []
                for col in self.df.select_dtypes(include=['int64','float64']).columns:
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    skew = self.df[col].skew()
                    insights.append(
                        f"Column '{col}' → Mean: {mean:.2f}, Std: {std:.2f}, Skewness: {skew:.2f}"
                    )
                return insights

            def risk_analysis(self):
                risks = []
                if self.df.isnull().sum().sum() > 0:
                    risks.append("Dataset contains missing values")
                if self.df.duplicated().sum() > 0:
                    risks.append("Duplicate rows detected")
                if len(self.df.select_dtypes(include=['int64','float64']).columns) < 2:
                    risks.append("⚠ Limited numeric data for ML modeling")
                return risks

            def recommendations(self):
                rec = []
                rec.append("✔ Consider normalizing numeric features")
                rec.append("✔ Remove duplicates for clean training data")
                rec.append("✔ Feature engineering can improve model accuracy")
                rec.append("✔ Try ML regression/classification depending on target")
                return rec

        agent = NeomAgent(df)

        st.subheader("Agent Scan Report")
        st.json(agent.scan_data())

        st.subheader("Pattern Detection")
        for insight in agent.detect_patterns():
            st.write("•", insight)

        st.subheader("Risk Analysis")
        risks = agent.risk_analysis()
        if risks:
            for r in risks:
                st.warning(r)
        else:
            st.success("No major risks detected")

        st.subheader("Agent Recommendations")
        for rec in agent.recommendations():
            st.success(rec)

    else:
        st.warning("Upload data first")

# ---------------- ABOUT ----------------
elif menu == "About":
    st.title("About Project")
    st.markdown("""
    **Smart DROP AI Dashboard**
    Developed by Abdul Haseeb Memon

    ### Purpose:
    - AI-based data analysis
    - Smart automation
    - Machine learning integration
    - Exhibition-ready project

    ### Technologies:
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Seaborn
    - Plotly
    - Google Gemini AI

    ### Vision:
    Build AI Agents for real-world intelligence systems 🚀
    """)