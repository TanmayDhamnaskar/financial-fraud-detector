import streamlit as st
import pandas as pd
import joblib
import sqlite3

from custom_transformers import InitialCleaner, Winsorizer

# Load pipeline and threshold
model = joblib.load("model/optimized_xgb_fraud_pipeline.pkl")
threshold = joblib.load("model/optimal_threshold.pkl")

# Page config
st.set_page_config(page_title="Fraud Detection", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    h1 {
        color: #083d77;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Real-Time Fraud Detection App")
st.markdown("---")

# @st.cache_data(show_spinner=False)
# def load_live_data():
#     conn = sqlite3.connect("Database.db")
#     df = pd.read_sql("SELECT * FROM Fraud_detection", conn)
#     conn.close()
#     return df.tail(1_000_000)

# def prepare_input(df):
#     return df.drop(columns=['isFraud'], errors='ignore')

# # Initialize session state
# if "data_loaded" not in st.session_state:
#     st.session_state.data_loaded = False
#     st.session_state.raw_df = None

@st.cache_data(show_spinner=False)
def load_live_data():
    df = pd.read_csv("fraud_sample.csv")
    return df  # Already trimmed to 1M rows

def prepare_input(df):
    return df.drop(columns=['isFraud'], errors='ignore')

# Initialize session state
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.raw_df = None

# Load Button
if st.button("🔄 Load Data and Predict") or st.session_state.data_loaded:
    if not st.session_state.data_loaded:
        raw_df = load_live_data()
        X_live = prepare_input(raw_df)
        y_prob = model.predict_proba(X_live.drop(columns=['nameOrig', 'nameDest'], errors='ignore'))[:, 1]
        raw_df['predicted_fraud'] = (y_prob >= threshold).astype(int)
        raw_df['predicted_fraud_label'] = raw_df['predicted_fraud'].map({0: 'Not Fraud', 1: 'Fraud'})
        st.session_state.raw_df = raw_df
        st.session_state.data_loaded = True
    else:
        raw_df = st.session_state.raw_df

    # Layout: Filters on left, Results on right
    left, right = st.columns([1, 2])

    with left:
        st.markdown("### 🔎 Filter Options")

        column_options = list(raw_df.columns)
        selected_column = st.selectbox("Select feature to filter by", column_options)

        unique_vals = raw_df[selected_column].dropna().unique()
        unique_vals_sorted = sorted(map(str, unique_vals))

        if len(unique_vals_sorted) <= 100:
            filter_value = st.selectbox(f"Select value to filter `{selected_column}`", unique_vals_sorted)
        else:
            filter_value = st.text_input(f"Enter value to filter `{selected_column}` (case-insensitive):")

        filtered_df = raw_df.copy()
        if filter_value:
            try:
                filter_val = float(filter_value)
                filtered_df = filtered_df[filtered_df[selected_column] == filter_val]
            except:
                filtered_df = filtered_df[filtered_df[selected_column].astype(str).str.lower() == filter_value.lower()]

        st.markdown("### ℹ️ Instructions")
        st.info("Choose a feature and either select or enter a value to filter predictions. Text filters are case-insensitive. You can also filter by `nameOrig` or `nameDest` to check specific transactions.")

    with right:
        st.markdown("### 📊 Raw Live Data Sample")
        st.dataframe(filtered_df.head(10), use_container_width=True)

        st.markdown("### 🔍 Prediction Results")
        fraud_df = filtered_df[filtered_df['predicted_fraud'] == 1][['step', 'amount', 'nameOrig', 'nameDest', 'predicted_fraud_label']]
        if fraud_df.empty:
            st.info("✅ No fraudulent transactions found in the filtered rows.")
        else:
            st.dataframe(fraud_df, use_container_width=True)

        st.markdown("### 📈 Summary")
        fraud_count = filtered_df['predicted_fraud'].sum()
        total_count = len(filtered_df)
        fraud_pct = (fraud_count / total_count) * 100 if total_count > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📌 Fraud Count", fraud_count)
        with col2:
            st.metric("🧾 Total Records", total_count)
        with col3:
            st.metric("⚠️ Fraud Rate", f"{fraud_pct:.2f}%")

        if total_count > 0:
            csv_all = filtered_df.to_csv(index=False).encode('utf-8')
            csv_fraud = fraud_df.to_csv(index=False).encode('utf-8')

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("⬇️ Download Fraud-Only CSV", data=csv_fraud, file_name="fraud_predictions_only.csv")
            with d2:
                st.download_button("⬇️ Download All Predictions CSV", data=csv_all, file_name="fraud_predictions_all.csv")
else:
    st.info("🔄 Click the **'Load Data and Predict'** button above to fetch and analyze live transactions.")
