# Financial Fraud Detection System

👉 **[Try the Live App](https://financial-fraud-detector.streamlit.app/)**

This project presents a **complete end-to-end implementation** of a **financial fraud detection system** built using **Streamlit** and **XGBoost**. The objective is to accurately identify fraudulent transactions based on a range of **financial and behavioral attributes**, with a user-friendly **Streamlit** web app for real-time detection.

## 📌 Features
- **Predicts fraudulent transactions** from large-scale financial datasets.
- **Trained on a 4 million record** and evaluated on 1 million for robust performance.
- Handles class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique).
- Applies **Winsorization** to mitigate the influence of extreme outliers.
- Custom **threshold tuning** to balance **precision** and **recall** for the fraud class.
- Uses a fine-tuned **XGBoost model** optimized via **cross-validation**.
- Ensures consistent preprocessing with a serialized **scikit-learn pipeline**.
- Integrates with **SQLite** to fetch and process live transaction data.
- Features an interactive **Streamlit UI** to process live data and download results.

## 🛠 Installation
### 1. Clone the repository
```sh
git clone https://github.com/TanmayDhamnaskar/financial-fraud-predictor
cd financial-fraud-predictor
```

### 2. Create a virtual environment (recommended)
```sh
python3 -m venv fraud_env
source fraud_env/bin/activate  # On Windows: fraud_env\Scripts\activate
```
### 3. Install dependencies
```sh
pip install -r requirements.txt
```

## 🚀 Running the Streamlit App
```sh
streamlit run app.py
```

## 📂 Project Structure
```sh
financial-fraud-detector/
│── app.py                                      # Streamlit frontend logic
│── custom_transformers.py                      # Custom transformers for pipeline
│── eda_analysis.ipynb                         # Statistical Analysis(EDA)
│── model_pipeline.ipynb                        # Training, preprocessing, tuning
│── model/
│   ├── optimized_xgb_fraud_pipeline.pkl        # Trained XGBoost pipeline
│   └── optimal_threshold.pkl                   # Stored best threshold (max F1-score)
│── requirements.txt                            # Python dependency list
└── README.md                                   # Project documentation
```

## 🏗 Model Details

- **Model:** XGBoost Classifier (tree-based boosting model)
- **Problem Type:** Binary classification (**Fraud** or **Not Fraud**)
- **Training Set:** 4 million records
- **Validation Set:** 1 million records
- **Live Testing:** 1 million records evaluated through **Streamlit**
- **Evaluation Metrics:** **Precision, Recall, F1-Score, AUC-ROC**
- **Threshold Tuning:** Optimal threshold selected and stored to **maximize F1-Score**

## ⚙ Preprocessing Techniques:

**Winsorization:** Applied to reduce the influence of extreme outliers on key numeric features.
**SMOTE:** Used to oversample the minority class (fraud cases) and balance class distribution.
**Encoding & Scaling:** Built-in as part of the training pipeline to ensure consistent transformation during inference.

## 🎯 Threshold Optimization:

- Instead of the default 0.5, a **custom probability threshold** was chosen to maximize the **F1-score** for fraud detection.
- This ensures better **recall** and **precision trade-off**, critical for imbalanced fraud detection tasks.

## 🔍 Features Used

- Transaction metadata (step, type, amount)
- Origin and destination account balances (before and after transaction)
- Engineered features from account behaviors and transaction patterns

## 📑 Notes

- Ensure the files `optimized_xgb_fraud_pipeline.pkl` and `optimal_threshold.pkl` are present in the `model/` directory before running the app.
- The database file (`Database.db`) is not included in the repository; the app operates on pre-processed and batch data.
- Outputs include both full predictions and filtered fraud cases, downloadable as CSV from the app.

## 📘 Alignment with Project Guidelines

- ✅ **Data Source:** Used 6.3 million records from `Fraud_detection` table
- ✅ **Record Split:**
  - Training: 4,000,000
  - Validation: 1,000,000
  - Live Testing: 1,000,000
- ✅ **Consistent Preprocessing:** Managed through a saved pipeline
- ✅ **Model Metrics:** Evaluated using precision, recall, F1, AUC (in `model_pipeline.ipynb`)
- ✅ **SMOTE:** Applied to training set to address severe class imbalance
- ✅ **Outlier Handling:** Winsorization applied to skewed numeric features
- ✅ **Threshold Tuning:** Best threshold saved to ensure model generalization
- ✅ **User Interface:** Real-time prediction with Streamlit + downloadable outputs
- ✅ **Deployment Ready:** Easily extendable to batch processing or REST API

> 📬 This project delivers a production-ready fraud detection system, built end-to-end from data preprocessing to deployment — optimized for large-scale financial transactions


---
💡 *Built with XGBoost, SMOTE, Scikit-learn, and Streamlit*









