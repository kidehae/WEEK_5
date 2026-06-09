Markdown
# Advanced Fraud Detection Architecture for E-Commerce & Bank Transactions

An end-to-end machine learning pipeline designed for **Adey Innovations Inc.** to detect fraudulent financial behaviors across two separate risk streams: context-heavy e-commerce actions and anonymized bank card transactions.

---

## ─── 🏗️ System Architecture ───

[Raw Ingestion Data] ──► [Structured Data Sanitization]
│
▼
[Feature Engineering: Velocity, Geolocation & Temporal Extractions]
│
▼
[Stratified Train/Test Split] ──► (20% Isolated Evaluation Holdout)
│
(80% Training Frame Only)
│
▼
[Synthetic Minority Over-sampling (SMOTE)]
│
▼
[Stratified 5-Fold Cross-Validation Loop]
│
┌──────────────┴──────────────┐
▼                             ▼
[Baseline: Logistic Regression]   [Ensemble: XGBoost / LightGBM]
│
▼
[Explainable AI Telemetry: SHAP Audits]


---

## ─── 📊 Data Profiles & Features ───

The architecture processes two distinct operational datasets located in `data/raw/`:

### 1. E-Commerce Vector Stream (`Fraud_Data.csv`)
* **Contextual Features:** Tracks raw sign-up and purchase timestamps, browser headers, user age, and IPv4 addresses.
* **Engineered Signals:** * `time_since_signup`: Real-time operational account lifespan calculated in total cumulative seconds ($\text{purchase\_time} - \text{signup\_time}$).
  * `device_velocity` & `ip_velocity`: High-frequency transactional burst counters grouped by hardware and IP fingerprints to catch automated script actions.
  * `hour_of_day` & `day_of_week`: Extracted cyclical values to capture off-peak fraud surges.
  * `country`: Derived by mapping raw IPv4 strings into 32-bit long integers and applying an optimized binary search (`pd.merge_asof`) against `IpAddress_to_Country.csv`.

### 2. Banking Card Ledger (`creditcard.csv`)
* **Anonymized Projections:** Features $V_1$ through $V_{28}$ are numerical vectors derived via Principal Component Analysis (PCA) for data privacy.
* **Target Imbalance:** Highly skewed risk target ($0.17\%$ fraud prevalence) requiring specialized processing guardrails.

---

## ─── 🛡️ Machine Learning Guardrails ───

* **Leakage Prevention:** To protect against data leakage, datasets are split using a **Stratified Train-Test Split** ($80/20$) *before* any resampling occurs.
* **Targeted Oversampling:** **SMOTE (Synthetic Minority Over-sampling Technique)** is applied strictly to the isolated training partitions, bringing the minority representation up to a stable 30% baseline for algorithm optimization.
* **Operational Performance Metrics:** Because missing actual fraud (False Negatives) costs significantly more than misclassifying legitimate customers (False Positives), standard classification accuracy is disregarded. Models are evaluated using:
  * **Area Under the Precision-Recall Curve (AUC-PR)**
  * **F1-Score Matrix**
  * **Stratified 5-Fold Cross-Validation Confusion Matrices**

---

## ─── ⚙️ Setup & Deployment ───

### 1. Virtual Environment Initialization
Ensure your local machine uses Python 3.11 or newer. Build and activate your environment container:
```bash
# Build environment
python -m venv venv

# Activation (Windows PowerShell)
venv\Scripts\activate

# Activation (Linux / macOS)
source venv/bin/activate
2. Dependency Management
Install the verified dependencies pinned inside your package ledger:

Bash
pip install -r requirements.txt
3. Running Unit Tests
Validate that your structural preprocessing scripts, bitwise IP operations, and schema requirements are running correctly:

Bash
pytest tests/
4. Repository Blueprint
Plaintext
fraud-detection/
├── data/
│   ├── raw/                  # Original raw immutable source sheets
│   └── processed/            # Scaled features, targets, and resampled matrices
├── models/                   # Serialized production-ready model checkpoints
├── src/
│   ├── __init__.py
│   └── data_preprocessing.py # Core preprocessing and geolocation logic
├── notebooks/
│   ├── eda-fraud-data.ipynb  # E-commerce visual discovery
│   ├── eda-creditcard.ipynb  # Bank ledger visual discovery
│   └── feature-engineering.ipynb # Vector generation and scaling engine
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py # Preprocessing unit tests
├── requirements.txt          # Explicit package ledger
└── README.md                 # System documentation