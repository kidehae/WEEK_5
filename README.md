
# Advanced Fraud Detection Architecture for E-Commerce & Bank Transactions

An end-to-end machine learning and explainable AI (XAI) pipeline designed for **Adey Innovations Inc.** to detect fraudulent financial behaviors across two separate risk streams: context-heavy e-commerce actions and anonymized bank card transactions.

---

## ─── 🏗️ System Architecture ───

```text
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
         [Baseline: Logistic Regression]   [Ensemble: XGBoost / Random Forest]
                                    │
                                    ▼
                 [Explainable AI Telemetry: SHAP Audits]

```

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
* **Target Imbalance:** Highly skewed risk target ($0.17\%$ fraud prevalence) requiring robust feature scaling on `Amount` and `Time` columns.

---

## ─── 🛡️ Machine Learning Guardrails & Model Performance ───

* **Leakage Prevention:** To protect against data leakage, datasets are split using a **Stratified Train-Test Split** ($80/20$) *before* any resampling occurs.
* **Targeted Oversampling:** **SMOTE (Synthetic Minority Over-sampling Technique)** is applied strictly to the isolated training partitions inside the cross-validation loops, bringing the minority representation up to a stable 30% baseline.

### 📈 Cross-Validation Evaluation Matrix (5-Fold Stratified CV)

#### Stream 1: E-Commerce Behavioral Dataset

| Model Candidate | Mean AUC-PR | AUC-PR SD | Mean F1-Score | F1-Score SD | Status |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression Baseline | 0.7142 | ± 0.0124 | 0.6894 | ± 0.0105 | Evaluated Baseline |
| **XGBoost Gradient Ensemble** | **0.8415** | **± 0.0091** | **0.8122** | **± 0.0083** | **Selected Winner** |

#### Stream 2: Anonymized Banking Credit Card Ledger

| Model Candidate | Mean AUC-PR | AUC-PR SD | Mean F1-Score | F1-Score SD | Status |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression Baseline | 0.7334 | ± 0.0217 | 0.2683 | ± 0.0090 | Evaluated Baseline |
| **Random Forest Ensemble** | **0.7761** | **± 0.0422** | **0.7342** | **± 0.0104** | **Selected Winner** |

---

## ─── 🧠 Explainable AI (XAI) & Business Recommendations ───

Using **SHAP (SHapley Additive exPlanations)** game-theoretic telemetry, we extracted global feature impacts and performed individual case audits (True Positives, False Positives, and False Negatives) to understand our model's decisions:

1. **Primary Fraud Drivers:** The engineered feature `time_since_signup` emerged as the single strongest predictor of fraud. A near-zero value (checkout occurring immediately after registration) strongly increases the model's risk classification probability.
2. **Velocity Traps:** High values for `device_velocity` and `ip_velocity` (multiple rapid transactions across the same hardware/IP) consistently drag individual predictions into high-risk domains.

### 💼 Operational Directives for Adey Innovations Inc.

* **🔒 Throttling Rule 1 (Account Lifespan Control):** Any user whose transactional sequence occurs within **300 seconds of account registration** (`time_since_signup`) must be automatically redirected to a step-up biometric identity verification layer (KYC) before payment authorization.
* **🚫 Throttling Rule 2 (Hardware Velocity Holds):** Implement a hard cap threshold on transaction frequencies. If a single `device_id` attempts more than **3 unique account checkouts within a rolling 1-hour window**, flag the network cluster and place a temporary holds on the transactions.
* **🌍 Throttling Rule 3 (Adaptive Geographic Friction):** For high-risk countries identified via the IP range lookup mapping, require multi-factor authentication (MFA) by default for all transactions exceeding a $100 threshold.

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

```

### 2. Dependency Management

Install the verified dependencies pinned inside your package ledger:

```bash
pip install -r requirements.txt

```

### 3. Running Unit Tests

Validate that your structural preprocessing scripts, bitwise IP operations, and schema requirements are running correctly:

```bash
pytest tests/

```

### 4. Repository Blueprint

```text
fraud-detection/
├── data/
│   ├── raw/                  # Original raw immutable source sheets (Git ignored)
│   └── processed/            # Cleaned, engineered, and scaled transaction frames
├── models/                   # Serialized production-ready model checkpoints (.pkl)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py # Core preprocessing and geolocation logic
│   └── modeling_pipeline.py  # Stratified CV loop and SMOTE pipeline script
├── notebooks/
│   ├── eda-fraud-data.ipynb  # E-commerce visual discovery
│   ├── eda-creditcard.ipynb  # Bank ledger visual discovery
│   ├── feature-engineering.ipynb # Vector generation and scaling engine
│   ├── modeling.ipynb        # Baseline vs Ensemble model training loop
│   └── shap-explainability.ipynb # SHAP global and local waterfall visualizations
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py # Preprocessing unit tests
├── requirements.txt          # Explicit package ledger
└── README.md                 # System documentation

```

---
