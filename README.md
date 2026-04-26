# 🚀 Customer Churn Prediction & Revenue Forecasting (Telecom ML Project)

---

## 📊 2. Dataset Information

**Source:**
https://www.kaggle.com/blastchar/telco-customer-churn

**Description:**

* 7043 customer records
* Features include demographics, services, contracts, and billing details

**Target Variables:**

* `churn` → Classification problem
* `monthlycharges` → Regression problem

---

## ⚙️ 3. Installation & Setup

```bash
git clone https://github.com/KeerthivasanV08/teleconnect-ml-assignment.git
cd teleconnect-ml-assignment

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## ▶️ 4. How to Run the Project

### Option 1 — Run Notebooks (Recommended)

```
notebooks/01_EDA.ipynb
notebooks/02_Preprocessing.ipynb
notebooks/03_Classification.ipynb
notebooks/04_Regression.ipynb
notebooks/05_Interpretation.ipynb
```

### Option 2 — Run Python Modules

```bash
python src/data_loader.py
python src/preprocessing.py
python src/classification.py
python src/regression.py
python src/interpretation.py
```

---

## 📂 5. Project Structure

```
teleconnect-ml-assignment/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
├── src/
├── reports/
│   ├── figures/
│   └── SHAP/
│
├── artifacts/
├── requirements.txt
└── README.md
```

---

## 📊 6. Pipeline Overview

### 1️⃣ Data Understanding & Cleaning (EDA)

* Missing value handling
* Outlier detection (IQR)
* Class imbalance analysis
* Correlation analysis

**Key Insights:**

* Month-to-month contracts → highest churn
* High monthly charges → higher churn risk
* Low tenure → more churn

---

### 2️⃣ Feature Engineering & Preprocessing

**Encoding:**

* One-Hot Encoding
* Label Encoding

**Scaling:**

* StandardScaler
* MinMaxScaler

**Feature Engineering:**

* AvgMonthlySpend = TotalCharges / tenure
* ServiceCount = number of services
* ContractValue = MonthlyCharges × remaining months

**Imbalance Handling:**

* SMOTE
* Random Undersampling
* Class Weights

---

### 3️⃣ Classification (Churn Prediction)

**Models:**

* Logistic Regression
* Decision Tree
* Random Forest
* SVM
* KNN

**Metrics:**

* Accuracy, Precision, Recall, F1-score
* ROC-AUC
* Confusion Matrix

---

### 4️⃣ Regression (Revenue Forecasting)

**Target:** MonthlyCharges

**Models:**

* Linear Regression
* Ridge, Lasso, ElasticNet
* Decision Tree Regressor
* Random Forest Regressor
* SVR

**Metrics:**

* MAE, MSE, RMSE
* R², Adjusted R²

---

### 5️⃣ Model Interpretation (Explainability)

**SHAP Analysis:**

* Global feature importance
* Local explanations

**PDP Analysis:**

* Tenure
* Monthly Charges
* Contract Type

**Key Drivers:**

* Contract type
* Tenure
* Monthly charges
* Tech support
* Fiber optic usage

---

## 📈 7. Business Insights

### 🔑 Top Churn Drivers

* Month-to-month contracts
* Low tenure
* High charges
* No tech support
* Fiber optic users

### 🎯 High-Risk Segments

* New customers (< 6 months)
* High-paying without support
* Monthly contract users

### 💡 Strategy

* Discounts for long-term contracts
* Bundle services
* Lower entry pricing

---

## 💰 8. ROI Analysis

**Assumptions:**

* Retention cost = $50
* Churn loss = $500

**Model Targeting (100 customers):**

* 70 correct predictions
* Savings = $35,000
* Cost = $5,000
* Net Gain = $30,000

**Random Targeting:**

* ~20 correct predictions
* Net gain ≈ $5,000

👉 **~6x better ROI using ML**

---

## 🧠 9. Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* SHAP
* Imbalanced-learn
* Jupyter Notebook

---

## 🔀 10. Git Workflow

**Branches:**

* feature/eda
* feature/preprocessing
* feature/classification
* feature/regression
* feature/interpretation

**Commit Types:**

* feat → new features
* fix → bug fixes
* docs → documentation
* refactor → improvements

---

## 🏁 11. Final Outcome

✔ End-to-end ML pipeline
✔ Feature engineering
✔ Model comparison
✔ Explainable AI
✔ ROI justification
✔ Clean Git workflow

---

## 👨‍💻 Author

**Keerthivasan V**
Machine Learning Project – Telecom Analytics
