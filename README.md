# 🚀 Customer Churn Prediction & Revenue Forecasting (Telecom ML Project)

## 📌 1. Project Title & Description

This project is an end-to-end machine learning pipeline for telecom analytics that includes:

- Customer Churn Prediction (Classification)
- Revenue Forecasting using Monthly Charges (Regression)
- Model Interpretation using SHAP & Partial Dependence Plots (PDP)
- Business Insights and ROI Analysis

The objective is to help a telecom company:
- Identify customers likely to churn
- Understand key churn drivers
- Forecast revenue patterns
- Improve customer retention strategies using data-driven decisions

---

## 📊 2. Dataset Information

**Source:**  
https://www.kaggle.com/blastchar/telco-customer-churn

**Description:**
- 7043 customer records
- Features include demographics, services, contracts, and billing details

**Target Variables:**
- `churn` → Classification problem
- `monthlycharges` → Regression problem

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

## ▶️ 4. How to Run the Project

Option 1 — Run Notebooks (Recommended)
notebooks/01_EDA.ipynb
notebooks/02_Preprocessing.ipynb
notebooks/03_Classification.ipynb
notebooks/04_Regression.ipynb
notebooks/05_Interpretation.ipynb

Option 2 — Run Python Modules
python src/data_loader.py
python src/preprocessing.py
python src/classification.py
python src/regression.py
python src/interpretation.py

## 📂 5. Project Structure

teleconnect-ml-assignment/
│
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned dataset
│
├── notebooks/              # EDA → Interpretation workflow
├── src/                    # Modular ML pipeline code
├── reports/
│   ├── figures/            # EDA & model visualizations
│   └── SHAP/               # Explainability outputs
│
├── artifacts/              # Saved models & scalers
├── requirements.txt
└── README.md

## 📊 6. Pipeline Overview

1️⃣ Data Understanding & Cleaning (EDA)
•	Missing value handling 
•	Outlier detection (IQR method) 
•	Class imbalance analysis 
•	Correlation analysis 
Key Insights:
•	Month-to-month contracts → highest churn 
•	High monthly charges → increased churn risk 
•	Low tenure customers → more likely to churn 
________________________________________

2️⃣ Feature Engineering & Preprocessing
Encoding:
•	One-Hot Encoding (categorical features) 
•	Label Encoding (selected cases) 
Scaling:
•	StandardScaler 
•	MinMaxScaler (comparison performed) 
Feature Engineering:
•	AvgMonthlySpend = TotalCharges / tenure 
•	ServiceCount = number of subscribed services 
•	ContractValue = MonthlyCharges × remaining contract months 
Imbalance Handling:
•	SMOTE 
•	Random Undersampling 
•	Class Weights 
________________________________________

## 3️⃣ Classification (Churn Prediction)
Models Used:
•	Logistic Regression 
•	Decision Tree 
•	Random Forest 
•	SVM 
•	KNN 
Metrics:
•	Accuracy 
•	Precision, Recall, F1-score 
•	ROC-AUC 
•	Confusion Matrix 
Outputs:
•	ROC Curve comparison 
•	Feature importance analysis 
•	Best model selection 

________________________________________
4️⃣ Regression (Revenue Forecasting)
Target: MonthlyCharges
Models Used:
•	Linear Regression 
•	Ridge 
•	Lasso 
•	ElasticNet 
•	Decision Tree Regressor 
•	Random Forest Regressor 
•	SVR 
Metrics:
•	MAE 
•	MSE 
•	RMSE 
•	R² 
•	Adjusted R² 
Evaluation:
•	Actual vs Predicted plots 
•	Residual analysis 
•	Model comparison table 
________________________________________
5️⃣ Model Interpretation (Explainability)
SHAP Analysis:
•	Global feature importance 
•	Local explanations (churned vs retained customers) 
PDP Analysis:
•	Tenure 
•	Monthly Charges 
•	Contract Type 
Key Churn Drivers:
•	Contract type (most important) 
•	Tenure 
•	Monthly charges 
•	Tech support absence 
•	Fiber optic usage 
________________________________________

## 📈 7. Business Insights

1. Top churn drivers
•	Month-to-month contracts 
•	Low tenure customers 
•	High monthly charges 
•	Lack of tech support 
•	Fiber optic service usage

2. High-risk customer segments
•	New customers (< 6 months) 
•	High-paying users without support 
•	Month-to-month contract users 

3. Pricing Strategy
•	Offer discounts for long-term contracts 
•	Bundle services (internet + support) 
•	Reduce entry pricing for new users 
4. Customer Targeting Strategy

Prioritize customers with:
•	High SHAP churn probability 
•	High monthly charges 
•	Low tenure 
________________________________________

## 💰 8. ROI Analysis

Assumptions:
•	Retention cost = $50 
•	Churn loss = $500 
Model-based targeting (100 customers):
•	70 correct churn predictions 
•	Savings = $35,000 
•	Cost = $5,000 
•	Net Gain = $30,000 
Random targeting:
•	~20 correct churn predictions 
•	Net gain ≈ $5,000 
👉 Final Insight:
Model provides ~6x better ROI than random selection.

________________________________________

## 🧠 9. Tech Stack
•	Python 
•	Pandas, NumPy 
•	Scikit-learn 
•	Matplotlib, Seaborn 
•	SHAP (Explainable AI) 
•	Imbalanced-learn (SMOTE) 
•	Jupyter Notebook 
________________________________________

## 🔀 10. Git Workflow

Branch Strategy:
•	feature/eda 
•	feature/preprocessing 
•	feature/classification 
•	feature/regression 
•	feature/interpretation 
Commit Style:
•	feat: new features 
•	fix: bug fixes 
•	docs: documentation 
•	refactor: code improvements 
________________________________________

## 🏁 11. Final Outcome

✔ End-to-end ML pipeline
✔ Feature engineering + preprocessing
✔ Multiple ML model comparison
✔ Explainable AI (SHAP + PDP)
✔ Business ROI justification
✔ Professional Git workflow
________________________________________

## 👨‍💻 Author

Keerthivasan V
Machine Learning Project – Telecom Analytics

