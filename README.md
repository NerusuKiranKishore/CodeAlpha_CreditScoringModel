# 💳 Credit Scoring Model — CodeAlpha ML Internship

A machine learning pipeline to predict an individual's creditworthiness using past financial data.

---

## 📁 Project Structure

```
CodeAlpha_CreditScoringModel/
├── data/
│   └── raw/                  # Place your dataset here (credit_data.csv)
├── notebooks/
│   └── EDA.ipynb             # Exploratory Data Analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load and split data
│   ├── preprocessing.py      # Clean, encode, scale features
│   ├── feature_engineering.py# Create new features
│   ├── train.py              # Train all models
│   ├── evaluate.py           # Metrics and plots
│   └── predict.py            # Predict on new input
├── models/                   # Saved .pkl model files
├── reports/
│   └── figures/              # ROC curves, confusion matrices
├── tests/
│   └── test_pipeline.py      # Unit tests
├── main.py                   # Run full pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
- Download from: https://www.kaggle.com/c/GiveMeSomeCredit
- Place `cs-training.csv` inside `data/raw/` folder
- Rename it to `credit_data.csv`

### 3. Run the full pipeline
```bash
python main.py
```

### 4. Predict on new data
```bash
python src/predict.py
```

---

## 📊 Models Used
| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosting (best performance) |

---

## 📈 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix

---

## 🛠️ Tech Stack
- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- Joblib

---

## 👤 Author
**Nerusu Kiran Kishore**  
CodeAlpha ML Internship  
GitHub: [CodeAlpha_CreditScoringModel](https://github.com/yourusername/CodeAlpha_CreditScoringModel)