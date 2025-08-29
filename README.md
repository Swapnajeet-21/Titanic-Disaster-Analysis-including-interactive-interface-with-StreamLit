# 🚢 Titanic Disaster Analysis & Prediction

This project explores the **Titanic Disaster dataset** from Kaggle and applies **Machine Learning techniques** to predict passenger survival.  
It includes **data preprocessing, feature engineering, model training, evaluation**, and an **interactive Streamlit web app** for predictions.

---

## 📂 Project Files

- `train.csv` → Training dataset (includes `Survived` column)  
- `test.csv` → Test dataset (without survival labels)  
- `Titanic Survival Analysis.ipynb` → Jupyter Notebook for data analysis and model building  
- `titanic_streamlit_app.py` → Streamlit app for interactive survival prediction  

---

## ✨ Features

- Data cleaning and handling missing values  
- Feature engineering:
  - Titles from passenger names  
  - Family size & IsAlone indicator  
  - Cabin letter extraction  
  - Ticket prefix extraction  
- ML Models implemented:
  - Logistic Regression  
  - Random Forest  
  - Support Vector Machine (RBF)  
  - K-Nearest Neighbors  
- Cross-validation with accuracy & ROC-AUC evaluation  
- **Streamlit Web App**:
  - Upload `train.csv` and `test.csv`  
  - Train selected model with cross-validation  
  - Make single passenger predictions interactively  
  - Generate `Submission.csv` file for Kaggle  

---

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Swapnajeet-21/titanic-disaster-analysis.git
   cd titanic-disaster-analysis
