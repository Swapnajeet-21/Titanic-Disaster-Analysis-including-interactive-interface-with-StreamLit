
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Survival Analysis & Prediction (Streamlit)")

st.markdown("""
Upload your **train.csv** (with the `Survived` column) and optionally **test.csv** from Kaggle's Titanic dataset.
This app will:
1) Clean & engineer features
2) Train a model with cross-validation
3) Let you make single/interactive predictions
4) Generate a `Submission.csv` for Kaggle (if you upload test.csv)
""")

# ------------------------------
# Utilities
# ------------------------------

RARE_TITLES = {
    "Lady":"Royalty", "Countess":"Royalty", "Capt":"Officer", "Col":"Officer", "Don":"Royalty",
    "Dr":"Officer", "Major":"Officer", "Rev":"Officer", "Sir":"Royalty", "Jonkheer":"Royalty",
    "Dona":"Royalty", "Ms":"Miss", "Mlle":"Miss", "Mme":"Mrs"
}

def extract_title(name: str) -> str:
    if pd.isna(name):
        return "None"
    # Extract the word between ", " and "."
    if ", " in name and "." in name:
        title = name.split(", ")[1].split(".")[0].strip()
    else:
        title = "None"
    return RARE_TITLES.get(title, title)

def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Title from Name
    out["Title"] = out["Name"].apply(extract_title) if "Name" in out.columns else "None"
    # Family size features
    if {"SibSp","Parch"}.issubset(out.columns):
        out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
        out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    else:
        out["FamilySize"] = 1
        out["IsAlone"] = 1

    # Cabin letter
    if "Cabin" in out.columns:
        out["CabinLetter"] = out["Cabin"].fillna("U").astype(str).str[0]
    else:
        out["CabinLetter"] = "U"

    # Ticket prefix (alphanumeric prefix before digits)
    if "Ticket" in out.columns:
        out["TicketPrefix"] = out["Ticket"].astype(str).str.replace(r"\\d", "", regex=True).str.strip()
        out.loc[out["TicketPrefix"]=="", "TicketPrefix"] = "NONE"
    else:
        out["TicketPrefix"] = "NONE"

    # Fill Embarked with mode
    if "Embarked" in out.columns:
        embarked_mode = out["Embarked"].mode().iloc[0] if out["Embarked"].notna().any() else "S"
        out["Embarked"] = out["Embarked"].fillna(embarked_mode)

    # Fill Fare with median per Pclass if available, else global median
    if "Fare" in out.columns:
        if "Pclass" in out.columns:
            out["Fare"] = out["Fare"].fillna(out.groupby("Pclass")["Fare"].transform("median"))
        out["Fare"] = out["Fare"].fillna(out["Fare"].median())

    # Age imputation: median by (Title, Sex, Pclass) when available
    if "Age" in out.columns:
        grp_cols = [c for c in ["Title","Sex","Pclass"] if c in out.columns]
        if grp_cols:
            out["Age"] = out["Age"].fillna(out.groupby(grp_cols)["Age"].transform("median"))
        out["Age"] = out["Age"].fillna(out["Age"].median())

    # Clip some unreasonable values (safety)
    if "Fare" in out.columns:
        out["Fare"] = out["Fare"].clip(lower=0)
    if "Age" in out.columns:
        out["Age"] = out["Age"].clip(lower=0, upper=100)

    return out

def build_pipeline(model_name: str):
    # Feature groups
    numeric_features = [c for c in ["Age","Fare","SibSp","Parch","FamilySize"] if c in X_cols]
    categorical_features = [c for c in ["Pclass","Sex","Embarked","Title","CabinLetter","TicketPrefix","IsAlone"] if c in X_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
        # OneHot handled by pandas.get_dummies later to keep it simple
    ])

    # We'll one-hot with pandas to keep model choices simple
    model_map = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, gamma="scale", C=1.0, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance")
    }
    model = model_map[model_name]
    return model, numeric_features, categorical_features

def one_hot(df: pd.DataFrame, cat_cols):
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ------------------------------
# Sidebar: Data upload & model choice
# ------------------------------

st.sidebar.header("1) Upload data")
train_file = st.sidebar.file_uploader("Upload train.csv (must include Survived)", type=["csv"])
test_file = st.sidebar.file_uploader("Upload test.csv (optional)", type=["csv"])

st.sidebar.header("2) Choose model")
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Random Forest", "SVM (RBF)", "KNN"])

st.sidebar.header("3) Cross-validation")
n_splits = st.sidebar.slider("Stratified K-Folds", 3, 10, 5)

# ------------------------------
# Main logic
# ------------------------------
if train_file is not None:
    train_df = pd.read_csv(train_file)
    st.subheader("Train data preview")
    st.dataframe(train_df.head())

    if "Survived" not in train_df.columns:
        st.error("Train file must include a 'Survived' column.")
        st.stop()

    test_df = pd.read_csv(test_file) if test_file is not None else None

    # Combine for consistent preprocessing / encoding
    combined = pd.concat([train_df.drop(columns=["Survived"]), test_df], axis=0, ignore_index=True) if test_df is not None else train_df.drop(columns=["Survived"]).copy()
    combined_proc = basic_preprocess(combined)

    # Candidate feature columns (after preprocess)
    global X_cols
    # Keep classic Kaggle columns if present
    base_cols = [c for c in ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
                             "Title","FamilySize","IsAlone","CabinLetter","TicketPrefix"] if c in combined_proc.columns]
    X_cols = base_cols

    # One-hot encode categoricals
    cat_cols = [c for c in ["Pclass","Sex","Embarked","Title","CabinLetter","TicketPrefix","IsAlone"] if c in X_cols]
    combined_encoded = one_hot(combined_proc[X_cols], cat_cols)

    # Split back
    X_all = combined_encoded.iloc[:len(train_df), :]
    y = train_df["Survived"].astype(int).values
    X_test_final = combined_encoded.iloc[len(train_df):, :] if test_df is not None else None

    # Build model
    model, num_feats, cat_feats = build_pipeline(model_choice)

    # Align train/test columns to avoid mismatch
    if X_test_final is not None:
        X_test_final = X_test_final.reindex(columns=X_all.columns, fill_value=0)

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = cross_val_score(model, X_all, y, cv=skf, scoring="accuracy")
    try:
        auc_scores = cross_val_score(model, X_all, y, cv=skf, scoring="roc_auc")
        auc_msg = f"AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}"
    except Exception:
        auc_msg = "AUC: (not available for this model/parameters)"

    st.markdown(f"**CV Accuracy:** {acc_scores.mean():.4f} ± {acc_scores.std():.4f} | **{auc_msg}**")

    # Fit on full training data
    model.fit(X_all, y)

    # Single passenger prediction widget
    st.subheader("🔮 Single Passenger Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        Pclass = st.selectbox("Pclass", [1,2,3])
        Sex = st.selectbox("Sex", ["male","female"])
        Age = st.number_input("Age", min_value=0, max_value=100, value=28)
        Fare = st.number_input("Fare", min_value=0.0, value=32.2, step=0.1)
    with col2:
        SibSp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
        Parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
        Embarked = st.selectbox("Embarked", ["S","C","Q"])
        Title = st.selectbox("Title", ["Mr","Mrs","Miss","Master","Royalty","Officer"])
    with col3:
        has_cabin = st.selectbox("Cabin known?", ["U","A","B","C","D","E","F","G","T"])
        TicketPrefix = st.text_input("Ticket Prefix (optional)", "NONE")

    FamilySize = SibSp + Parch + 1
    IsAlone = int(FamilySize == 1)

    # Build a one-row DataFrame using the same columns
    single = pd.DataFrame([{
        "Pclass": Pclass, "Sex": Sex, "Age": Age, "SibSp": SibSp, "Parch": Parch,
        "Fare": Fare, "Embarked": Embarked, "Title": Title, "FamilySize": FamilySize,
        "IsAlone": IsAlone, "CabinLetter": has_cabin, "TicketPrefix": TicketPrefix if TicketPrefix else "NONE"
    }])

    single_encoded = one_hot(basic_preprocess(single)[X_cols], cat_cols)
    single_encoded = single_encoded.reindex(columns=X_all.columns, fill_value=0)

    prob = model.predict_proba(single_encoded)[0][1] if hasattr(model, "predict_proba") else None
    pred = model.predict(single_encoded)[0]
    st.markdown(f"**Prediction:** {'Survived ✅' if pred==1 else 'Did not survive ❌'}")
    if prob is not None:
        st.markdown(f"**Survival probability:** {prob:.3f}")

    # Batch predictions for test.csv
    st.subheader("📦 Kaggle Submission")
    if test_df is not None:
        preds = model.predict_proba(X_test_final)[:,1] if hasattr(model, "predict_proba") else model.predict(X_test_final)
        # If using probabilities, threshold at 0.5
        submission_pred = (preds >= 0.5).astype(int) if preds.ndim==1 else model.predict(X_test_final)
        # Ensure PassengerId exists
        if "PassengerId" in test_df.columns:
            submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": submission_pred})
        else:
            # Fallback index-based
            submission = pd.DataFrame({"PassengerId": np.arange(892, 892+len(submission_pred)), "Survived": submission_pred})

        st.dataframe(submission.head())
        csv_bytes = submission.to_csv(index=False).encode("utf-8")
        st.download_button("Download Submission.csv", data=csv_bytes, file_name="Submission.csv", mime="text/csv")
    else:
        st.info("Upload **test.csv** to generate a Kaggle submission file.")

else:
    st.info("Please upload **train.csv** to begin.")
