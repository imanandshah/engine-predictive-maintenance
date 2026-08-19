import os, json
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, classification_report)
from preprocessing import IQRCapper

RS = 42
HF_USERNAME  = "imanandshah"
DATASET_REPO = f"{HF_USERNAME}/engine-predictive-maintenance"
MODEL_REPO   = f"{HF_USERNAME}/engine-maintenance-classifier"
TOKEN = os.environ["HF_TOKEN"]

RENAME = {"Engine rpm": "Engine_RPM", "Lub oil pressure": "Lub_Oil_Pressure",
          "Fuel pressure": "Fuel_Pressure", "Coolant pressure": "Coolant_Pressure",
          "lub oil temp": "Lub_Oil_Temperature", "Coolant temp": "Coolant_Temperature",
          "Engine Condition": "Engine_Condition"}
FEATURES = ["Engine_RPM", "Lub_Oil_Pressure", "Fuel_Pressure",
            "Coolant_Pressure", "Lub_Oil_Temperature", "Coolant_Temperature"]
TARGET = "Engine_Condition"

print("Loading data from Hugging Face ...")
path = hf_hub_download(DATASET_REPO, "engine_data.csv", repo_type="dataset", token=TOKEN)
df = pd.read_csv(path).rename(columns=RENAME).drop_duplicates().reset_index(drop=True)
X, y = df[FEATURES], df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=RS, stratify=y)

def make_pipe(clf):
    return Pipeline([("capper", IQRCapper(1.5)),
                     ("scaler", StandardScaler()),
                     ("model", clf)])

grid = {"model__n_estimators": [120, 200, 300], "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3], "model__subsample": [0.8, 1.0]}
cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)
print("Tuning Gradient Boosting ...")
search = GridSearchCV(make_pipe(GradientBoostingClassifier(random_state=RS)),
                      grid, scoring="f1", cv=cv3, n_jobs=-1).fit(Xtr, ytr)
best = search.best_estimator_

yp, ypr = best.predict(Xte), best.predict_proba(Xte)[:, 1]
metrics = {"Train_Accuracy": accuracy_score(ytr, best.predict(Xtr)),
           "Test_Accuracy": accuracy_score(yte, yp),
           "Balanced_Accuracy": balanced_accuracy_score(yte, yp),
           "Precision": precision_score(yte, yp), "Recall": recall_score(yte, yp),
           "F1": f1_score(yte, yp), "ROC_AUC": roc_auc_score(yte, ypr)}
print("=== Evaluation ===")
print(classification_report(yte, yp, target_names=["Normal", "Needs Maintenance"]))
print({k: round(v, 4) for k, v in metrics.items()})

import joblib
joblib.dump(best, "engine_maintenance_pipeline.joblib")
meta = {"artifact": "engine_maintenance_pipeline.joblib",
        "pipeline": ["IQRCapper", "StandardScaler", "GradientBoostingClassifier"],
        "requires_module": "preprocessing.py", "input_features": FEATURES,
        "positive_class": 1, "class_names": {"0": "Normal", "1": "Requires Maintenance"},
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "model_version": "1.0.0"}
json.dump(meta, open("model_metadata.json", "w"), indent=2)

api = HfApi(token=TOKEN)
api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
for f in ["engine_maintenance_pipeline.joblib", "model_metadata.json", "preprocessing.py"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=MODEL_REPO, repo_type="model")
print("Model registered ->", MODEL_REPO)
