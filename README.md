# Engine Predictive Maintenance — MLOps Pipeline

Predicts whether a vehicle engine needs maintenance from six sensor readings, with a fully
automated train → register → deploy pipeline.

## Architecture
HF Dataset → train.py (IQRCapper → StandardScaler → GradientBoosting, GridSearchCV) →
HF Model Hub → deploy.py → HF Streamlit Space, orchestrated by GitHub Actions on every push to main.

## Folder structure
- .github/workflows/pipeline.yml   CI/CD workflow
- data/                            dataset placeholder (source registered on the HF Hub)
- space/                           Streamlit app assets (app.py, preprocessing.py, requirements.txt, Dockerfile, README.md)
- preprocessing.py                 IQRCapper (required to unpickle the model)
- train.py                         train, evaluate, register model to the HF Model Hub
- deploy.py                        redeploy the Space
- requirements.txt                 CI training dependencies
- README.md                        this file

## Input schema (exact order)
Engine_RPM, Lub_Oil_Pressure, Fuel_Pressure, Coolant_Pressure, Lub_Oil_Temperature, Coolant_Temperature
Target: Engine_Condition (0 = Normal, 1 = Needs Maintenance)

## Metrics (held-out test)
Recall 0.86 · Precision 0.68 · F1 0.76 · ROC-AUC 0.70 · Accuracy 0.66

## Pipeline stages (CI/CD)
1. Checkout  2. Set up Python 3.12  3. pip install -r requirements.txt
4. python train.py (train, evaluate, register)  5. python deploy.py (redeploy Space)

## Local run
pip install -r requirements.txt
export HF_TOKEN=...        # write-scoped token
python train.py && python deploy.py

## Secrets
GitHub Actions secret HF_TOKEN (write access) is required for registration and deployment.

## Artifacts
Model repo: engine_maintenance_pipeline.joblib, model_metadata.json, preprocessing.py, README.md
Dataset repo: engine_data.csv, train.csv, test.csv

## Links
- Space:   https://huggingface.co/spaces/imanandshah/engine-maintenance-app
- Model:   https://huggingface.co/imanandshah/engine-maintenance-classifier
- Dataset: https://huggingface.co/datasets/imanandshah/engine-predictive-maintenance

## Limitations
Current-condition classifier (not a failure-date forecast); six overlapping sensors give a moderate
ceiling; not a replacement for physical inspection.

## CI/CD trigger
Any push to `main` (or manual workflow_dispatch) runs the full pipeline.
