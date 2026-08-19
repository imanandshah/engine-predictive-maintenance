# Engine Predictive Maintenance — MLOps Pipeline
Predicts whether a vehicle engine needs maintenance from six sensor readings.

- **Model:** Gradient Boosting pipeline (IQR capping → scaling → classifier)
- **Data & model:** hosted on the Hugging Face Hub
- **App:** https://huggingface.co/spaces/imanandshah/engine-maintenance-app
- **CI/CD:** GitHub Actions retrains, registers, and redeploys on every push to `main`.
