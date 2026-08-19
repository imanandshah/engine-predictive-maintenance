import joblib, pandas as pd, streamlit as st
from huggingface_hub import hf_hub_download
import preprocessing  # noqa: F401  (registers IQRCapper for unpickling)

MODEL_REPO = "imanandshah/engine-maintenance-classifier"
MODEL_FILE = "engine_maintenance_pipeline.joblib"
FEATURES = ["Engine_RPM", "Lub_Oil_Pressure", "Fuel_Pressure",
            "Coolant_Pressure", "Lub_Oil_Temperature", "Coolant_Temperature"]

st.set_page_config(page_title="Engine Predictive Maintenance", page_icon="🔧", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE))

model = load_model()
st.title("🔧 Engine Predictive Maintenance")
st.write("Enter the six engine sensor readings to estimate whether the engine is "
         "**Normal** or **Needs Maintenance**.")

c1, c2 = st.columns(2)
with c1:
    rpm = st.number_input("Engine RPM", 0.0, 5000.0, 746.0, 1.0)
    oil_p = st.number_input("Lub Oil Pressure (bar)", 0.0, 15.0, 3.16, 0.01)
    fuel_p = st.number_input("Fuel Pressure (bar)", 0.0, 30.0, 6.20, 0.01)
with c2:
    cool_p = st.number_input("Coolant Pressure (bar)", 0.0, 15.0, 2.17, 0.01)
    oil_t = st.number_input("Lub Oil Temperature (°C)", 0.0, 150.0, 76.82, 0.1)
    cool_t = st.number_input("Coolant Temperature (°C)", 0.0, 250.0, 78.35, 0.1)

if st.button("Predict engine condition", type="primary"):
    X = pd.DataFrame([[rpm, oil_p, fuel_p, cool_p, oil_t, cool_t]], columns=FEATURES)
    pred = int(model.predict(X)[0]); proba = float(model.predict_proba(X)[:, 1][0])
    if pred == 1:
        st.error(f" **Needs Maintenance** — estimated risk: **{proba*100:.1f}%**")
        st.write("Recommendation: schedule an inspection; priority rises with the risk score.")
    else:
        st.success(f" **Normal** — estimated maintenance risk: **{proba*100:.1f}%**")
        st.write("Recommendation: no immediate action; continue routine monitoring.")
    st.caption("Decision-support tool: catches ~86% of faults but raises some false alarms.")

st.markdown("---")
st.caption("Gradient Boosting pipeline (IQR capping → scaling → classifier). "
           "1 = Needs Maintenance, 0 = Normal.")
