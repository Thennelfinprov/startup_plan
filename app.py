import streamlit as st
import joblib
import numpy as np
import base64

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Startup Funding Predictor",
    layout="wide"
)

# --------------------------------------------------
# Background Image Function
# --------------------------------------------------
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem;
        border-radius: 15px;
    }}
    </style>
    """, unsafe_allow_html=True)

# Set Background
set_background("startup.png")

# --------------------------------------------------
# Load Model & Encoders
# --------------------------------------------------
model = joblib.load("funding_model.pkl")

le_company = joblib.load("company_encoder.pkl")
le_headquarter = joblib.load("hq_encoder.pkl")
le_sector = joblib.load("sector_encoder.pkl")
le_what = joblib.load("what_encoder.pkl")
le_founders = joblib.load("founders_encoder.pkl")
le_investor = joblib.load("investor_encoder.pkl")

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🚀 Startup Funding Prediction App")
st.markdown("### Predict Investment Amount Based on Startup Details")
st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("🏢 Company", le_company.classes_)
    sector = st.selectbox("📊 Sector", le_sector.classes_)
    headquarter = st.selectbox("📍 HeadQuarter", le_headquarter.classes_)
    founded = st.number_input("📅 Founded Year", min_value=1900, max_value=2026, step=1)

with col2:
    what = st.selectbox("💡 What it does", le_what.classes_)
    founders = st.selectbox("👤 Founders", le_founders.classes_)
    investor = st.selectbox("💰 Investor", le_investor.classes_)

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Funding Amount"):

    try:
        # Encode inputs (hidden from user)
        company_enc = le_company.transform([company])[0]
        headquarter_enc = le_headquarter.transform([headquarter])[0]
        sector_enc = le_sector.transform([sector])[0]
        what_enc = le_what.transform([what])[0]
        founders_enc = le_founders.transform([founders])[0]
        investor_enc = le_investor.transform([investor])[0]

        # ⚠️ Order must match X.columns from training
        input_data = np.array([[
            company_enc,
            headquarter_enc,
            sector_enc,
            what_enc,
            founders_enc,
            investor_enc,
            founded
        ]])

        prediction = model.predict(input_data)

        st.success(f"💵 Predicted Funding Amount: ${prediction[0]:,.2f}")

    except Exception as e:
        st.error("⚠️ Prediction Error. Please check model input order.")

import streamlit as st
import joblib
import traceback

try:
    model = joblib.load("funding_model.pkl")
    st.success("Model Loaded Successfully")
except Exception as e:
    st.error("Error loading model:")
    st.text(str(e))
    st.text(traceback.format_exc())


