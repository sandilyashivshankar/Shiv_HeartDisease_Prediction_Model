
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import time
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Heart Disease Prediction AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# LOAD MODEL
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "knn_heart_model.pkl")
scaler_path = os.path.join(BASE_DIR, "heart_scaler.pkl")
columns_path = os.path.join(BASE_DIR, "columns.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
expected_columns = joblib.load(columns_path)

# ----------------------------
# CUSTOM CSS
# ----------------------------

st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"]{
font-family:'Poppins',sans-serif;
}

.stApp{

background:linear-gradient(135deg,#edf7ff,#ffffff);

}

.main-title{

font-size:50px;

font-weight:700;

text-align:center;

color:#0F52BA;

margin-bottom:0;

}

.subtitle{

font-size:20px;

text-align:center;

color:#555;

margin-bottom:30px;

}

.card{

background:white;

padding:25px;

border-radius:18px;

box-shadow:0 8px 30px rgba(0,0,0,.12);

margin-bottom:20px;

}

.metric-card{

background:linear-gradient(135deg,#0F52BA,#3BA3FF);

padding:20px;

border-radius:20px;

color:white;

text-align:center;

box-shadow:0 6px 20px rgba(0,0,0,.2);

}

.result-good{

background:#d4edda;

padding:25px;

border-radius:20px;

font-size:28px;

font-weight:bold;

text-align:center;

color:#155724;

}

.result-bad{

background:#f8d7da;

padding:25px;

border-radius:20px;

font-size:28px;

font-weight:bold;

text-align:center;

color:#721c24;

}

.footer{

text-align:center;

margin-top:50px;

padding:20px;

color:gray;

}

.stButton>button{

width:100%;

height:60px;

font-size:22px;

font-weight:bold;

background:linear-gradient(90deg,#0F52BA,#2196F3);

color:white;

border:none;

border-radius:12px;

transition:.4s;

}

.stButton>button:hover{

background:linear-gradient(90deg,#1565C0,#42A5F5);

transform:scale(1.02);

}

</style>

""",unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------

st.markdown("<h1 class='main-title'> Shiv ❤️ Heart Disease Prediction AI</h1>",unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Machine Learning Based Cardiovascular Risk Assessment System</p>",unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------

with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png",width=120)

    st.title(" Shiv Heart AI")

    st.success(" Machine Learning Model")

    st.markdown("---")

    st.markdown("""
### About

This application predicts the likelihood of Heart Disease using Machine Learning.

**Model Used**

✔ K-Nearest Neighbors

**Developer**

Shiv Shankar Tiwari

**Technology**

- Python
- Scikit-Learn
- Streamlit
- Pandas
""")

    st.markdown("---")

    st.info("This application was developed by Shiv Shankar Tiwari as a Machine Learning project to demonstrate predictive analytics using Python, Scikit-learn, and Streamlit. It is intended for learning, research, and portfolio demonstration purposes only and should not be used as a substitute for professional medical diagnosis or treatment.")
    # ==========================================================
# PATIENT INFORMATION
# ==========================================================

st.markdown("## 👤 Patient Information")
st.markdown("Please enter the patient's clinical information.")

col1, col2 = st.columns(2)

# ---------------- LEFT COLUMN ----------------

with col1:

    st.markdown("### 🩺 Personal Details")

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=40
    )

    sex = st.radio(
        "Gender",
        ["M", "F"],
        horizontal=True
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        (
            "ATA",
            "NAP",
            "TA",
            "ASY"
        ),
        help="Select the patient's chest pain category."
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mmHg)",
        min_value=80,
        max_value=220,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=650,
        value=200
    )


# ---------------- RIGHT COLUMN ----------------

with col2:

    st.markdown("### ❤️ Clinical Information")

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar (>120 mg/dL)",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        (
            "Normal",
            "ST",
            "LVH"
        )
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        min_value=60,
        max_value=220,
        value=150
    )

    exercise_angina = st.radio(
        "Exercise Induced Angina",
        ["N", "Y"],
        horizontal=True
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        (
            "Up",
            "Flat",
            "Down"
        )
    )


st.markdown("---")

# ==========================================================
# PATIENT SUMMARY CARD
# ==========================================================

st.subheader("📋 Patient Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:

    st.metric("Age", f"{age} Years")

    st.metric("Blood Pressure", f"{resting_bp} mmHg")

with summary_col2:

    st.metric("Cholesterol", f"{cholesterol} mg/dL")

    st.metric("Maximum HR", max_hr)

with summary_col3:

    st.metric("Gender", "Male" if sex == "M" else "Female")

    st.metric(
        "Fasting Sugar",
        "High" if fasting_bs == 1 else "Normal"
    )

st.markdown("---")

# ==========================================================
# PREDICT BUTTON
# ==========================================================

predict = st.button("🔍 Predict Heart Disease Risk")
# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    with st.spinner("🧠 AI is analyzing patient data..."):

        time.sleep(2)

        # -----------------------------
        # Create input dictionary
        # -----------------------------

        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,

            "Sex_M": 1 if sex == "M" else 0,

            "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
            "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
            "ChestPainType_TA": 1 if chest_pain == "TA" else 0,

            "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
            "RestingECG_ST": 1 if resting_ecg == "ST" else 0,

            "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,

            "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
            "ST_Slope_Up": 1 if st_slope == "Up" else 0
        }

        # -----------------------------
        # DataFrame
        # -----------------------------

        input_df = pd.DataFrame([raw_input])

        # Add missing columns

        for col in expected_columns:

            if col not in input_df.columns:

                input_df[col] = 0

        # Arrange columns

        input_df = input_df[expected_columns]

        # Scale

        scaled = scaler.transform(input_df)

        # Prediction

        prediction = model.predict(scaled)[0]

        probability = model.predict_proba(scaled)[0]

        risk_probability = probability[1]

        safe_probability = probability[0]

    # ======================================================
    # RESULT
    # ======================================================

    st.markdown("---")

    st.header("📊 Prediction Result")

    col1, col2 = st.columns([2,1])

    with col1:

        if prediction == 1:

            st.markdown(
                """
                <div class="result-bad">
                ⚠️ HIGH RISK OF HEART DISEASE
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-good">
                ✅ LOW RISK OF HEART DISEASE
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:

        if prediction == 1:

            st.metric(
                "Risk Probability",
                f"{risk_probability*100:.2f}%"
            )

        else:

            st.metric(
                "Healthy Probability",
                f"{safe_probability*100:.2f}%"
            )

    st.markdown("### 📈 AI Confidence")

    if prediction == 1:

        st.progress(float(risk_probability))

    else:

        st.progress(float(safe_probability))

    st.markdown("---")

    # ======================================================
    # HEALTH RECOMMENDATION
    # ======================================================

    st.subheader("🩺 Personalized Health Recommendations")

    if prediction == 1:

        st.error("""
### Immediate Recommendations

• Schedule a consultation with a Cardiologist.

• Monitor Blood Pressure regularly.

• Reduce cholesterol-rich foods.

• Stop smoking and alcohol.

• Exercise at least 30 minutes daily.

• Maintain a healthy body weight.

• Reduce stress using yoga or meditation.

• Eat more fruits and vegetables.

• Follow your doctor's advice.
""")

    else:

        st.success("""
### Great News!

Your prediction indicates a lower risk of heart disease.

Maintain these healthy habits:

• Exercise regularly.

• Eat a balanced diet.

• Stay hydrated.

• Sleep 7–8 hours.

• Avoid smoking.

• Continue annual health checkups.

• Maintain a healthy BMI.

Keep taking care of your heart ❤️
""")
        # ==========================================================
# ADVANCED DASHBOARD
# ==========================================================

if predict:

    st.markdown("---")

    st.header("📊 AI Analysis Dashboard")

    dashboard1, dashboard2 = st.columns(2)

    # ==========================================
    # DONUT CHART
    # ==========================================

    with dashboard1:

        st.subheader("Prediction Probability")

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Healthy", "Heart Disease"],
                    values=[
                        safe_probability * 100,
                        risk_probability * 100
                    ],
                    hole=0.70,
                    textinfo="percent+label"
                )
            ]
        )

        fig.update_layout(
            height=420,
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ==========================================
    # RISK BAR
    # ==========================================

    with dashboard2:

        st.subheader("Risk Level")

        risk = risk_probability * 100

        st.metric(
            "Risk Percentage",
            f"{risk:.1f}%"
        )

        st.progress(risk_probability)

        if risk < 30:

            st.success("🟢 Low Risk")

        elif risk < 60:

            st.warning("🟡 Moderate Risk")

        else:

            st.error("🔴 High Risk")

        st.markdown("### Heart Health Score")

        score = int((1 - risk_probability) * 100)

        st.metric(
            "Health Score",
            f"{score}/100"
        )

# ==========================================================
# PATIENT REPORT
# ==========================================================

    st.markdown("---")

    st.header("📋 Patient Report")

    report = pd.DataFrame({

        "Feature":[
            "Age",
            "Gender",
            "Chest Pain",
            "Blood Pressure",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Rest ECG",
            "Maximum HR",
            "Exercise Angina",
            "OldPeak",
            "ST Slope",
            "Prediction"
        ],

        "Value":[
            age,
            sex,
            chest_pain,
            resting_bp,
            cholesterol,
            fasting_bs,
            resting_ecg,
            max_hr,
            exercise_angina,
            oldpeak,
            st_slope,
            "High Risk" if prediction==1 else "Low Risk"
        ]

    })

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

    csv = report.to_csv(index=False).encode()

    st.download_button(

        label="📥 Download Prediction Report",

        data=csv,

        file_name="heart_prediction_report.csv",

        mime="text/csv"

    )

# ==========================================================
# HEART HEALTH TIPS
# ==========================================================

st.markdown("---")

st.header("❤️ Heart Health Tips")

tips1, tips2, tips3 = st.columns(3)

with tips1:

    st.info("""

🥗 Healthy Diet

• Eat fruits

• Green vegetables

• Whole grains

• Reduce sugar

• Reduce salt

""")

with tips2:

    st.success("""

🏃 Physical Activity

• Walk daily

• Exercise 30 min

• Yoga

• Cycling

• Swimming

""")

with tips3:

    st.warning("""

🩺 Medical Care

• Annual Check-up

• Blood Pressure

• Cholesterol Test

• Blood Sugar Test

""")

# ==========================================================
# MODEL DETAILS
# ==========================================================

st.markdown("---")

st.subheader("🤖 Machine Learning Model")

model1, model2, model3 = st.columns(3)

with model1:

    st.metric("Algorithm","KNN")

with model2:

    st.metric("Training Samples","918")

with model3:

    st.metric("Prediction","Binary")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""

<div class="footer">

<h3>❤️ Heart Disease Prediction AI</h3>

Developed by <b>Shiv Shankar Tiwari</b>

Python • Streamlit • Scikit-Learn • Machine Learning

⚠️ This application is intended for getting my personal work expertise.

</div>

""",unsafe_allow_html=True)