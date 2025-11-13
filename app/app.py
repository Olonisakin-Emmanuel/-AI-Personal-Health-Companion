# ============================================
# app.py — Professional Interactive Health Assistant (ML + AI + Dashboard)
# ============================================
import streamlit as st
import joblib
import json
import pandas as pd
from collections import Counter
from difflib import get_close_matches
from fuzzywuzzy import process
import os
from openai import OpenAI
import time
from datetime import datetime
import csv
import uuid
import plotly.express as px
from streamlit_lottie import st_lottie

# ============================================
# 🧠 PAGE SETUP
# ============================================
st.set_page_config(page_title="My AI Health Assistant", layout="wide")
st.title("My AI Health Assistant 🩺")
st.write("Let’s find out what’s going on. Enter your symptoms 🤒")

# ============================================
# BASE DIRECTORIES
# ============================================
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
CSV_FILE = os.path.join(BASE_DIR, "session_logs.csv")

# ============================================
# 1️⃣ LOAD MODELS & DATA (CACHED)
# ============================================
@st.cache_resource
def load_models_and_data():
    # Load ML models
    models_dict = {
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree_model.pkl")),
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl")),
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl")),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"))
    }
    
    # Label encoder
    le_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    
    # Symptom columns
    with open(os.path.join(MODEL_DIR, "symptom_columns.json"), "r") as f:
        symptom_cols = [s.lower().strip() for s in json.load(f)]
    
    return models_dict, le_encoder, symptom_cols

# Load models and columns
models, le, symptom_columns = load_models_and_data()
st.success("🚀 Ready to Diagnose!")

# ============================================
# 2️⃣ WELCOME ANIMATION (ONCE PER SESSION)
# ============================================
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = True
    with open(os.path.join(BASE_DIR, "welcome_animation.json"), "r") as f:
        welcome_animation = json.load(f)
    anim_placeholder = st.empty()
    with anim_placeholder.container():
        st_lottie(welcome_animation, height=300, key="welcome_lottie")
        st.markdown(
            "<h2 style='text-align:center; color:#0D47A1;'>👋Hello Welcome to My AI Health Assistant 🩺</h2>",
            unsafe_allow_html=True
        )
    time.sleep(3)
    anim_placeholder.empty()

# ============================================
# 3️⃣ PAGE NAVIGATION
# ============================================
if "page" not in st.session_state:
    st.session_state.page = "main"

st.sidebar.title("🔍 Navigation")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "main"
if st.sidebar.button("📊 Dashboard"):
    st.session_state.page = "dashboard"

st.sidebar.markdown(
    """
    ---
    **⚠️ Disclaimer:**  
    This Health Assistant is an ML/AI-powered tool for educational purposes only.  
    It **does not replace professional medical advice**. Always consult a licensed healthcare provider.  
    """
)

# ============================================
# 4️⃣ DASHBOARD
# ============================================
def show_dashboard():
    st.title("📊 Predictions Dashboard")
    if os.path.exists(CSV_FILE):
        df_logs = pd.read_csv(CSV_FILE, dtype={"session_id": str})
        last5 = df_logs.tail(5).sort_values("timestamp", ascending=False).reset_index(drop=True)
        last5["session_label"] = [f"Prediction {i+1}" for i in range(len(last5))]

        selected_session = st.selectbox("Select a prediction:", last5["session_label"])
        session_data = last5[last5["session_label"] == selected_session].iloc[0]

        st.markdown("### 🩺 Prediction Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Timestamp:** {session_data['timestamp']}")
            st.write(f"**Symptoms:** {session_data['symptoms']}")
        with col2:
            st.write(f"**Predicted Disease:** {session_data['predicted_disease']}")
            st.write(f"**Confidence:** {session_data['confidence']:.1f}%")
            st.write(f"**Risk Level:** {session_data['risk']}")
            if 'ai_tip' in session_data and pd.notna(session_data['ai_tip']):
                st.info(f"💡 Tip: {session_data['ai_tip']}")

        # Charts
        df_graph = pd.DataFrame({
            "Metric": ["Confidence", "Risk Score"],
            "Value": [
                float(session_data['confidence']),
                0 if session_data['risk']=="Low" else 50 if session_data['risk']=="Medium" else 100
            ]
        })
        fig = px.bar(df_graph, x="Metric", y="Value", color="Metric", text="Value",
                     color_discrete_map={"Confidence":"#4CAF50","Risk Score":"#F44336"}, height=400)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No previous predictions found.")
    st.stop()

if st.session_state.page == "dashboard":
    show_dashboard()

# ============================================
# 5️⃣ MAIN HEALTH ASSISTANT
# ============================================
st.subheader("Select your symptoms:")
selected_symptoms = st.multiselect("Select symptoms", options=symptom_columns)
manual_input = st.text_input("Or type additional symptoms (comma-separated):", placeholder="fever, cough")
user_symptoms = list(set([s.lower().strip() for s in (selected_symptoms + manual_input.split(",")) if s.strip()]))

# Initialize session state
for key in ["followup_index","selected_main_symptom","answers","ml_result","session_id","followup_queue"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key=="followup_index" else None if key in ["selected_main_symptom","ml_result","session_id"] else {}

# Follow-up map
followup_map = {
    "fever":[{"question":"Do you have cough?","symptom_key":"cough"}],
    "cough":[{"question":"Is it dry?","symptom_key":"dry cough"}]
}

# Symptom matching
def match_symptoms(symptoms, columns):
    matched=[]
    for s in symptoms:
        s = s.lower().strip()
        closest = get_close_matches(s, columns, n=1, cutoff=0.7)
        if closest:
            matched.append(closest[0])
        else:
            fuzzy_match, score = process.extractOne(s, columns)
            if score >= 70:
                matched.append(fuzzy_match)
    return matched

# Disease prediction
def predict_disease(symptoms):
    matched = match_symptoms(symptoms, symptom_columns)
    if not matched:
        return None, 0
    input_data = [1 if s in matched else 0 for s in symptom_columns]
    df = pd.DataFrame([input_data], columns=symptom_columns)
    preds=[]
    weights=[20]*len(models)
    for i, model in enumerate(models.values()):
        try:
            pred = le.inverse_transform([model.predict(df)[0]])[0]
            preds.append((pred, weights[i]))
        except:
            continue
    final_pred = Counter()
    for p,w in preds:
        final_pred[p]+=w
    disease = final_pred.most_common(1)[0][0] if final_pred else None
    conf = (final_pred.most_common(1)[0][1]/sum(weights))*100 if final_pred else 0
    return disease, conf

# --- RUN PREDICTION ---
def run_prediction():
    disease, conf = predict_disease(user_symptoms)
    st.success(f"Predicted Disease: {disease} | Confidence: {conf:.1f}%")
    st.balloons()

# Trigger prediction
if st.button("Predict Disease"):
    run_prediction()
