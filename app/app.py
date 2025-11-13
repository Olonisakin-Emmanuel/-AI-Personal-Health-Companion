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

# --- Only show animation once per session ---
if "welcome_shown" not in st.session_state:
    st.session_state.welcome_shown = True

    # Base directory
    BASE_DIR = os.path.dirname(__file__)

    # Load animation
    with open(os.path.join(BASE_DIR, "welcome_animation.json"), "r") as f:
        welcome_animation = json.load(f)

    # Create a placeholder
    anim_placeholder = st.empty()

    # Display animation inside placeholder
    with anim_placeholder.container():
        st_lottie(welcome_animation, height=300, key="welcome_lottie")
        st.markdown(
            "<h2 style='text-align:center; color:#0D47A1;'>👋Hello Welcome to My AI Health Assistant 🩺</h2>",
            unsafe_allow_html=True
        )

    # Keep animation visible for 3 seconds
    time.sleep(3)

    # Clear the placeholder so the animation disappears
    anim_placeholder.empty()


# ============================================
# 🧠 PAGE SETUP
# ============================================
st.set_page_config(page_title="My AI Health Assistant", layout="wide")
st.title("My AI Health Assistant 🩺")
st.write("Let’s find out what’s going on. Enter your symptoms 🤒")

# ============================================
# 🧭 PAGE NAVIGATION
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
    This Health Assistant is an Ml/AI-powered tool for educational and informational purposes only.  
    It **does not replace professional medical advice, diagnosis, or treatment**.  
    Always consult a licensed healthcare provider for any medical concerns.  
    Use this tool responsibly. **Thank you 🙏**
    """
)

# ============================================
# 1️⃣ LOAD MODELS & DATA (CACHED)
# ============================================
@st.cache_resource
def load_models_and_data():
    # Base directory where app.py resides
    BASE_DIR = os.path.dirname(__file__)
    
    # Models folder inside app/
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    
    # Load all ML models
    models_dict = {
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree_model.pkl")),
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl")),
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl")),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"))
    }
    
    # Load label encoder
    le_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    
    # Load symptom columns
    with open(os.path.join(MODEL_DIR, "symptom_columns.json"), "r") as f:
        symptom_cols = [s.lower().strip() for s in json.load(f)]
    
    return models_dict, le_encoder, symptom_cols


# ============================================
# 2️⃣ MULTI-LANGUAGE UI
# ============================================
language = st.selectbox(
    "Select Language / Zaɓi Yare / Yan aṣayan rẹ / Họrọ Asụsụ gị:",
    ["English", "Yoruba", "Hausa", "Igbo"]
)
translations = {
    "English": {"select_symptoms": "Select your symptoms:", "predict": "🔍 Predict Disease"},
    "Yoruba": {"select_symptoms": "Yan awọn aami aisan rẹ:", "predict": "🔍 Ṣe ayẹwo Arun"},
    "Hausa": {"select_symptoms": "Zaɓi alamomin cutar ku:", "predict": "🔍 Yi Hasashen Cuta"},
    "Igbo": {"select_symptoms": "Họrọ mgbaàmà gị:", "predict": "🔍 Mee Amụma Ọrịa"},
}
lang_for_ai = language

# ============================================
# 3️⃣ DASHBOARD
# ============================================
CSV_FILE = "session_logs.csv"

def show_dashboard():
    st.title("📊 Predictions Dashboard")
    if os.path.exists(CSV_FILE):
        df_logs = pd.read_csv(CSV_FILE, quotechar='"', encoding="utf-8", dtype={"session_id": str})
        last5 = df_logs.tail(5).sort_values(by="timestamp", ascending=False).reset_index(drop=True)
        last5["session_label"] = [f"Prediction {i+1}" for i in range(len(last5))]

        selected_session = st.selectbox("Select a prediction to view:", last5["session_label"])
        session_data = last5[last5["session_label"] == selected_session].iloc[0]

        st.markdown("### 🩺 Prediction Details")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Timestamp:** {session_data['timestamp']}")
            st.write(f"**Symptoms:** {session_data['symptoms']}")
            st.write(f"**Follow-up Answers:** {session_data['followup_answers']}")
        with col2:
            st.write(f"**Predicted Disease:** {session_data['predicted_disease']}")
            st.write(f"**Confidence:** {session_data['confidence']:.1f}%")
            st.write(f"**Risk Level:** {session_data['risk']}")
            if 'ai_tip' in session_data and pd.notna(session_data['ai_tip']):
                st.info(f"💡 Health Tip: {session_data['ai_tip']}")

        # Confidence & Risk Chart
        df_graph = pd.DataFrame({
            "Metric": ["Confidence", "Risk Score"],
            "Value": [
                float(session_data['confidence']),
                0 if session_data['risk'] == "Low" else 50 if session_data['risk'] == "Medium" else 100
            ]
        })
        fig = px.bar(df_graph, x="Metric", y="Value", color="Metric", text="Value",
                     color_discrete_map={"Confidence": "#4CAF50", "Risk Score": "#F44336"},
                     title="Confidence vs Risk", height=400)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        # Summary Card
        st.markdown("---")
        st.markdown(f"""
        <div style="padding:10px; border-radius:8px; background-color:#E3F2FD;">
        <h4 style="color:#0D47A1;">Quick Summary</h4>
        <ul>
            <li><b>Symptoms Count:</b> {len(session_data['symptoms'].split(','))}</li>
            <li><b>Follow-up Questions Answered:</b> {len(json.loads(session_data['followup_answers']))}</li>
            <li><b>Predicted Disease:</b> {session_data['predicted_disease']}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # Analytics
        st.markdown("---")
        st.markdown("### 📈 Top Predicted Diseases")
        top_diseases = df_logs['predicted_disease'].value_counts().head(5)
        st.bar_chart(top_diseases)
    else:
        st.warning("No previous predictions found.")
    st.stop()

if st.session_state.page == "dashboard":
    show_dashboard()


# ============================================
# Load models and data
# ============================================
models, le, symptom_columns = load_models_and_data()

# --- MAIN HEALTH ASSISTANT
st.subheader(translations[language]["select_symptoms"])
selected_symptoms = st.multiselect("Select symptoms", options=symptom_columns)
manual_input = st.text_input("Or type additional symptoms (comma-separated):", placeholder="fever, cough")
user_symptoms = list(set(
    [s.lower().strip() for s in selected_symptoms] +
    [s.lower().strip() for s in manual_input.split(",") if s.strip()]
))


# --- FOLLOW-UP QUESTIONS MAP ---
followup_map = {
    "fever": [
        {"question": "Do you have cough or chest pain?", "symptom_key": "cough"},
        {"question": "Are you feeling chills or sweating?", "symptom_key": "chills"},
        {"question": "Do you have headache?", "symptom_key": "headache"}
    ],
    "cough": [
        {"question": "Is it a dry cough?", "symptom_key": "dry cough"},
        {"question": "Do you have difficulty breathing?", "symptom_key": "shortness of breath"},
        {"question": "Do you have chest pain?", "symptom_key": "chest pain"}
    ],
    "headache": [
        {"question": "Do you feel dizzy?", "symptom_key": "dizziness"},
        {"question": "Do you have blurred vision?", "symptom_key": "blurred vision"},
        {"question": "Do you have nausea or vomiting?", "symptom_key": "nausea"}
    ],
}

# --- SESSION STATE ---
for key in ["followup_index", "selected_main_symptom", "answers", "ml_result", "session_id", "followup_queue"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if key == "followup_index" else None if key in ["selected_main_symptom","ml_result","session_id"] else {}

# --- FUNCTIONS ---
ai_tip_cache = {}

def match_symptoms(symptoms, symptom_columns):
    matched = []
    for s in symptoms:
        s = s.lower().strip()
        closest = get_close_matches(s, symptom_columns, n=1, cutoff=0.7)
        if closest:
            matched.append(closest[0])
        else:
            fuzzy_match, score = process.extractOne(s, symptom_columns)
            if score >= 70:
                matched.append(fuzzy_match)
    return matched

def predict_disease(symptoms):
    matched = match_symptoms(symptoms, symptom_columns)
    if not matched:
        return None, 0
    input_data = [1 if symptom in matched else 0 for symptom in symptom_columns]
    df = pd.DataFrame([input_data], columns=symptom_columns)
    preds = []
    weights = [20,20,20,20,20]  
    for i, model in enumerate(models.values()):
        try:
            pred = le.inverse_transform([model.predict(df)[0]])[0]
            preds.append((pred, weights[i]))
        except Exception as e:
            with open("error_logs.txt", "a") as f:
                f.write(f"{datetime.now()}: {str(e)}\n")
    final_pred = Counter()
    for p, w in preds:
        final_pred[p] += w
    disease = final_pred.most_common(1)[0][0] if final_pred else None
    conf = (final_pred.most_common(1)[0][1] / sum(weights)) * 100 if final_pred else 0
    return disease, conf

def get_ai_tip(disease, language):
    key = f"{disease}_{language}"
    if key in ai_tip_cache:
        return ai_tip_cache[key]
    prompt = f"Provide a very short safe health tip for {disease}, in {language}."
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.4
        )
        tip = resp.choices[0].message.content.strip()
        ai_tip_cache[key] = tip
        return tip
    except Exception as e:
        with open("error_logs.txt", "a") as f:
            f.write(f"{datetime.now()}: {str(e)}\n")
        return "💡 Tip unavailable right now."

# --- RUN PREDICTION (safe version) ---
def run_prediction():
    # Work on a copy to avoid mutating original user_symptoms
    prediction_symptoms = list(user_symptoms)
    
    for k, v in st.session_state.answers.items():
        if v and k not in prediction_symptoms:
            prediction_symptoms.append(k)

    disease, conf = predict_disease(prediction_symptoms)
    ai_tip = ""
    risk = "Low" if conf < 40 else "Medium" if conf < 70 else "High"

    # AI fallback
    if disease is None or conf < 50:
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            ai_prompt = f"Based on these symptoms: {', '.join(prediction_symptoms)}. Suggest 1–3 possible diseases and a short safe health tip in {lang_for_ai}."
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": ai_prompt}],
                max_tokens=150,
                temperature=0.6
            )
            ai_message = response.choices[0].message.content.strip()
            ai_tip = ai_message
            disease = "AI Suggested"
            conf = 0

            st.info(f"🤖 AI Suggested Possible Diseases & Tip:\n{ai_tip}")
            st.success(f"Confidence: {conf:.1f}% | Risk: {risk}")

        except Exception as e:
            with open("error_logs.txt", "a") as f:
                f.write(f"{datetime.now()}: {str(e)}\n")
            ai_tip = "💡 Tip unavailable right now"
            st.warning(ai_tip)

    else:
        # ML prediction
        ai_tip = get_ai_tip(disease, lang_for_ai)
        st.info(f"**Predicted Disease:** {disease}")
        st.success(f"Confidence: {conf:.1f}% | Risk: {risk}")
        st.write(f"💡 {ai_tip}")

    # Log session
    if st.session_state.session_id is None:
        st.session_state.session_id = str(uuid.uuid4())

    result = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symptoms": ", ".join(prediction_symptoms),
        "followup_answers": json.dumps(st.session_state.answers),
        "predicted_disease": disease,
        "confidence": conf,
        "risk": risk,
        "ai_tip": ai_tip,
    }
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(result)

    st.balloons()

    if st.button("🔁 Start New Prediction"):
        for k in ["followup_index", "selected_main_symptom", "answers", "ml_result", "followup_queue"]:
            st.session_state[k] = None
        st.session_state.page = "main"
        st.rerun()



# --- FOLLOW-UP FLOW (MULTI-SYMPTOM) ---
if st.session_state.page == "followup":
    # Build follow-up queue if not exists
    if st.session_state.followup_queue is None:
        st.session_state.followup_queue = [s for s in user_symptoms if s in followup_map]

    if st.session_state.followup_queue:
        main_symptom = st.session_state.followup_queue[0]
        questions = followup_map[main_symptom]
        index = st.session_state.followup_index
        total_q = len(questions)

        st.markdown(f"### 🧠 Follow-up for: **{main_symptom.capitalize()}**")
        st.progress(index / total_q)

        if index < total_q:
            q = questions[index]
            with st.expander(f"Question {index+1} of {total_q}"):
                answer = st.radio(q['question'], ["Yes","No"], key=f"{main_symptom}_{index}")
                if st.button("Next ➡️", key=f"next_{main_symptom}_{index}"):
                    st.session_state.answers[q["symptom_key"]] = (answer=="Yes")
                    st.session_state.followup_index += 1
                    st.rerun()
        else:
            # Remove finished symptom from queue
            st.session_state.followup_queue.pop(0)
            st.session_state.followup_index = 0
            st.rerun()
    else:
        run_prediction()
    st.stop()

# --- MAIN INPUT PAGE ---
if len(user_symptoms) > 0 and st.button(translations[language]["predict"]):
    st.session_state.selected_main_symptom = user_symptoms[0]
    st.session_state.followup_index = 0
    st.session_state.answers = {}
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.followup_queue = None
    st.session_state.page = "followup"
    st.toast("🧠 Starting follow-up...", icon="⚙️")
    time.sleep(0.8)
    st.rerun() 