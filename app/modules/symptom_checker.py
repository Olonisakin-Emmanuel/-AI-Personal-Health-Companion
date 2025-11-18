# symptom_checker.py
import streamlit as st
import joblib
import json
import pandas as pd
from collections import Counter
from difflib import get_close_matches
from fuzzywuzzy import process
from datetime import datetime
import os, csv, uuid
from openai import OpenAI

CSV_FILE = "session_logs.csv"
ai_tip_cache = {}

# -------------------------------
# Load models & symptom columns
# -------------------------------
@st.cache_resource
def load_models_and_data():
    BASE_DIR = os.path.dirname(__file__)
    MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

    models_dict = {
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree_model.pkl")),
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl")),
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl")),
        "SVM": joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes_model.pkl"))
    }

    le_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODEL_DIR, "symptom_columns.json"), "r") as f:
        symptom_cols = [s.lower().strip() for s in json.load(f)]

    return models_dict, le_encoder, symptom_cols

# -------------------------------
# Symptom matching & prediction
# -------------------------------
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

def predict_disease(symptoms, models, le, symptom_columns):
    matched = match_symptoms(symptoms, symptom_columns)
    if not matched:
        return None, 0
    input_data = [1 if symptom in matched else 0 for symptom in symptom_columns]
    df = pd.DataFrame([input_data], columns=symptom_columns)
    preds = []
    weights = [20, 20, 20, 20, 20]
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

# -------------------------------
# Follow-up questions
# -------------------------------
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

# -------------------------------
# Main function
# -------------------------------
def run_symptom_checker():
    st.title("🤖 Symptom Checker")

    # --- Initialize session_state safely ---
    for key in ["page", "followup_index", "selected_main_symptom", "answers", "followup_queue", "session_id"]:
        if key not in st.session_state:
            st.session_state[key] = "main" if key == "page" else 0 if key=="followup_index" else None if key in ["selected_main_symptom", "session_id"] else {}

    models, le, symptom_columns = load_models_and_data()

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

    st.subheader(translations[language]["select_symptoms"])
    selected_symptoms = st.multiselect("Select symptoms", options=symptom_columns)
    manual_input = st.text_input("Or type additional symptoms (comma-separated):", placeholder="fever, cough")

    user_symptoms = list(set(
        [s.lower().strip() for s in selected_symptoms] +
        [s.lower().strip() for s in manual_input.split(",") if s.strip()]
    ))

    # --- Trigger follow-up ---
    if len(user_symptoms) > 0 and st.button(translations[language]["predict"]):
        st.session_state.selected_main_symptom = user_symptoms[0]
        st.session_state.followup_index = 0
        st.session_state.answers = {}
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.followup_queue = None
        st.session_state.page = "followup"
        st.rerun()

    # --- Follow-up logic ---
    if st.session_state.page == "followup":
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
                st.session_state.followup_queue.pop(0)
                st.session_state.followup_index = 0
                st.rerun()
        else:
            run_prediction(user_symptoms, models, le, symptom_columns, language)

# -------------------------------
# Prediction runner
# -------------------------------
def run_prediction(user_symptoms, models, le, symptom_columns, language):
    prediction_symptoms = list(user_symptoms)
    for k, v in st.session_state.answers.items():
        if v and k not in prediction_symptoms:
            prediction_symptoms.append(k)

    disease, conf = predict_disease(prediction_symptoms, models, le, symptom_columns)
    risk = "Low" if conf < 40 else "Medium" if conf < 70 else "High"

    # AI fallback if ML is weak
    ai_tip = ""
    if disease is None or conf < 50:
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            ai_prompt = f"Based on these symptoms: {', '.join(prediction_symptoms)}. Suggest 1–3 possible diseases and a short safe health tip in English."
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": ai_prompt}],
                max_tokens=150,
                temperature=0.6
            )
            ai_tip = response.choices[0].message.content.strip()
            disease = "AI Suggested"
            conf = 0
            st.info(f"🤖 AI Suggested Possible Diseases & Tip:\n{ai_tip}")
            st.success(f"Confidence: {conf:.1f}% | Risk: {risk}")
        except Exception as e:
            st.warning("💡 AI tip unavailable right now.")
    else:
        ai_tip = get_ai_tip(disease, language)
        st.info(f"**Predicted Disease:** {disease}")
        st.success(f"Confidence: {conf:.1f}% | Risk: {risk}")
        st.write(f"💡 {ai_tip}")

    # --- Log session ---
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
        for k in ["followup_index", "selected_main_symptom", "answers", "followup_queue", "session_id"]:
            st.session_state[k] = None if k in ["selected_main_symptom","session_id"] else 0 if k=="followup_index" else {}
        st.session_state.page = "main"
        st.rerun()
