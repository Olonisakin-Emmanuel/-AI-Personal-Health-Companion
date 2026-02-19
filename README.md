# 🩺 AI Personal Health Companion

An intelligent hybrid AI healthcare assistant that combines Machine Learning classification with LLM-powered reasoning to predict illnesses, analyze medical reports, provide multilingual AI health support, and visualize prediction history through an interactive dashboard.

🌍 **Live App:**  
👉 https://olonisakin-emmanuel-ai-partner.streamlit.app/

---

## 🚀 Features

- 🤖 Hybrid ML-based symptom disease prediction
- 🧠 Confidence-based AI fallback system
- 📊 Interactive predictions dashboard
- 💬 Multilingual AI Health Chat Assistant
- 📄 Medical Report Analyzer (PDF/TXT)
- 🌍 Supports English, Yoruba, Hausa & Igbo
- 🔐 Secure session-based logging
- ☁️ Deployed on Streamlit Community Cloud

---

## 🖼️ Application Preview

### 🏠 Home Page
![Home](assets/home.png)

---

### 🤖 Symptom Checker (After Prediction)
![Symptom Checker](assets/symptom_checker.png)

---

### 📊 Predictions Dashboard
![Dashboard](assets/dashboard.png)

---

### 💬 AI Health Chat Assistant
![Chat Interface](assets/chat.png)

---
### 🩺 Medical Report Analyzer
![Medical Report Analyzer](assets/medical_report.png)

---
## 🧠 Hybrid AI Architecture

The Symptom Checker uses a hybrid ML–LLM architecture:

1️⃣ A trained Scikit-learn Machine Learning model predicts possible diseases from selected symptoms.

2️⃣ The system calculates a confidence score and assigns a risk level (Low, Medium, High).

3️⃣ If the risk is Medium or High, the ML prediction result is prioritized.

4️⃣ If the risk is Low, the system automatically falls back to the OpenAI API to generate intelligent health guidance and recommendations.

This architecture combines:

- Structured ML classification
- Confidence-based risk assessment
- AI-powered natural language reasoning
- Fallback orchestration logic

## 🧠 How It Works

### 1️⃣ Symptom Checker
Users select symptoms or enter custom symptoms.  
The system:
- Encodes selected symptoms into model-ready features
- Uses a trained Scikit-learn classifier for disease prediction
- Calculates prediction confidence
- Assigns dynamic risk level (Low / Medium / High)
- Triggers AI fallback logic when risk is low


---

### 2️⃣ Dashboard
- Displays previous predictions
- Shows confidence vs risk visualization
- Tracks session logs

---

### 3️⃣ AI Health Chat
- Users ask health-related questions
- Responses generated using OpenAI API
- Supports multiple Nigerian languages

---

### 4️⃣ Medical Report Analyzer
- Upload PDF or TXT medical reports
- AI summarizes key findings in simple language

---

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn**
- **Streamlit**
- **OpenAI API**
- **Pandas**
- **Plotly**
- **PyPDF2**

---

## ⚙️ Installation (Run Locally)

```bash
git clone https://github.com/Olonisakin-Emmanuel/AI-Personal-Health-Companion.git
cd AI-Personal-Health-Companion
pip install -r requirements.txt
streamlit run app/app.py

