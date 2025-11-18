# app.py — AI Personal Health Companion
import streamlit as st
import os
from streamlit_lottie import st_lottie
import json
from openai import OpenAI
from modules.symptom_checker import run_symptom_checker
from modules.dashboard import show_dashboard
from modules.report_analyzer import analyze_medical_report
import datetime
from io import BytesIO
from fpdf import FPDF
import io
import streamlit as st
from fpdf import FPDF
# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Personal Health Companion",
    page_icon="🩺",
    layout="wide"
)

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("🩺 AI Health Companion")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🤖 Symptom Checker","📄 Medical Report Analyzer", "💬 AI Health Chat", "📊 Dashboard"]
)
# -------------------------
# Sidebar Disclaimer
# -------------------------
st.sidebar.markdown("""
---
**⚠️ Disclaimer:**  
This AI Health Companion is for **educational and informational purposes only**.  
It is **not a substitute for professional medical advice, diagnosis, or treatment**.  
Always consult a qualified healthcare provider regarding any medical concerns.
""")


# -------------------------
# HOME PAGE
# -------------------------
if page == "🏠 Home":
    st.title("🩺 AI Personal Health Companion")
    st.subheader("Your AI-powered health assistant")

    # Load Lottie animation with relative path
    def load_lottie(filename):
        base_path = os.path.dirname(__file__)  # ensures relative to app.py
        file_path = os.path.join(base_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    lottie_animation = load_lottie("welcome_animation.json")
    
    # Layout with animation on the side
    col1, col2 = st.columns([2,3])
    with col1:
        st_lottie(lottie_animation, speed=1, height=300, key="welcome")
    with col2:
        st.write("""
This app helps you:
- ✅ Predict possible illnesses from symptoms  
- ✅ Get follow-up questions for clarification  
- ✅ View dashboards of all past predictions  
- ✅ Chat with an AI health advisor  
- ✅ Multilingual support (English, Yoruba, Hausa, Igbo)  
- ✅ Secure session-based logging
        """)
        st.success("Select a feature from the left sidebar to begin.")

# -------------------------
# SYMPTOM CHECKER
# -------------------------
elif page == "🤖 Symptom Checker":
    run_symptom_checker()

# -------------------------
# Medical Report Analyzer
# -------------------------
elif page == "📄 Medical Report Analyzer":
    analyze_medical_report()

# -------------------------
# DASHBOARD
# -------------------------
elif page == "📊 Dashboard":
    show_dashboard()

# -------------------------
# AI HEALTH CHAT
# -------------------------
elif page == "💬 AI Health Chat":
    st.title("💬 AI Health Chat Assistant")
    st.write("Ask any health-related question (not a medical diagnosis).")

    # -------------------------
    # Initialize session state
    # -------------------------
    if "health_chat" not in st.session_state:
        st.session_state.health_chat = []
    if "ai_language" not in st.session_state:
        st.session_state.ai_language = "English"
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False

    # -------------------------
    # Language selection
    # -------------------------
    lang_options = ["English", "Yoruba", "Hausa", "Igbo"]
    st.selectbox(
        "Choose response language:",
        lang_options,
        index=lang_options.index(st.session_state.ai_language),
        key="ai_lang_selector",
        on_change=lambda: st.session_state.health_chat.clear()
    )
    st.session_state.ai_language = st.session_state.ai_lang_selector

    # -------------------------
    # Clear chat button
    # -------------------------
    if st.button("🗑️ Clear Chat"):
        st.session_state.health_chat = []

    # -------------------------
    # Greetings
    # -------------------------
    greetings = {
        "English": "Hello! 👋 I am your AI Health Assistant. You can ask me health-related questions. How can I help you today?",
        "Yoruba": "Pẹlẹ o! 👋 Emi ni Oluranlọwọ Ilera AI rẹ. O le beere awọn ibeere nipa ilera. Bawo ni MO ṣe le ran ọ lọwọ loni?",
        "Hausa": "Sannu! 👋 Ni ne Mataimakin Lafiya na AI. Kuna iya tambayar tambayoyi game da lafiya. Ta yaya zan iya taimaka muku a yau?",
        "Igbo": "Ndewo! 👋 Abụ m Onye Nrụzi Ahụike AI gị. Ị nwere ike ịjụ ajụjụ gbasara ahụike. Kedu ka m ga-esi nyere gị taa?"
    }
    if len(st.session_state.health_chat) == 0:
        st.session_state.health_chat.append({
            "role": "assistant",
            "content": greetings[st.session_state.ai_language],
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # -------------------------
    # Quick suggested questions
    # -------------------------
    suggestions_translated = {
        "English": [
            "What are symptoms of malaria?",
            "What should I eat for healthy blood pressure?",
            "How can I manage diabetes?",
            "How much exercise is recommended daily?",
            "How often should I check my sugar levels?"
        ],
        "Yoruba": [
            "Kí ni àwọn àmì àrùn ibà?",
            "Kini lati jẹ fun titẹ ẹjẹ to dara?",
            "Bawo ni MO ṣe le ṣakoso àtọgbẹ?",
            "Melo ni adaṣe yẹ ki n ṣe lojoojumọ?",
            "Bawo ni igbagbogbo ni MO yẹ ki n ṣayẹwo suga mi?"
        ],
        "Hausa": [
            "Mene ne alamomin zazzabin cizon sauro?",
            "Me ya kamata in ci don lafiyar hawan jini?",
            "Ta yaya zan sarrafa ciwon suga?",
            "Nawa motsa jiki ake bukata kowace rana?",
            "Sau nawa ya kamata in duba matakin suga na?"
        ],
        "Igbo": [
            "Kedu ihe bu ihe mgbaàmà nke ibà?",
            "Kedu ihe m kwesịrị iri maka ahụike mgbali ọbara?",
            "Kedu ka m ga-esi jikwaa shuga?",
            "Ole ka m kwesịrị mmega kwa ụbọchị?",
            "Kedu ugboro m kwesịrị ilele ọkwa shuga m?"
        ]
    }

    st.write("💡 Quick questions:")
    cols = st.columns(len(suggestions_translated[st.session_state.ai_language]))
    for i, suggestion in enumerate(suggestions_translated[st.session_state.ai_language]):
        if cols[i].button(suggestion):
            st.session_state.health_chat.append({
                "role": "user",
                "content": suggestion,
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.awaiting_response = True

    # -------------------------
    # Display chat
    # -------------------------
    for msg in st.session_state.health_chat:
        timestamp = msg.get("time", "")
        if msg["role"] == "user":
            st.chat_message("user").write(f"{msg['content']} \n\n*{timestamp}*")
        else:
            st.chat_message("assistant").write(f"{msg['content']} \n\n*{timestamp}*")

    # -------------------------
    # User input form
    # -------------------------
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", placeholder="Type your message here...")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.health_chat.append({
            "role": "user",
            "content": user_input,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state.awaiting_response = True

    # -------------------------
    # Generate AI response with typing placeholder
    # -------------------------
    if st.session_state.awaiting_response:
        # Add "typing" message if not present
        if not st.session_state.health_chat or st.session_state.health_chat[-1]["content"] != "AI is typing...":
            st.session_state.health_chat.append({
                "role": "assistant",
                "content": "AI is typing... ⏳",
                "time": ""
            })
            st.rerun()

        # Generate AI reply
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with st.spinner("AI is thinking... ⏳"):
            reply = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[msg for msg in st.session_state.health_chat if msg["content"] != "AI is typing..."],
                temperature=0.6
            )
            bot_message = reply.choices[0].message.content

        # Replace the "typing" message with the actual response
        st.session_state.health_chat[-1] = {
            "role": "assistant",
            "content": bot_message,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.awaiting_response = False
        st.rerun()

    # -------------------------
    # Download chat as PDF & TXT
    # -------------------------
    if st.session_state.health_chat:
        st.markdown("---")
        # PDF
        pdf = FPDF()
        pdf.add_page()
        font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSansMono.ttf")
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 12)

        for msg in st.session_state.health_chat:
            role = msg["role"].capitalize()
            content = msg["content"]
            timestamp = msg.get("time", "")
            pdf.multi_cell(0, 10, f"{role}: {content} ({timestamp})")
            pdf.ln(2)

        pdf_bytes = pdf.output(dest="S").encode("latin1")
        st.download_button(
            label="📄 Download Conversation as PDF",
            data=pdf_bytes,
            file_name="ai_health_chat.pdf",
            mime="application/pdf"
        )

        # TXT
        chat_text = ""
        for msg in st.session_state.health_chat:
            role = msg["role"].capitalize()
            content = msg["content"]
            timestamp = msg.get("time", "")
            chat_text += f"{role}: {content} ({timestamp})\n\n"
        txt_buffer = BytesIO(chat_text.encode("utf-8"))
        st.download_button(
            label="📝 Download Conversation as TXT",
            data=txt_buffer,
            file_name="ai_health_chat.txt",
            mime="text/plain"
        )
