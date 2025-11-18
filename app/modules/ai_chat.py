# modules/ai_chat.py
import streamlit as st
import os
from openai import OpenAI

OPENAI_KEY = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

def run_ai_chat():
    st.title("💬 AI Health Chat")
    st.write("Ask simple health questions. This is not a substitute for medical advice.")

    if OPENAI_KEY is None:
        st.warning("OpenAI API key not configured. Add OPENAI_API_KEY to Streamlit secrets or environment to enable chat.")
        return

    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = [{"role":"system","content":"You are a helpful, conservative health assistant. Give general guidance, not diagnoses."}]

    user_input = st.text_input("Ask the AI a question:")

    client = OpenAI(api_key=OPENAI_KEY)

    if user_input:
        st.session_state.ai_chat_history.append({"role":"user","content":user_input})
        with st.spinner("Thinking..."):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.ai_chat_history,
                    max_tokens=400,
                    temperature=0.4
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"Chat failed: {e}")
                return
        st.session_state.ai_chat_history.append({"role":"assistant","content":answer})

    # render chat
    for m in st.session_state.ai_chat_history[1:]:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**AI:** {m['content']}")
