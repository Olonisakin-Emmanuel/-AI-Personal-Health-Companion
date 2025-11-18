# report_analyzer.py
import streamlit as st
from openai import OpenAI
import os

def analyze_medical_report():
    st.title("📄 Medical Report Analyzer")
    st.write("Upload your medical report (PDF or TXT) and get a short summary or insights.")

    # File uploader
    uploaded_file = st.file_uploader("Upload medical report", type=["pdf", "txt"])

    if uploaded_file is not None:
        st.info("Processing your report...")
        content = ""

        if uploaded_file.type == "application/pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                return

        elif uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")

        if content:
            st.write("✅ Report successfully loaded. Generating insights...")

            # Initialize OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            try:
                prompt = f"Analyze the following medical report and provide a concise summary with key findings, in simple language:\n\n{content}"
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.5
                )
                summary = response.choices[0].message.content
                st.subheader("📌 Report Summary / Insights")
                st.write(summary)

            except Exception as e:
                st.error(f"AI analysis failed: {e}")
