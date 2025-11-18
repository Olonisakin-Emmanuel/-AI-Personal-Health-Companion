# dashboard.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px

CSV_FILE = "session_logs.csv"

def show_dashboard():
    st.title("📊 Predictions Dashboard")
    
    if not os.path.exists(CSV_FILE):
        st.warning("No predictions found yet. Run the Symptom Checker first!")
        return

    df_logs = pd.read_csv(CSV_FILE, quotechar='"', encoding="utf-8", dtype={"session_id": str})
    if df_logs.empty:
        st.warning("No prediction records available.")
        return

    # Show last 5 predictions
    last5 = df_logs.tail(5).sort_values(by="timestamp", ascending=False).reset_index(drop=True)
    last5["session_label"] = [f"Prediction {i+1}" for i in range(len(last5))]

    selected_session = st.selectbox("Select a prediction to view:", last5["session_label"])
    session_data = last5[last5["session_label"] == selected_session].iloc[0]

    # Prediction Details
    st.markdown("### 🩺 Prediction Details")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Timestamp:** {session_data['timestamp']}")
        st.write(f"**Symptoms:** {session_data['symptoms']}")
        st.write(f"**Follow-up Answers:** {session_data['followup_answers']}")
    with col2:
        st.write(f"**Predicted Disease:** {session_data['predicted_disease']}")
        st.write(f"**Confidence:** {session_data['confidence']:.1f}%")
        st.write(f"**Risk:** {session_data['risk']}")
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

    # Quick Summary Card
    st.markdown("---")
    st.markdown(f"""
    <div style="padding:10px; border-radius:8px; background-color:#E3F2FD;">
    <h4 style="color:#0D47A1;">Quick Summary</h4>
    <ul>
        <li><b>Symptoms Count:</b> {len(session_data['symptoms'].split(','))}</li>
        <li><b>Follow-up Questions Answered:</b> {len(eval(session_data['followup_answers']))}</li>
        <li><b>Predicted Disease:</b> {session_data['predicted_disease']}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Analytics: Top Diseases
    st.markdown("---")
    st.markdown("### 📈 Top Predicted Diseases")
    top_diseases = df_logs['predicted_disease'].value_counts().head(5)
    st.bar_chart(top_diseases)
