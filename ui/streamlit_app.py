import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Agent", layout="centered")
st.title("AI Agent Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask the agent..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    try:
        response = requests.post(f"{API_URL}/agent/run", json={"query": query})
        response.raise_for_status()
        result = response.json()
        answer = result["final_answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)
    except Exception as e:
        st.error(f"Error: {e}")
