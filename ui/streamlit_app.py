import streamlit as st
import requests

API_URL = "http://localhost:8080"  # this agent's own API — the restaurant backend owns :8000

st.set_page_config(page_title="AI Agent", layout="centered")
st.title("AI Agent Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "customer_id" not in st.session_state:
    st.session_state.customer_id = None

# Identity is resolved once, here, outside the chat entirely — the agent
# never asks the model who it's talking to (see app/main.py: POST /session).
if st.session_state.customer_id is None:
    st.subheader("Who's dining with us?")
    with st.form("login"):
        phone = st.text_input("Phone")
        email = st.text_input("Email")
        name = st.text_input("Name (only needed if you're new)")
        if st.form_submit_button("Continue") and (phone or email):
            try:
                response = requests.post(
                    f"{API_URL}/session", json={"phone": phone or None, "email": email or None, "name": name or None}
                )
                response.raise_for_status()
                session = response.json()
                st.session_state.customer_id = session["customer_id"]
                st.session_state.customer_name = session["name"]
                st.rerun()
            except Exception as e:
                st.error(f"Couldn't find or create that customer: {e}")
    st.stop()

st.caption(f"Signed in as {st.session_state.get('customer_name', 'customer ' + str(st.session_state.customer_id))}")


def render(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("trace"):
            with st.expander("Chain of thought"):
                for step in msg["trace"]:
                    st.markdown(f"- {step}")
        st.write(msg["content"])


for msg in st.session_state.messages:
    render(msg)

if query := st.chat_input("Ask the agent..."):
    user_msg = {"role": "user", "content": query}
    st.session_state.messages.append(user_msg)
    render(user_msg)

    try:
        response = requests.post(
            f"{API_URL}/agent/run", json={"query": query, "customer_id": st.session_state.customer_id}
        )
        response.raise_for_status()
        result = response.json()
        reply_msg = {
            "role": "assistant",
            "content": result["final_answer"] or f"({result['status']})",
            "trace": result.get("trace", []),
        }
        st.session_state.messages.append(reply_msg)
        render(reply_msg)
    except Exception as e:
        st.error(f"Error: {e}")
