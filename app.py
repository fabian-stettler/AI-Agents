
import streamlit as st
from micronova_agent import graph

st.set_page_config(page_title="Micronova Agent Chat", page_icon="🤖")
st.title("Micronova Agent Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input("Ask your question..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        # Call your agent here
        result = graph.invoke({"question": prompt})
        answer = result["answer"]
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)