import streamlit as st
from rag import rag_answer
from agents import agent_answer

st.title("🎓 Assistant Académique")

# Historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # LOGIQUE DE ROUTAGE (C'est ton rôle d'intégrateur !)
    with st.chat_message("assistant"):
        if "calcule" in prompt.lower() or "météo" in prompt.lower():
            response = agent_answer(prompt)
        else:
            response = rag_answer(prompt)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})