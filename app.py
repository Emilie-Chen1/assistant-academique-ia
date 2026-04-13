import streamlit as st
from langchain_openai import ChatOpenAI
from rag import rag_answer
from agents import agent_answer

direct_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

def direct_answer(question: str) -> str:
    response = direct_llm.invoke(question)
    return response.content if hasattr(response, "content") else str(response)

st.title("🎓 Assistant Académique")

mode = st.sidebar.selectbox(
    "Mode de réponse",
    ["RAG", "Agent", "Direct LLM"]
)

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
        if mode == "Agent":
            result = agent_answer(prompt)
            tools_text = ", ".join(result["tools"]) if result["tools"] else "aucun"
            response = (
                f'{result["content"]}\n\n'
                f'_Source: {result["source"]} | Model: {result["model"]} | Tools: {tools_text}_'
            )
        elif mode == "Direct LLM":
            response = direct_answer(prompt)
        else:
            response = rag_answer(prompt)
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})