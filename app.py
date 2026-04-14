import streamlit as st
from langchain_openai import ChatOpenAI
from rag import rag_answer
from agents import agent_answer

direct_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3
)

router_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

def direct_answer(question: str) -> str:
    response = direct_llm.invoke(question)
    return response.content if hasattr(response, "content") else str(response)

def auto_route(question: str) -> str:
    routing_prompt = f"""
Tu es un routeur pour un assistant académique.
Tu dois choisir exactement une seule catégorie parmi :
- rag
- agent
- direct

Règles :
- Choisis rag si la question doit être répondue à partir des documents/cours internes.
- Choisis agent si la question nécessite un outil : calcul, météo, recherche web, information en temps réel.
- Choisis direct si une salutation, une reformulation, ou une explication simple qui ne dépend ni des documents ni d'un outil.

Question utilisateur :
{question}

Réponds avec un seul mot parmi :
rag
agent
direct
""".strip()

    response = router_llm.invoke(routing_prompt)
    route = response.content.strip().lower() if hasattr(response, "content") else str(response).strip().lower()

    if route not in {"rag", "agent", "direct"}:
        return "direct"

    return route

st.title("🎓 Assistant Académique")

mode = st.sidebar.selectbox(
    "Mode de réponse",
    ["Auto", "RAG", "Agent", "Direct LLM"]
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
        if mode == "Auto":
            selected_mode = auto_route(prompt)
        elif mode == "Agent":
            selected_mode = "agent"
        elif mode == "Direct LLM":
            selected_mode = "direct"
        else:
            selected_mode = "rag"

        if selected_mode == "agent":
            result = agent_answer(prompt)
            tools_text = ", ".join(result["tools"]) if result["tools"] else "aucun"
            response = (
                f'{result["content"]}\n\n'
                f'_Source: {result["source"]} | Model: {result["model"]} | Tools: {tools_text}_'
            )
        elif selected_mode == "direct":
            response = direct_answer(prompt)
            if mode == "Auto":
                response += "\n\n_Source: direct | Model: gpt-4o-mini_"
        else:
            response = rag_answer(prompt)
            if mode == "Auto":
                response += "\n\n_Source: rag_"
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})