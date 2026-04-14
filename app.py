import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from rag import rag_answer
from agents import agent_answer

direct_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
)

router_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


def direct_answer(question: str, history: list) -> str:
    messages = [SystemMessage(content="Tu es un assistant utile.")] + [
        HumanMessage(content=msg["content"])
        if msg["role"] == "user"
        else AIMessage(content=msg["content"])
        for msg in history
    ]
    response = direct_llm.invoke(messages + [HumanMessage(content=question)])
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

    response = router_llm.invoke([SystemMessage(content=routing_prompt)])
    route = (
        response.content.strip().lower()
        if hasattr(response, "content")
        else str(response).strip().lower()
    )

    if route not in {"rag", "agent", "direct"}:
        return "direct"

    return route


def get_recent_history(messages, max_turns=6):
    return messages[-max_turns * 2 :]


st.title("🎓 Assistant Académique")

mode = st.sidebar.selectbox(
    "Mode de réponse",
    ["Auto", "RAG", "Agent", "Direct LLM"],
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history = get_recent_history(st.session_state.messages[:-1], max_turns=6)

        if mode == "Auto":
            selected_mode = auto_route(prompt)
        elif mode == "Agent":
            selected_mode = "agent"
        elif mode == "Direct LLM":
            selected_mode = "direct"
        else:
            selected_mode = "rag"

        if selected_mode == "agent":
            result = agent_answer(prompt, history)
            tools_text = ", ".join(result["tools"]) if result["tools"] else "aucun"
            response = (
                f'{result["content"]}\n\n'
                f'_Source: {result["source"]} | Model: {result["model"]} | Tools: {tools_text}_'
            )
        elif selected_mode == "direct":
            response = direct_answer(prompt, history)
            if mode == "Auto":
                response += "\n\n_Source: direct | Model: gpt-4o-mini_"
        else:
            response = rag_answer(prompt, history)
            if mode == "Auto":
                response += "\n\n_Source: rag_"

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})