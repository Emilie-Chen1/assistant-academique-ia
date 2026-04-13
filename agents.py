from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent

load_dotenv(find_dotenv())

AGENT_MODEL = "openai:gpt-4o-mini"
AGENT_TOOLS = []


_agent = create_agent(
    model=AGENT_MODEL,
    tools=AGENT_TOOLS,
    system_prompt="Tu es un assistant utile. Réponds en français.",
)


def agent_answer(question: str) -> str:
    try:
        result = _agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result.get("messages", [])
        if not messages:
            return "Je n'ai pas pu produire de réponse."
        content = messages[-1].content
        return content if isinstance(content, str) else str(content)
    except Exception as e:
        return f"Erreur agent: {e}"