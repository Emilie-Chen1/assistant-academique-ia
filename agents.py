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
            content = "Je n'ai pas pu produire de réponse."
        else:
            raw = messages[-1].content
            content = raw if isinstance(raw, str) else str(raw)

        return {
            "content": content,
            "source": "agent",
            "model": AGENT_MODEL,
            "tools": [getattr(t, "name", str(t)) for t in AGENT_TOOLS],
        }
    except Exception as e:
        return {
            "content": f"Erreur agent: {e}",
            "source": "agent",
            "model": AGENT_MODEL,
            "tools": [getattr(t, "name", str(t)) for t in AGENT_TOOLS],
        }