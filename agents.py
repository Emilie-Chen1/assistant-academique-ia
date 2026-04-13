import numexpr

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv(find_dotenv())

_tavily = TavilyClient()

@tool
def calculator(expression: str) -> str:
    """Calcule une expression mathématique simple."""
    try:
        value = numexpr.evaluate(expression.strip())
        if hasattr(value, "item"):
            value = value.item()  # convertit numpy scalar en float/int
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    except Exception as e:
        return f"Erreur calculatrice: {e}"

@tool
def web_search(query: str) -> str:
    """Recherche web générale via Tavily."""
    try:
        result = _tavily.search(query, max_results=5)
        return str(result)
    except Exception as e:
        return f"Erreur recherche web: {e}"


AGENT_MODEL = "openai:gpt-4o-mini"
AGENT_TOOLS = [calculator, web_search]

_agent = create_agent(
    model=AGENT_MODEL,
    tools=AGENT_TOOLS,
    system_prompt=
        "Tu es un assistant utile. "
        "Utilise calculator pour les calculs et web_search pour les recherches web. "
        "Réponds en français.",
    # debug=True
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