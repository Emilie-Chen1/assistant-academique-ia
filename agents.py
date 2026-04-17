import numexpr
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from tavily import TavilyClient
from urllib.parse import quote
from urllib.request import urlopen
from typing import TypedDict, List, Optional

class AgentResponse(TypedDict):
    content: str
    source: str
    model: str
    tools: List[str]

load_dotenv(find_dotenv())

_tavily = TavilyClient()

_ALLOWED = re.compile(r"^[0-9+\-*/().,%^ \t]+$")
def _validate_expression(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        raise ValueError("expression vide")
    if len(expr) > 200:
        raise ValueError("expression trop longue")
    if not _ALLOWED.match(expr):
        raise ValueError("caractères non autorisés")
    return expr

def _format_tavily(result: dict) -> str:
    items = result.get("results", [])[:5]
    if not items:
        return "Aucun résultat trouvé."
    lines = []
    for i, r in enumerate(items, 1):
        title = r.get("title", "Sans titre")
        url = r.get("url", "URL indisponible")
        content = (r.get("content", "") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title}\n   {url}\n   {content[:220]}")
    return "\n\n".join(lines)

@tool
def current_datetime(timezone: str = "Europe/Paris") -> str:
    """Retourne date et heure actuelles dans un fuseau donné."""
    now = datetime.now(ZoneInfo(timezone))
    return now.strftime("%A %d %B %Y, %H:%M:%S (%Z)")

@tool
def calculator(expression: str) -> str:
    """Calcule une expression mathématique via numexpr."""
    try:
        expression = _validate_expression(expression)
        value = numexpr.evaluate(expression).item()
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
        return _format_tavily(result)
    except Exception as e:
        return f"Erreur recherche web: {e}"

@tool
def weather(city: str) -> str:
    """Donne la météo actuelle pour une ville (Open-Meteo)."""
    try:
        city = city.strip()
        if not city:
            return "Erreur météo: ville vide."

        # 1) Géocodage de la ville -> latitude/longitude
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?name={quote(city)}&count=1&language=fr&format=json"
        )
        with urlopen(geo_url, timeout=10) as resp:
            geo_data = json.loads(resp.read().decode("utf-8"))

        results = geo_data.get("results", [])
        if not results:
            return f"Aucune ville trouvée pour: {city}"

        place = results[0]
        lat = place["latitude"]
        lon = place["longitude"]
        resolved_name = place.get("name", city)
        country = place.get("country", "")

        # 2) Météo actuelle
        meteo_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
            "&timezone=auto"
        )
        with urlopen(meteo_url, timeout=10) as resp:
            meteo_data = json.loads(resp.read().decode("utf-8"))

        current = meteo_data.get("current", {})
        if not current:
            return f"Météo indisponible pour {resolved_name}"

        temp = current.get("temperature_2m", "N/A")
        hum = current.get("relative_humidity_2m", "N/A")
        wind = current.get("wind_speed_10m", "N/A")
        code = current.get("weather_code", "N/A")

        return (
            f"Météo actuelle à {resolved_name}"
            f"{', ' + country if country else ''}: "
            f"{temp}°C, humidité {hum}%, vent {wind} km/h, code météo {code}."
        )
    except Exception as e:
        return f"Erreur météo: {e}"
    
AGENT_MODEL = "openai:gpt-4o-mini"
AGENT_TOOLS = [calculator, web_search, weather, current_datetime]

_agent = create_agent(
    model=AGENT_MODEL,
    tools=AGENT_TOOLS,
    system_prompt=
        "Tu es un assistant utile. "
        "Utilise calculator pour les calculs, weather pour la météo, web_search pour information en temps réel et les recherches web, et current_datetime pour obtenir la date et l'heure actuelles. "
        "Réponds en français.",
    # debug=True
)


def _history_to_openai_messages(history: Optional[List[dict]]) -> List[dict]:
    if not history:
        return []
    out: List[dict] = []
    for m in history:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content")
        if content is None:
            content = ""
        out.append({"role": role, "content": str(content)})
    return out


def agent_answer(question: str, history: list | None = None) -> AgentResponse:
    try:
        messages = _history_to_openai_messages(history)
        messages.append({"role": "user", "content": question})
        result = _agent.invoke({"messages": messages})
        messages = result.get("messages", [])

        if not messages:
            content = "Je n'ai pas pu produire de réponse."
            tools_used = []
        else:
            raw = messages[-1].content
            content = raw if isinstance(raw, str) else str(raw)

            # Récupérer les outils utilisés depuis les ToolMessage
            tools_used = []
            for msg in messages:
                msg_type = type(msg).__name__
                if msg_type == "ToolMessage":
                    tool_name = getattr(msg, "name", None)
                    if tool_name and tool_name not in tools_used:
                        tools_used.append(tool_name)

        return {
            "content": content,
            "source": "agent",
            "model": AGENT_MODEL,
            "tools": tools_used,
        }
    except Exception as e:
        return {
            "content": f"Erreur agent: {e}",
            "source": "agent",
            "model": AGENT_MODEL,
            "tools": [],
        }