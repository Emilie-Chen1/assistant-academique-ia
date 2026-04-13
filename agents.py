import numexpr
import json
from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from tavily import TavilyClient
from urllib.parse import quote
from urllib.request import urlopen

load_dotenv(find_dotenv())

_tavily = TavilyClient()

@tool
def calculator(expression: str) -> str:
    """Calcule une expression mathématique via numexpr."""
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
AGENT_TOOLS = [calculator, web_search, weather]

_agent = create_agent(
    model=AGENT_MODEL,
    tools=AGENT_TOOLS,
    system_prompt=
        "Tu es un assistant utile. "
        "Utilise calculator pour les calculs, weather pour la météo, et web_search pour les recherches web. "
        "Réponds en français.",
    # debug=True
)


def agent_answer(question: str) -> str:
    try:
        result = _agent.invoke({"messages": [{"role": "user", "content": question}]})
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