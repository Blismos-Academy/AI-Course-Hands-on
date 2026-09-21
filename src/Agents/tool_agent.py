import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.tools import tool
from langchain_groq import ChatGroq


load_dotenv()


OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org"
INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")


# ============================================================
# Weather API helpers
# ============================================================

def _validate_api_key() -> None:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is not configured."
        )


def _get_coordinates(location: str) -> dict:
    """Resolve an Indian location to latitude and longitude."""

    _validate_api_key()

    location = location.strip()

    if not location:
        raise ValueError("Location cannot be empty.")

    location = (
        location
        .replace(", India", "")
        .replace(",India", "")
        .strip()
    )

    response = requests.get(
        f"{OPENWEATHER_BASE_URL}/geo/1.0/direct",
        params={
            "q": f"{location},IN",
            "limit": 1,
            "appid": OPENWEATHER_API_KEY,
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    if not data:
        raise ValueError(
            f"Could not find location '{location}' in India."
        )

    place = data[0]

    return {
        "name": place.get("name", location),
        "state": place.get("state"),
        "country": place.get("country"),
        "lat": place["lat"],
        "lon": place["lon"],
    }


def _weather_request(endpoint: str, **params) -> dict:
    """Make a request to OpenWeather."""

    _validate_api_key()

    response = requests.get(
        f"{OPENWEATHER_BASE_URL}{endpoint}",
        params={
            **params,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
        },
        timeout=10,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# Weather tools
# ============================================================

@tool
def get_current_weather(location: str) -> dict:
    """Get the current weather for an Indian location."""

    coordinates = _get_coordinates(location)

    data = _weather_request(
        "/data/2.5/weather",
        lat=coordinates["lat"],
        lon=coordinates["lon"],
    )

    weather = data["weather"][0]
    now = datetime.now(INDIA_TIMEZONE)

    return {
        "location": coordinates["name"],
        "state": coordinates["state"],
        "country": coordinates["country"],
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "Asia/Kolkata",
        "temperature_celsius": data["main"]["temp"],
        "feels_like_celsius": data["main"]["feels_like"],
        "humidity_percent": data["main"]["humidity"],
        "weather": weather["main"],
        "description": weather["description"],
        "wind_speed_mps": data["wind"]["speed"],
        "cloudiness_percent": data["clouds"]["all"],
    }


@tool
def get_weather_forecast(location: str) -> dict:
    """Get the weather forecast for an Indian location."""

    coordinates = _get_coordinates(location)

    data = _weather_request(
        "/data/2.5/forecast",
        lat=coordinates["lat"],
        lon=coordinates["lon"],
    )

    forecast = []

    for item in data.get("list", []):
        forecast.append(
            {
                "date": item["dt_txt"].split()[0],
                "time": item["dt_txt"].split()[1],
                "temperature_celsius": item["main"]["temp"],
                "feels_like_celsius": item["main"]["feels_like"],
                "humidity_percent": item["main"]["humidity"],
                "weather": item["weather"][0]["main"],
                "description": item["weather"][0]["description"],
                "rain_probability": item.get("pop", 0),
                "wind_speed_mps": item["wind"]["speed"],
            }
        )

    now = datetime.now(INDIA_TIMEZONE)

    return {
        "location": coordinates["name"],
        "state": coordinates["state"],
        "country": coordinates["country"],
        "generated_date": now.strftime("%Y-%m-%d"),
        "generated_time": now.strftime("%H:%M:%S"),
        "timezone": "Asia/Kolkata",
        "forecast": forecast,
    }


@tool
def get_current_datetime() -> dict:
    """Get the current date and time in India."""

    now = datetime.now(INDIA_TIMEZONE)

    return {
        "date": now.strftime("%d %B %Y"),
        "time": now.strftime("%I:%M:%S %p"),
        "timezone": "Asia/Kolkata",
        "day": now.strftime("%A"),
    }


weather_tools: list[BaseTool] = [
    get_current_weather,
    get_weather_forecast,
    get_current_datetime,
]


# ============================================================
# Tool Agent
# ============================================================

class ToolAgent:
    """Selects and executes the required external tools."""

    SYSTEM_PROMPT = """
You are the tool execution component of a trip-planning assistant.

Determine which available tools are necessary to answer the user's request.

Rules:
- Execute only the tools that are required.
- Use the correct arguments.
- Do not answer the user.
- Do not invent information.
- Return the actual tool results.
""".strip()

    def __init__(
        self,
        llm: ChatGroq,
        tools: list[BaseTool],
    ) -> None:
        self.tools = {
            tool.name: tool
            for tool in tools
        }

        self.llm = llm.bind_tools(tools, tool_choice="auto")

    def _requires_weather_tool(self, question: str) -> bool:
        """
        Guard the tool-calling model.

        Only requests that clearly need one of the registered weather/date
        tools are allowed to reach the tool-bound LLM. This prevents casual
        conversation, safety refusals, and unrelated requests from producing
        Groq tool_use_failed errors.
        """
        text = question.strip().lower()

        weather_terms = {
            "weather",
            "temperature",
            "forecast",
            "rain",
            "raining",
            "humidity",
            "wind",
            "cloud",
            "cloudy",
            "sunny",
            "hot",
            "cold",
        }

        datetime_phrases = {
            "what time is it",
            "current time",
            "what is the time",
            "what's the time",
            "what date is it",
            "what is the date",
            "today's date",
            "current date",
            "what day is it",
        }

        words = set(
            text.replace("?", " ")
            .replace(",", " ")
            .replace(".", " ")
            .split()
        )

        return bool(words & weather_terms) or any(
            phrase in text for phrase in datetime_phrases
        )

    def invoke(
        self,
        question: str,
        history: list[BaseMessage] | None = None,
    ) -> list[str]:

        # Critical guard: do not call the tool-bound LLM for requests that
        # do not require a weather/date tool.
        if not self._requires_weather_tool(question):
            return []

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            *(history or []),
            HumanMessage(content=question),
        ]

        try:
            response = self.llm.invoke(messages)
        except Exception as exc:
            # Never expose the provider's raw tool_use_failed error to the UI.
            raise RuntimeError(
                "Weather tool execution could not be completed."
            ) from exc

        results: list[str] = []

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args", {})

            tool_instance = self.tools.get(name)

            if tool_instance is None:
                raise ValueError(
                    f"Requested tool '{name}' is not available."
                )

            try:
                result = tool_instance.invoke(args)
            except Exception as exc:
                raise RuntimeError(
                    f"Tool '{name}' failed: {exc}"
                ) from exc

            results.append(str(result))

        return results