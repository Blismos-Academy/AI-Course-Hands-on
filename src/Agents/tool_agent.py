import os
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org"


def get_coordinates(location: str):
    location = location.replace(", India", "").replace(",India", "")

    params = {
        "q": f"{location},IN",
        "limit": 1,
        "appid": OPENWEATHER_API_KEY
    }

    response = requests.get(
        f"{BASE_URL}/geo/1.0/direct",
        params=params,
        timeout=10
    )
    response.raise_for_status()

    data = response.json()

    if not data:
        raise ValueError(f"Could not find location '{location}' in India.")

    return {
        "name": data[0].get("name"),
        "state": data[0].get("state"),
        "country": data[0].get("country"),
        "lat": data[0]["lat"],
        "lon": data[0]["lon"]
    }


@tool
def get_current_weather(location: str) -> str:
    """Get current weather for an Indian location."""

    coordinates = get_coordinates(location)

    params = {
        "lat": coordinates["lat"],
        "lon": coordinates["lon"],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(
        f"{BASE_URL}/data/2.5/weather",
        params=params,
        timeout=10
    )
    response.raise_for_status()

    data = response.json()
    weather = data["weather"][0]
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    return str({
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
        "cloudiness_percent": data["clouds"]["all"]
    })


@tool
def get_weather_forecast(location: str) -> str:
    """Get weather forecast for an Indian location."""

    coordinates = get_coordinates(location)

    params = {
        "lat": coordinates["lat"],
        "lon": coordinates["lon"],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(
        f"{BASE_URL}/data/2.5/forecast",
        params=params,
        timeout=10
    )
    response.raise_for_status()

    data = response.json()
    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    forecast = []

    for item in data["list"]:
        forecast.append({
            "date": item["dt_txt"].split()[0],
            "time": item["dt_txt"].split()[1],
            "temperature_celsius": item["main"]["temp"],
            "feels_like_celsius": item["main"]["feels_like"],
            "humidity_percent": item["main"]["humidity"],
            "weather": item["weather"][0]["main"],
            "description": item["weather"][0]["description"],
            "rain_probability": item.get("pop", 0),
            "wind_speed_mps": item["wind"]["speed"]
        })

    return str({
        "location": coordinates["name"],
        "state": coordinates["state"],
        "country": coordinates["country"],
        "generated_date": now.strftime("%Y-%m-%d"),
        "generated_time": now.strftime("%H:%M:%S"),
        "timezone": "Asia/Kolkata",
        "forecast": forecast
    })
@tool
def get_current_datetime() -> str:
    """Get the current date and time in India."""

    now = datetime.now(ZoneInfo("Asia/Kolkata"))

    return str({
        "date": now.strftime("%d %B %Y"),
        "time": now.strftime("%I:%M:%S %p"),
        "timezone": "Asia/Kolkata",
        "day": now.strftime("%A")
    })

weather_tools = [
    get_current_weather,
    get_weather_forecast,
    get_current_datetime
]