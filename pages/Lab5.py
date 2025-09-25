import os
import json
import requests
import streamlit as st

# Optional vendors
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# =============================
# 1️⃣ Streamlit Config
# =============================
st.set_page_config(
    page_title="🧥 Lab 5: WeatherWear Bot",
    page_icon="🌦️",
    layout="centered",
)

st.title("🧥 Lab 5: WeatherWear Bot")
st.caption("Enter a city and I'll suggest what to wear based on real-time weather 🌦️")

# =============================
# 2️⃣ API Keys
# =============================
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", os.getenv("OPENWEATHER_API_KEY", ""))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =============================
# 3️⃣ Model Settings
# =============================
MODEL_OPTIONS = {
    "OpenAI": "gpt-5-mini",
    "Anthropic": "claude-sonnet-4-20250514",
    "Google": "gemini-2.5-pro",
}

# =============================
# 4️⃣ Helper Functions
# =============================
def k_to_c(k): return k - 273.15
def k_to_f(k): return (k - 273.15) * 9 / 5 + 32

def get_current_weather(location: str, api_key: str):
    """Fetch weather data from OpenWeather API."""
    if "," in location:
        location = location.split(",")[0].strip()

    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}"
    r = requests.get(url, timeout=10)
    data = r.json()

    if r.status_code != 200 or "main" not in data:
        raise RuntimeError(data.get("message", "Weather fetch failed"))

    tempK = data["main"]["temp"]
    feelsK = data["main"]["feels_like"]
    minK = data["main"]["temp_min"]
    maxK = data["main"]["temp_max"]
    humidity = data["main"]["humidity"]
    desc = (data.get("weather") or [{}])[0].get("description", "n/a")

    return {
        "location": location,
        "temperature_c": round(k_to_c(tempK), 1),
        "temperature_f": round(k_to_f(tempK), 1),
        "feels_like_c": round(k_to_c(feelsK), 1),
        "feels_like_f": round(k_to_f(feelsK), 1),
        "temp_min_c": round(k_to_c(minK), 1),
        "temp_max_c": round(k_to_c(maxK), 1),
        "humidity": int(humidity),
        "description": desc,
    }

# =============================
# 5️⃣ OpenAI: Tool Calling Flow
# =============================

openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather for a city (default: Syracuse NY).",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name."},
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }
]

SYSTEM_PROMPT = """You are a helpful clothing assistant.
When the user asks for weather-based outfit advice, decide if you need to call the weather tool.
If no location is provided, use 'Syracuse NY' as the default.
After you get weather data, you will be called again to give clothing suggestions.
"""

def openai_suggest_outfit(location_input: str):
    client = OpenAI(api_key=OPENAI_API_KEY)

    user_msg = f"User wants outfit suggestions for: '{location_input}'. Use the weather tool if needed."

    # --- Step 1: Tool calling ---
    first = client.chat.completions.create(
        model=MODEL_OPTIONS["OpenAI"],
        tool_choice="auto",
        tools=openai_tools,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    tool_calls = first.choices[0].message.tool_calls or []

    # If no tool called, fallback to default location
    if not tool_calls and not location_input.strip():
        tool_calls = [{
            "id": "default_call",
            "type": "function",
            "function": {"name": "get_current_weather", "arguments": json.dumps({"location": "Syracuse NY"})}
        }]

    # --- Execute tool ---
    weather = None
    for call in tool_calls:
        if call.type == "function" and call.function.name == "get_current_weather":
            args = json.loads(call.function.arguments or "{}")
            loc = args.get("location", "Syracuse NY")
            weather = get_current_weather(loc, OPENWEATHER_API_KEY)
            break

    if weather is None:
        loc = location_input.strip() if location_input.strip() else "Syracuse NY"
        weather = get_current_weather(loc, OPENWEATHER_API_KEY)

    # --- Step 2: Call again with weather info ---
    weather_summary = (
        f"Location: {weather['location']}\n"
        f"Now: {weather['temperature_c']}°C / {weather['temperature_f']}°F\n"
        f"Feels like: {weather['feels_like_c']}°C / {weather['feels_like_f']}°F\n"
        f"Range: {weather['temp_min_c']}–{weather['temp_max_c']}°C\n"
        f"Humidity: {weather['humidity']}%\n"
        f"Conditions: {weather['description']}"
    )

    second = client.chat.completions.create(
        model=MODEL_OPTIONS["OpenAI"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Given this weather, suggest what to wear (tops, bottoms, shoes, accessories):\n\n{weather_summary}"},
        ],
    )

    suggestion = second.choices[0].message.content
    return weather, suggestion

# =============================
# 6️⃣ Anthropic
# =============================

def anthropic_suggest_outfit(location_input: str):
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    loc = location_input.strip() if location_input.strip() else "Syracuse NY"
    weather = get_current_weather(loc, OPENWEATHER_API_KEY)

    weather_summary = (
        f"Location: {weather['location']}\n"
        f"Now: {weather['temperature_c']}°C / {weather['temperature_f']}°F\n"
        f"Feels like: {weather['feels_like_c']}°C / {weather['feels_like_f']}°F\n"
        f"Range: {weather['temp_min_c']}–{weather['temp_max_c']}°C\n"
        f"Humidity: {weather['humidity']}%\n"
        f"Conditions: {weather['description']}"
    )

    response = client.messages.create(
        model=MODEL_OPTIONS["Anthropic"],
        max_tokens=400,
        system="You are a helpful clothing assistant.",
        messages=[
            {"role": "user", "content": f"Given this weather, suggest what to wear:\n\n{weather_summary}"},
        ],
    )

    suggestion = "".join([block.text for block in response.content if hasattr(block, "text")])
    return weather, suggestion

# =============================
# 7️⃣ Gemini
# =============================

def gemini_suggest_outfit(location_input: str):
    loc = location_input.strip() if location_input.strip() else "Syracuse NY"
    weather = get_current_weather(loc, OPENWEATHER_API_KEY)

    weather_summary = (
        f"Location: {weather['location']}\n"
        f"Now: {weather['temperature_c']}°C / {weather['temperature_f']}°F\n"
        f"Feels like: {weather['feels_like_c']}°C / {weather['feels_like_f']}°F\n"
        f"Range: {weather['temp_min_c']}–{weather['temp_max_c']}°C\n"
        f"Humidity: {weather['humidity']}%\n"
        f"Conditions: {weather['description']}"
    )

    prompt = f"Given this weather, suggest what to wear (tops, bottoms, shoes, accessories):\n\n{weather_summary}"
    model = genai.GenerativeModel(MODEL_OPTIONS["Google"])
    response = model.generate_content(prompt)
    return weather, response.text

# =============================
# 8️⃣ Streamlit UI
# =============================

col1, col2 = st.columns([2, 1])
with col1:
    city = st.text_input("🏙️ Enter City", placeholder="e.g., Syracuse NY or London", value="")
with col2:
    vendor = st.selectbox("🤖 Choose Model", ["OpenAI", "Anthropic", "Google"])

if st.button("👕 Get Outfit Suggestion", type="primary"):
    with st.spinner("Fetching weather and generating outfit plan..."):
        try:
            if vendor == "OpenAI":
                weather, suggestion = openai_suggest_outfit(city)
            elif vendor == "Anthropic":
                weather, suggestion = anthropic_suggest_outfit(city)
            else:
                weather, suggestion = gemini_suggest_outfit(city)

            # --- Weather Display ---
            st.markdown("---")
            st.markdown(
                f"""
                <h2 style='text-align:center;'>🌤️ Weather — {weather['location']}</h2>
                <div style='text-align:center; font-size:18px;'>
                <b>Now:</b> {weather['temperature_c']}°C / {weather['temperature_f']}°F &nbsp;&nbsp; 
                <b>Feels like:</b> {weather['feels_like_c']}°C / {weather['feels_like_f']}°F  <br>
                <b>Range:</b> {weather['temp_min_c']}–{weather['temp_max_c']}°C &nbsp;&nbsp; 
                <b>Humidity:</b> {weather['humidity']}% <br>
                <b>Conditions:</b> {weather['description'].title()}
                </div>
                """,
                unsafe_allow_html=True
            )

            # --- Outfit Suggestion ---
            st.markdown("### 👔 Outfit Recommendation")
            st.info(suggestion)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            st.info("Check if your API keys are correctly set in `.streamlit/secrets.toml`.")
