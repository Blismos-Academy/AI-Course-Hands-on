import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# 1. LOAD ENV VARIABLES

load_dotenv()

if "GEMINI_API_KEY" not in os.environ:
    raise ValueError(" GEMINI_API_KEY not found in .env file")

# 2. TOOL DEFINITION (SAME LOGIC)

@tool
def get_current_weather(location: str) -> int:
    """
    Get the current weather for a given location.
    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    return f"The weather in {location} is sunny and 72°F."



# 3. MAIN FUNCTION

def main():

    primary_model = "gemini-2.5-flash"
    fallback_model = "gemini-2.5-pro"

    prompt = "What is the weather like in Bengaluru right now?"

    
    # 4. RETRY + FALLBACK LOOP
    
    for model_name in [primary_model, fallback_model]:
        try:
            print(f"\n Attempting with model: {model_name}")

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.0
            )

            # Bind tool
            llm_with_tools = llm.bind_tools([get_current_weather])

            # Send request
            response = llm_with_tools.invoke(
                [HumanMessage(content=prompt)]
            )

          
          
            if response.tool_calls:
                tool_call = response.tool_calls[0]


                location_arg = tool_call["args"].get("location", "Unknown")

                # Execute tool
                tool_output = get_current_weather.invoke(
                    {"location": location_arg}
                )

                print(f" Tool Output: {tool_output}")

                return

            else:
                print("\n Model responded with text instead of tool call:")
                print(response.content)
                return

        except Exception as e:
            print(f" Error with {model_name}: {e}")
            print(" Switching to fallback model...")
            continue




if __name__ == "__main__":
    main()
