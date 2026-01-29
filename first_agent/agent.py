import requests 
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

system_prompt = """
You are a weather assistant.

When you use a tool:
- NEVER describe the tool output structure
- NEVER explain JSON or data formats
- ALWAYS produce a final, natural-language answer for the user
- Summarize the information in 1–2 short sentences
- Add one light joke
- SAY THANK YOU AT THE END OF YOUR RESPONSE

The user must only see the final answer, not raw data.
"""

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)


@tool('get_weather', description="Return weather information for a given city.", return_direct = False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

agent = create_agent(
    model = llm,
    tools = [get_weather],
    system_prompt= system_prompt
)

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like in New York City today?'}
    ]
})

#print(response)
print(response['messages'][-1].content)