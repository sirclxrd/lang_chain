import requests 
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver


#tool is a decorator to define tools for the agent
#the tools are functions called in the prompt
#ResponseFromat is not supported by OLLAMA
#if i change the thread_id in the config the conversation is reset

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.1
)

checkpointer = InMemorySaver() # linked to a thread id

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

@tool('get_weather', description="Return weather information for a given city.")
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@tool('locate_user', description="Locate the user based on the context.")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'New York City'
        case 'DEF456':
            return 'San Francisco'
        case _:
            return 'Unknown Location'
        
agent = create_agent(
    model = llm,
    tools = [get_weather, locate_user],
    system_prompt= "You are a weather assistant. Use the tools to get the user's location and provide the current weather information.Formulate the response in a simple and clear manner.",
    context_schema=Context,
    checkpointer=checkpointer
)

config = {'configurable': {'thread_id':1}}

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like today?'}
    ]},
    config = config,
    context = Context(user_id = 'DEF456')
)

#print(response)
print(response['messages'][-1].content)

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'And is this usual?'}
    ]},
    config = config,
    context = Context(user_id = 'DEF456')
)

#print(response)
print(response['messages'][-1].content)