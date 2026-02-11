import requests 
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from base64 import b64encode
from langchain.messages import HumanMessage

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.1
)

message = {
    'role': 'user',
    'content': [
        {'type': 'text', 'text': 'Describe the content of this image.'},
        {'type': 'image', 
         'base64': b64encode(open('imgs/goku.png', 'rb').read()),
         'mime_type': 'image/png'
        }
    ]
}

# message = HumanMessage(
#     content= [
#         {'type': 'text', 'text': 'Describe the content of this image.'},
#         {'type': 'image', 
#          'base64': b64encode(open('imgs/goku.png', 'rb').read()),
#          'mime_type': 'image/png'
#         }
#     ]
# )

response = llm.invoke([message])
print(response.content)
