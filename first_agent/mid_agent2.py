from dataclasses import dataclass
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
import time

# middleware is a hook that lets you modify the state of the agent 
# during its execution.

#The Middleware methods are called automatically during the invoke

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.1
)

class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        self.start_time = time.time()
        print('before_agent triggered')

    def before_model(self, state: AgentState, runtime):
        print('before_model')

    def after_model(self, state: AgentState, runtime):
        print('after_model')

    def after_agent(self, state: AgentState, runtime):
        print('after_agent triggered:', time.time() - self.start_time)
        print(state)

agent = create_agent(llm, middleware=[HooksDemo()])

response = agent.invoke({
    'messages': [
        SystemMessage('You are a helpful assistant.'),
        HumanMessage('What is PCA? in 10 words')
    ]
})

print(response['messages'][-1].content)