from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt
from dataclasses import dataclass
from langchain_ollama import ChatOllama


# swap the system prompt based on the level of the expertise.

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.1
)

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    match request.runtime.context.user_role:
        case 'beginner':
            return "You are a helpful assistant that explains concepts in simple terms suitable for beginners."
        case 'intermediate':
            return "You are a knowledgeable assistant that provides detailed explanations suitable for intermediate users."
        case 'expert':
            return "You are an expert assistant that provides in-depth technical explanations suitable for experts."
        case _:
            return "You are a pirate."
        
agent = create_agent(
    model = llm,  
    middleware=[user_role_prompt],
    context_schema=Context
)

response = agent.invoke({
    'messages': [{'role':'user', 'content': 'Explain PCA in 5 sentences.'}]},
     context = Context(user_role = 'yvygy')          
)

print(response['messages'][-1].content)