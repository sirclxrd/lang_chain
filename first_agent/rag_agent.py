import requests 
from langchain.tools import tool, ToolRuntime
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest" 
)

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.1
)

texts = [
    'I love apples.',
    'I enjoy oranges.',
    'I think pears tast very good.',
    'I hate banans',
    'I dislike raspberries.',
    'I despite mangos.',
    'I love Linux.',
    'I hate Windows'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

#print(vector_store.similarity_search('I love fruit', k=3))

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='kb_search',
    description='Search the small product / fruit knowledge base for information.')

agent = create_agent(
    model = llm,
    tools = [retriever_tool],
    system_prompt= ("You are a helpful assistant that provides information about fruits and operating systems,"
    "first call the kb_search tool to retrieve context, then answer succintly. Maybe you have to use it" \
    "multiple times before answering.",
    )

)

result = agent.invoke({
    'messages': [{"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}]
})

print(result["messages"][-1].content)