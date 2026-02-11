import requests 
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain_qdrant import QdrantVectorStore

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
    'I think pears taste very good.',
    'I hate bananas',
    'I dislike raspberries.',
    'I despise mangos.',
    'I love Linux.',
    'I hate Windows'
]

# Qdrant Docker test instead of the classic FAISS.
vector_store = QdrantVectorStore.from_texts(
    texts,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="fruit_collection"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='kb_search',
    description='Search the small product / fruit knowledge base for information.'
)

agent = create_agent(
    model=llm,
    tools=[retriever_tool]
)

result = agent.invoke({
    'messages': [{"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}]
})

print(result["messages"][-1].content)