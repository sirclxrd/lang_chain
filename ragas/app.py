from datasets import Dataset
import os
import faiss
import openai
import numpy as np
from ragas import evaluate
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain_qdrant import QdrantVectorStore
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
    ContextPrecision,
    ContextRecall,
)

# faithfulness = risposta rispetto al contesto
# answer_relevancy = risposta rispetto all'input, quanto bene la risposta matcha l'intento dell'input
# answer_correctness = risposta rispetto al ground truh
# context_precision = misura l'abilita del'retriever di ottenere chunk rilevant rispetto all'input
# context_recall = quanti documenti rilevanti sono stati scelti rispetto a tutti i documenti rilevanti

# i valori delle formule li dà un'altro LLM più grande
# fine-tunato per questo



emb = OllamaEmbeddings(
    model="nomic-embed-text:latest" 
)

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.1
)

ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(emb)

def generate_answer(agent, question):
    result = agent.invoke({
    'messages': [{"role": "user", "content": question}]
    })

    res = result["messages"][-1].content
    return res

def retrieve(retriever, question):
    docs = retriever.invoke(question)
    context = [d.page_content for d in docs]   # IMPORTANT
    return context


docs = [
    "Paris is the capital and most populous city of France. The city is famed for the Eiffel Tower.",
    "Jane Austen was an English novelist best known for 'Pride and Prejudice' and 'Sense and Sensibility'.",
    "The Great Wall of China is a series of fortifications built to protect the ancient Chinese states.",
    "Mount Everest, part of the Himalayas, is Earth’s highest mountain above sea level.",
    "Mike loves the color pink more than any other color."
]

questions = [
    "What is the capital of France?",
    "Who wrote Pride and Prejudice?",
    "Where is Mount Everest located?",
    "What is Mike's favorite color?"
]

ground_truths = [
    "Paris",
    "Jane Austen",
    "the Himalayas",
    "Pink"
]

vector_store = FAISS.from_texts(
    docs,
    embedding=emb
)

# Create retriever (top 3 results)
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='kb_search',
    description='Retrieve context based on question'
)

agent = create_agent(
    model=llm,
    tools=[retriever_tool]
)

rows = []
for question, ground_truth in zip(questions, ground_truths):
    context = retrieve(retriever, question)
    answer = generate_answer(agent, question)
    rows.append(
        {
            "question": question,
            "contexts": context,
            "answer": answer,
            "reference": ground_truth,
        }
    )
evaluation_dataset = Dataset.from_list(rows)

scores = evaluate(
    evaluation_dataset,
    metrics=[
        AnswerCorrectness(),
        AnswerRelevancy(),
        Faithfulness(),
        ContextPrecision(),
        ContextRecall(),
    ],
    llm=ragas_llm,
    embeddings=ragas_emb,
)
print(scores)