from datasets import load_dataset, Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# -----------------------------------------
# Load dataset
# -----------------------------------------
dataset = load_dataset("hotpot_qa", "fullwiki", split="validation[:100]")
documents = []

for example in dataset:
    for title, sentences in zip(
        example["context"]["title"],
        example["context"]["sentences"]
    ):
        text = " ".join(sentences)
        documents.append(text)

# Remove duplicates
documents = list(set(documents))

# -----------------------------------------
# Init models
# -----------------------------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.1
)

# Wrap per RAGAS (necessario nelle versioni ≥ 0.2)
ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(embeddings)

# -----------------------------------------
# Build Qdrant index
# -----------------------------------------
vector_store = QdrantVectorStore.from_texts(
    documents,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="hotpot_collection"
)

# -----------------------------------------
# Run evaluation loop
# -----------------------------------------
questions = []
answers = []        # risposta generata dal modello  ← FIX: era "references"
contexts_list = []
references = []     # ground truth / riferimento     ← FIX: era "answers"

for example in dataset:
    query = example["question"]
    ground_truth = example["answer"]

    retrieved_docs = vector_store.similarity_search(query, k=10)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""Answer strictly using the provided context.
If the answer is not in the context, say "Not found".

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    generated_answer = response.content  # ← risposta del modello

    # Costruisci il testo di riferimento dai supporting facts
    ref_texts = []
    for title, sent_id in zip(
        example["supporting_facts"]["title"],
        example["supporting_facts"]["sent_id"]
    ):
        for ctx_title, ctx_sents in zip(
            example["context"]["title"],
            example["context"]["sentences"]
        ):
            if ctx_title == title:
                if sent_id < len(ctx_sents):
                    ref_texts.append(ctx_sents[sent_id])
                break  # FIX: interrompi dopo il primo match, evita duplicati

    questions.append(query)
    answers.append(generated_answer)           # ← FIX: risposta del modello
    contexts_list.append([doc.page_content for doc in retrieved_docs])
    references.append(" ".join(ref_texts) if ref_texts else ground_truth)
    # ↑ FIX: fallback al ground_truth se i supporting facts sono vuoti

# -----------------------------------------
# Build RAGAS dataset
# -----------------------------------------
# RAGAS ≥ 0.2 si aspetta:
#   "user_input"      → la domanda
#   "response"        → la risposta generata dal modello
#   "retrieved_contexts" → lista di stringhe di contesto
#   "reference"       → il testo di riferimento (ground truth)
eval_dataset = Dataset.from_dict({
    "user_input": questions,          # FIX: rinominato da "question"
    "response": answers,              # FIX: rinominato da "answer", ora contiene la risposta del modello
    "retrieved_contexts": contexts_list,  # FIX: rinominato da "contexts"
    "reference": references,
})

# -----------------------------------------
# Evaluate
# -----------------------------------------
# FIX: passa llm e embeddings anche alle metriche singole (RAGAS ≥ 0.2)
metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
    ContextRecall(llm=ragas_llm),
]

results = evaluate(
    eval_dataset,
    metrics=metrics,
    llm=ragas_llm,
    embeddings=ragas_emb,
)

print("\nBASELINE HOTPOTQA METRICS:")
print(results)