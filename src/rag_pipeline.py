import re
from transformers import pipeline
from src.embedder import get_embedding
from src.chroma_manager import get_or_create_collection

# --- Initialize ChromaDB collection ---
collection = get_or_create_collection()

# --- Load language model (PyTorch) ---
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base",
    framework="pt",  # Ensure PyTorch
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2
)

# --- Check if a text contains a number (e.g. 2+ digits) ---
def contains_number(text):
    return bool(re.search(r'\d{2,}', text))

# --- Main QA function ---
def ask_question(query, top_k=6):
    """
    Retrieves relevant context from ChromaDB and uses LLM to generate an answer.
    If the question asks for a number, only number-containing docs are used.
    """
    # Step 1: Embed the question
    query_embedding = get_embedding(query)

    # Step 2: Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    docs = results.get("documents", [[]])[0]

    # Step 3: If question wants a number, filter for number-containing docs
    if "number" in query.lower() or "how many" in query.lower():
        numbered_docs = [doc for doc in docs if contains_number(doc)]
        context = " ".join(numbered_docs) if numbered_docs else "No number found."
    else:
        context = " ".join(docs) if docs else "No relevant context found."

    # Step 4: Build prompt with context and instructions
    prompt = f"""Context: {context}

Instructions: If the question asks for a number, respond ONLY with the number. If no number is found, say "No number found."

Question: {query}
Answer:"""

    # Step 5: Generate answer using LLM
    response = generator(prompt)[0]["generated_text"]
    print("=== MODEL RESPONSE ===")
    print(response)
    return response
