import chromadb

#NEW: use PersistentClient for local storage
client = chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection(name="web_docs"):
    return client.get_or_create_collection(name=name)

def add_document_to_chroma(collection, text, doc_id, embedding):
    collection.add(
        documents=[text],
        embeddings=[embedding.tolist()],
        ids=[doc_id]
    )
