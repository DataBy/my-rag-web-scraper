import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chroma_manager import get_or_create_collection, add_document_to_chroma
from src.embedder import get_embedding

text = "Python is a versatile programming language used in AI and web development."
embedding = get_embedding(text)
collection = get_or_create_collection()

add_document_to_chroma(collection, text, doc_id="doc1", embedding=embedding)

print("Document stored in ChromaDB!")
