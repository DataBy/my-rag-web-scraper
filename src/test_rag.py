import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_pipeline import ask_question

query = "What is Python used for?"
answer = ask_question(query)

print("\nAnswer:", answer)
