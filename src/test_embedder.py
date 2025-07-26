import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedder import get_embedding

text = "Wikipedia is a free online encyclopedia."
vector = get_embedding(text)

print("Embedding shape:", vector.shape)
print("First 10 values:", vector[:10])
