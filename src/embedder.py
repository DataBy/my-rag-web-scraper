from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np

# Choose a lightweight TF-compatible model (can change later)
MODEL_NAME = "distilbert-base-uncased"

# Load tokenizer and model (TensorFlow version)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModel.from_pretrained(MODEL_NAME)

def get_embedding(text):
    """
    Generate a single embedding vector from a text string.
    """
    # Tokenize and pad/trim the input
    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=512  # Most models limit to 512 tokens
    )

    # Run the model and get the hidden states
    outputs = model(**inputs)

    # Mean pooling: average over token embeddings
    embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1)

    return embedding.numpy()[0]  # Return as 1D numpy array
