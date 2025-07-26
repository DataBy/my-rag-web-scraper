import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
from src.scraper import scrape_website
from src.embedder import get_embedding
from src.chroma_manager import get_or_create_collection, add_document_to_chroma
from src.rag_pipeline import ask_question

# Get Chroma collection
collection = get_or_create_collection()

# Step 1: Scrape + embed URL content
def process_url(url):
    text = scrape_website(url)
    if not text:
        return "❌ Failed to scrape content."
    
  
    existing_docs = collection.get()
    if "ids" in existing_docs and existing_docs["ids"]:
        collection.delete(ids=existing_docs["ids"])


    embedding = get_embedding(text)
    doc_id = url.replace("https://", "").replace("http://", "").replace("/", "_")  # simple ID
    add_document_to_chroma(collection, text, doc_id, embedding)
    return " Content scraped and indexed successfully!"

# Step 2: Ask a question
def handle_question(question):
    return ask_question(question)

# 🎨 Custom CSS for glassy/neomorphic look
custom_css = """
body { background: #e0e5ec; font-family: 'Segoe UI', sans-serif; }
.gradio-container { background: rgba(255,255,255,0.35) !important; backdrop-filter: blur(12px); border-radius: 20px; box-shadow: 8px 8px 20px #c5c9ce, -8px -8px 20px #ffffff; padding: 20px; }
textarea, input[type="text"] { background: rgba(255,255,255,0.25); border: none; border-radius: 12px; padding: 10px; box-shadow: inset 2px 2px 5px #c5c9ce, inset -2px -2px 5px #ffffff; }
button { border-radius: 12px !important; background: #ffffff22 !important; box-shadow: 4px 4px 12px #c5c9ce, -4px -4px 12px #ffffff; border: none; }
"""

# Gradio Interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## 🤖 Minimalist AI Webpage Chatbot")

    with gr.Row():
        url_input = gr.Textbox(label="Enter a URL to scrape", placeholder="https://example.com")
        scrape_button = gr.Button("Scrape & Index")

    scrape_output = gr.Textbox(label="Status")

    gr.Markdown("---")

    question_input = gr.Textbox(label="Ask a question about the page")
    answer_output = gr.Textbox(label="Answer")

    ask_button = gr.Button("Ask")

    # Actions
    scrape_button.click(fn=process_url, inputs=[url_input], outputs=[scrape_output])
    ask_button.click(fn=handle_question, inputs=[question_input], outputs=[answer_output])

# Run app
if __name__ == "__main__":
    demo.launch()
