# app.py

import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pdfplumber
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer
from rerankers import Reranker
from transformers import GPT2TokenizerFast
from groq import Groq

import gradio as gr

# Retrieve the API key from environment variables (Hugging Face Secrets)
groq_api_key = os.environ.get('GROQ_API_KEY')

# Initialize the chat client with the API key
chat_client = Groq(api_key=groq_api_key)
model = "llama-3.2-90b-text-preview"

# Define your functions (same as before)
def preprocess_text(text):
    # ... (same as your original function)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def call_Llama_api(query, context):
    # ... (same as your original function)
    chat_completion = chat_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a car technician. Given the user's question and relevant excerpts from different car manuals, answer the question by including direct quotes from the correct car manual. Be concise and to the point in your response."
            },
            {
                "role": "user",
                "content": "User Question: " + query + "\n\nRelevant Excerpt(s):\n\n" + context,
            }
        ],
        temperature=0.7,
        max_tokens=50,
        top_p=1,
        stream=False,
        stop=None,
        model=model
    )
    response = chat_completion.choices[0].message.content
    return response

def is_car_model_available(query, available_models):
    # ... (same as your original function)
    for model in available_models:
        if model.lower() in query.lower():
            return model
    return None

def colbert_rerank(query=None, chunks=None):
    # ... (same as your original function)
    d = ranker.rank(query=query, docs=chunks)
    reranked_chunks = [d[i].text for i in range(len(chunks))]
    return reranked_chunks[:10]

def process_query(query):
    # Use global variables
    global available_car_models, collection

    car_model = is_car_model_available(query, available_car_models)
    if not car_model:
        return "The manual for the specified car model is not present."

    # Initial retrieval from ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=50,
        where={"car_model": car_model},
        include=['documents', 'metadatas']
    )

    if not results['documents']:
        return "No relevant information found in the manual."

    # Extract chunks and metadata
    chunks = results['documents'][0]
    metadatas = results['metadatas'][0]

    reranked_chunks = colbert_rerank(query, chunks)
    final_context = " ".join(reranked_chunks[:10])

    answer = call_Llama_api(query, final_context)

    # Prepare citations
    citations = [
        f"Page {meta.get('page_number', 'N/A')}" for meta in metadatas[:5]
    ]

    citations_text = "Citations:\n" + "\n".join(citations)

    return f"{answer}\n\n{citations_text}"

# Initialize global variables
def initialize():
    global collection, available_car_models, ranker

    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize embedding model
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L12-v2", device=device
    )

    # Load the persisted ChromaDB collection
    client = PersistentClient(path="./chromadb")

    # Get the collection
    collection_name = "car_manuals5"
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Set available car models
    available_car_models = ['TIAGO', 'Astor']

    # Initialize the ranker
    ranker = Reranker("answerdotai/answerai-colbert-small-v1", model_type='colbert')

# Call initialize function
initialize()

# Set up the Gradio interface
iface = gr.Interface(
    fn=process_query,
    inputs=gr.inputs.Textbox(lines=2, placeholder='Enter your question here...'),
    outputs='text',
    title='Car Manual Assistant',
    description='Ask a question about your car manual.',
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
