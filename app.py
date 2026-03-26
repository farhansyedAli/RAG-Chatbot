import gradio as gr
import os
import fitz  # PyMuPDF
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chat_history = []

def extract_text_with_metadata(file_paths):
    text_chunks = []
    metadata = []

    for file_path in file_paths:
        if file_path.name.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                for i, page in enumerate(doc):
                    content = page.get_text()
                    for chunk in split_text(content):
                        text_chunks.append(chunk)
                        metadata.append(f"{file_path.name}, page {i + 1}")
        elif file_path.name.endswith(".docx"):
            doc = docx.Document(file_path)
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    for chunk in split_text(para.text):
                        text_chunks.append(chunk)
                        metadata.append(f"{file_path.name}, section {i + 1}")
    return text_chunks, metadata

def split_text(text, max_length=300):
    sentences = text.split(". ")
    chunks = []
    chunk = ""
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def get_embeddings(chunks):
    return embedding_model.encode(chunks)

def retrieve_relevant_chunks(question, chunks, chunk_embeddings, metadata, top_k=3):
    question_embedding = embedding_model.encode([question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    selected_chunks = [chunks[i] for i in top_indices]
    selected_sources = [metadata[i] for i in top_indices]
    return selected_chunks, selected_sources

def query_groq_llm(question, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer the question using the context below:\n\n{context}\n\nQuestion: {question}"}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()

        if "choices" in result:
            return result['choices'][0]['message']['content']
        elif "error" in result:
            return f"Groq API error: {result['error']['message']}"
        else:
            return "Unexpected response from Groq API."

    except Exception as e:
        return f"Failed to connect to Groq API: {str(e)}"

def rag_chatbot(files, question):
    chunks, metadata = extract_text_with_metadata(files)
    embeddings = get_embeddings(chunks)
    top_chunks, top_sources = retrieve_relevant_chunks(question, chunks, embeddings, metadata)
    context = "\n".join(top_chunks)
    sources = "\n".join([f"• {s}" for s in top_sources])
    answer = query_groq_llm(question, context)
    full_response = f"{answer}\n\n<i><b>Sources:</b><br>{sources}</i>"
    chat_history.append((question, full_response))
    return format_chat_history(chat_history)

def format_chat_history(history):
    formatted = ""
    for q, a in history:
        formatted += f"""
<div style='text-align: right; margin: 10px 0;'><span style='background-color: #333333; padding: 8px 12px; border-radius: 10px; display: inline-block;'>{q}</span></div>
<div style='text-align: left; margin: 10px 0;'><span style='background-color: #333333; padding: 8px 12px; border-radius: 10px; display: inline-block;'>{a}</span></div>
"""
    return formatted

demo = gr.Interface(
    fn=rag_chatbot,
    inputs=[
        gr.File(file_types=[".pdf", ".docx", ".txt"], label="Upload PDF or DOCX", file_count="multiple"),
        gr.Textbox(label="Ask a question")
    ],
    outputs=gr.HTML(label="Chat History"),
    title="📄💬 Enhanced RAG Chatbot (PDF & DOCX) with Groq LLM",
    description="Upload PDF or Word files, ask questions, and see answers with source references. Powered by Groq's LLaMA3 LLM."
)

demo.launch()