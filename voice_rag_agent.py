# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)

# ✅ Feature 4: Voice Q&A (RAG Agent)

# 🎯 Objective: Enable users to ask questions via voice input. The system transcribes the query, searches relevant documents, and responds with a contextually accurate answer using a language model.


import whisper
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline

def voice_query_to_answer(audio_path, documents): # audio in .wav format
    # Load Whisper and Transcribe Audio
    model = whisper.load_model("base")  # Or "small", "medium"
    question = model.transcribe(audio_path)["text"]

    #  Set Up a Simple Chroma Vector Store
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    db = chromadb.Client()
    col = db.create_collection("rag")
    for i, doc in enumerate(documents):
        col.add(documents=[doc], embeddings=[embedder.encode(doc).tolist()], ids=[f"doc_{i}"])

    # Retrieve Relevant Context
    query_vector = embedder.encode(question).tolist()
    result = col.query(query_embeddings=[query_vector], n_results=2)
    context = "\n".join(result["documents"][0])

    # Use LLM to Generate Final Answer
    generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")
    prompt = f"Context:\n{context}\n\nQ: {question}\nA:"
    return generator(prompt, max_new_tokens=100)[0]['generated_text']


"""
How It Works: 
    - Whisper transcribes spoken queries
    - ChromaDB + embeddings retrieve relevant facts
    - LLM answers using retrieved knowledge → a full RAG pipeline
"""

"""
Feature Summary:
    - Input: Voice question ( .wav )
    - Output: Generated, context-aware answer
    - Components: Whisper + Embedding Retriever + LLM Generator
    - Use Case: Voice Q&A, hands-free research assistant, voice-controlled chatbo
"""

"""
✍️ Technical Description
-----------
This feature implements a Retrieval-Augmented Generation (RAG) pipeline, integrated with voice input. It allows spoken questions to be processed and answered in natural language with contextual relevance.

1) Workflow Overview:
    a) Speech-to-Text: Use OpenAI Whisper (offline ASR) to transcribe the user’s question from a voice recording.
    b) Document Retrieval:
        - Encode the transcribed query using SentenceTransformers.
        - Search for semantically similar documents in a vector store using ChromaDB.
    c) Answer Generation:
        - Concatenate top matching documents as context.
        - Feed the context and question into a language model (LLM) such as falcon-7b-instruct or GPT for generating the final answer.

2) Key Technologies:
- Whisper: Automatic speech recognition
- ChromaDB: Lightweight document store with vector search
- SentenceTransformers: Text embedding for similarity comparison
- Transformers (LLM): Context-aware text generation

3) This feature is ideal for:
- Hands-free information lookup
- Voice-enabled intelligent agents
- Conversational search assistants
"""