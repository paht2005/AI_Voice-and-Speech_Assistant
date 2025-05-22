# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)

# ✅ Feature 6: Podcast Summarizer
# 🎯 Objective: Automatically summarize long-form audio content such as podcasts into concise bullet points or highlights.
import nltk 
nltk.download('punkt') 

import whisper
from nltk.tokenize import sent_tokenize
from transformers import pipeline


def summarize_podcast(audio_path): # audio in .wav format file
    # Transcribe the Podcast
    model = whisper.load_model("base") # Or "medium"/"large" for better accuracy
    text = model.transcribe(audio_path)["text"]
    sentences = sent_tokenize(text)

    # Chunk the Transcript for Long-Form Input
    chunks, current = [], ""
    for sent in sentences:
        if len(current.split()) + len(sent.split()) < 200:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    chunks.append(current.strip())

    # Summarize Each Chunk Using an LLM
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    return "\n\n".join(summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks)



""""
1) Key Concepts:
- Chunking avoids hitting token limits in LLMs
- Summarization models condense each section
- Combines ASR (Whisper) + NLP summarization = complete pipeline
2) Feature Summary:
- Input: Long audio podcast ( .mp3 , .wav )
- Process: Transcribe → Chunk → Summarize
- Output: Concise summary or newsletter content
- Use Case: Podcast producers, note-taking, show notes, accessibility
"""


"""
✍️ Technical Description
This module transforms lengthy podcast episodes into readable summaries, helping users consume audio content quickly and efficiently. It combines speech recognition with NLP summarization techniques.

Step-by-Step Workflow:
1) Audio Transcription:
    - Use Whisper (OpenAI ASR model) to transcribe the entire podcast episode into text.
    - Whisper supports long audio files and multiple languages.
2) Text Chunking
    - Split the full transcript into smaller, manageable chunks based on sentence boundaries (using token count or sentence length).
    - This prevents exceeding token limits when processing with transformer models.
3) Summarization
    - Apply an NLP summarization model such as T5, DistilBART, or Falcon-Instruct on each chunk.
    - Summaries are generated per chunk, then concatenated to form the final summary.
"""