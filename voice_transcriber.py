# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)


# ✅ Feature 1: Voice Input & Transcription
# 🎯 Objective: Capture voice input from either a microphone or an audio file, and convert it to text using the Whisper model (offline).
# ✍️ Technical Description: This module is responsible for converting spoken language into written text, supporting two types of input: 
          #   Microphone (real-time voice capture) or Audio files (e.g., .wav, .mp3)

import whisper
import speech_recognition as sr

def transcribe_from_file(audio_path): # Audio File Transcription with Whisper
    model = whisper.load_model("base") # or use "small", "medium", "large
    result = model.transcribe(audio_path)
    print("📝Transcribed Text:\n", result["text"])
    return result["text"]

# Call it with: transcribe_from_file("audio_sample.mp3")

def transcribe_from_mic(): # Live Mic Transcription
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except Exception as e:
        return f"[❌] Could not transcribe: {e}"
    
# Call it with: transcribe_from_mic()

def save_transcription(text, output_file="transcript.txt"): # Save Transcription to a File
    with open(output_file, "w") as f:
        f.write(text)
    print(f"✅Saved transcription to {output_file}")

"""
Key Concepts
    - speech_recognition: Works with online Google API (lightweight)
    - whisper: More accurate, multilingual, works offline
    - pydub/ffmpeg: Needed for audio conversion if using MP3s
"""
"""
Feature Summary
    - Input: Microphone or audio file (MP3, WAV)
    - Model: Google STT or OpenAI Whisper
    - Output: Transcribed text (terminal or saved)
    - Use Case: Podcasts, lectures, meetings, accessibility tools
"""

