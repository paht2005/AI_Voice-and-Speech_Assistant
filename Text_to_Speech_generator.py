# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)

# ✅ Feature 2: Text-to-Speech (Voice Answering)
# 🎯 Objective: Convert text-based answers into natural-sounding speech using either ElevenLabs (cloud-based) or Coqui TTS (offline, open-source).



from TTS.api import TTS

def speak_text_offline(text, output_file="output.wav"):
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC") # Load multilingual or English model
    tts.tts_to_file(text=text, file_path=output_file)


"""
Technical Description
-----------
This module transforms the assistant’s textual responses into spoken audio, enhancing interactivity and accessibility through voice-based feedback.
Two implementation paths are supported:
- ElevenLabs API:
    Provides ultra-realistic voice synthesis with expressive prosody and high clarity. Ideal for production-level applications. Requires an internet connection and a valid API key. It supports predefined and custom voices.
- Coqui TTS:
    A fully open-source, offline-capable text-to-speech engine. It supports multiple languages and models, and can be run entirely on local hardware, making it a strong choice for privacy-preserving or edge-device deployments.

Voice synthesis is a critical component of human-computer interaction, and this feature enables the assistant to 
respond audibly—bridging the gap between typed output and real-world voice communication.
"""