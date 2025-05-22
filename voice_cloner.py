# Copyright @[Phat Nguyen Cong) (https://github.com/paht2005)

# ✅ Feature 3: Voice Cloning
# 🎯 Objective: Generate synthetic speech that mimics the user's voice, using a short reference audio clip.

"""
✍️ Technical Description
This module enables voice cloning, allowing the system to replicate the speaker’s unique vocal characteristics — such as tone, 
pitch, and accent — to produce personalized text-to-speech (TTS) output.

To clone a voice, the system requires:
- A short reference WAV audio clip of the target voice (at least 3–5 seconds, mono, 16kHz).
- A multispeaker TTS model with support for speaker embeddings.

The process involves:
- Extracting the speaker embedding from the reference audio.
- Using that embedding during TTS synthesis to generate speech that sounds like the original speaker.

Coqui TTS's your_tts or other speaker-conditioned models are commonly used for this task, and can run fully offline.

This feature is particularly useful for:
- Creating personalized AI assistants.
- Generating audio content in a consistent speaker voice.
- Accessibility solutions using familiar voices.
"""

from TTS.api import TTS as TTS_Clone

def clone_and_speak(text, speaker_audio, output_file="cloned.wav"): #  Use a .wav file of someone speaking and make sure it’s mono, 16kHz, WAV format
    tts = TTS_Clone(model_name="tts_models/multilingual/multi-dataset/your_tts") # Pick a multi-speaker model, like tts --model_name tts_models/multilingual/multi-dataset/your_tts
    tts.tts_to_file(text=text, speaker_wav=speaker_audio, file_path=output_file)  # Generate speech using the cloned voice

# Output: You’ll get a WAV file ( cloned_voice.wav ) that says your input text in the cloned voice

"""
Key Concepts:
    - Speaker Embedding: Captures vocal tone, pitch, and timbre
    - Voice Transfer: Applies that voice style to arbitrary text
    - Few-shot cloning: Works with as little as 3–5 second
"""


"""
Feature Summary
    - Input: Short voice sample + target text
    - Output: Synthesized audio mimicking the original speaker
    - Framework: Coqui TTS ( your_tts or vits + speaker embedding)
    - Use Cases: Personalized AI, content creation, accessibility
"""