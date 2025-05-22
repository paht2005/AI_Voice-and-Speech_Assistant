# 🎙️ AI-powered Voice Assistant Suite (6-in-1 Web App)

An AI Voice Assistant web platform combining **6 smart voice modules** including transcription, voice answering, emotion detection, podcast summarization, document Q&A, and voice cloning — all accessible via a lightweight **Flask web interface**.

![demo](./images/UI.png)

---

## 📌 Table of Contents

1. [✨ Project Overview](#-project-overview)  
2. [🚀 Features](#-features)  
3. [🗂️ Project Structure](#-project-structure)
4. [🧰 Tech Stack](#-tech-stack)
5. [⚙️ Installation & Setup](#-installation-&-setup)  
6. [✅ Feature Details](#-features-details)
7. [🧪 Known Issuess](#-known-issues)
8. [🧭 Future Work](#-future-work)  
9. [📄 License](#-license)
10. [🤝 Contributing](#-contributing)
11. [📬 Contact](#-contact)

---

## ✨ Project Overview

### 1. This voice assistant toolkit empowers users with:
- Transcription from audio to text
- Conversational voice responses
- Real-time Q&A over documents via speech
- Emotion detection via CNN model
- Long podcast summarization
- Voice cloning and TTS using speaker sample
### 2. Accessible through a simple Flask UI for demonstration & prototyping.

---

## 🚀 Features

| Feature Name                         | Description                                                | Technique/Model Used                                  |
|--------------------------------------|------------------------------------------------------------|--------------------------------------------------------
| **Voice Transcription**              | Convert audio (mic/file) into text                         | ``Whisper`` (openai/whisper base)                     |   
| **Text-to-Speech (TTS) Answering**   | Generate voice output from text                            | ``CoquiTTS`` / ``Tacotron2-DDC``                      |
| **Voice Cloning**                    | Clone user voice & read text                               | ``YourTTS`` / ``speaker_wav`` TTS synthesis           |
| **Emotion Detection**                | Detect emotion from voice (e.g., angry, sad)               | ``CNN`` + ``MFCC`` (custom trained on RAVDESS)        |
| **Document Q&A**                     | Ask voice-based questions over documents                   | ``Whisper`` + ``ChromaDB`` + ``SentenceTransformer``  |
| **Podcast Summarizer**               | Transcribe & summarize long podcasts into bullet summary   | ``Whisper`` + ``BART``/``DistilBART`` summarizer      |


---
## 🗂️ Project Structure
```
├── flask_app.py                  # Flask web app
├── __pycache__/  
├── images/                     
├── ravdess-data/                     # RAVDESS dataset
├── templates/                     # HTML interface
├── static/                        # Output audio files
├── uploads/                       # Uploaded inputs
├── voice_transcriber.py           # Feature 1
├── Text_to_Speech_generator.py    # Feature 2
├── voice_cloner.py                # Feature 3
├── emotion_detector.py       # Feature 4 (CNN)
├── voice_rag_agent.py            # Feature 5
├── podcast_summarizer.py         # Feature 6
├── train_emotion_cnn.py          # CNN training script
├── train_emotion_model.py          # RandomForest training script
├── emotion_cnn.pth               # Trained CNN weights
├── emotion_label_encoder.pkl     # Label encoder for emotion
├── requirements-ai.txt                     # Python dependencies
└── README.md
└── LICENSE

```
---

## 🧰 Tech Stack

| Purpose                  | Libraries Used                                        |
|--------------------------|-------------------------------------------------------|
| **Transcription**        | ``whisper``, ``ffmpeg``                               |
| **Text-to-Speech**       | ``TTS``, ``CoquiTTS``                                 |
| **Cloning**              | ``yourTTS``, `sentence-transformers`, `transformers`  |
| **Q&A, Embedding**       | ``EasyOCR``, ``OpenCV``                               |
| **Emotion Detection**    | ``PyTorch``, ``librosa``, `scikit-learn``, `joblib``  |
| **Summarization**        | ``transformers`` (``distilbart-cnn-12-6``)            |
| **Web UI**               | ``Flask`, ``Jinja2`, `HTML5`                          |


---

## ⚙️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/paht2005/ai-voice-assistant-suite.git
cd ai-voice-assistant-suite

# Install dependencies
pip install -r requirements-ai.txt

# Run Flask web app
python flask_app.py


```
Then open your browser: http://127.0.0.1:5000

---
## ✅ Feature Details

### 1.  Voice Transcription
- Uses Whisper model to transcribe audio files or mic input.
- Auto language detection & punctuation recovery.
### 2. TTS Answering
- Enter any text → generate voice using Coqui TTS model.
- Output saved as ``static/tts_output.wav``.
### 3. Voice Cloning
- Upload a 3–5 sec voice sample (.wav).
- Type any text → generates response in that speaker's voice.
### 4. Emotion Detection (CNN)
- Trained using RAVDESS dataset.
- Input ``.wav`` → predicts 1 of 8 emotions.
- Model: ``CNN + MFCC`` → ``emotion_cnn.pth``
### 5. Document Q&A
- Upload voice question (.wav)
- Uses Whisper to transcribe → SentenceTransformer + ChromaDB to retrieve doc context → Falcon or LLM to answer.
### 6.  Podcast Summarization
- Upload long ``.wav`` podcast → splits into chunks → summarizes using BART-based model.
- Summary returned as paragraph.


--- 
## 🧪 Known Issues

| Issue                         | Cause                       | Solution                                                                |
|-------------------------------|-----------------------------|-------------------------------------------------------------------------
| **❗ Whisper FP16 Warningn**  | No GPU                      | Ignore or use GPU for speed                                            |   
| **❌ ``punkt`` not found**    | NLTK missing tokenizer      | Run ``nltk.download('punkt')``                                         |
| **❌ Audio shape mismatchg**  | CNN flatten mismatch        | Use dynamic flatten in CNN                                             |
| **❌ ffmpeg not found**       | Whisper depends on it       | [Install ffmpeg](https://ffmpeg.org/download.html) & add to PATH       |

--- 
## 🧭 Future Work
-  Add real-time streaming voice interface
- WebSocket for fast speech interaction
- Export as REST API (for mobile use)
- Integrate multi-lingual support (Vietnamese, etc.)
- User auth system for personalized interaction
---
## 📄 License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.


---
## 🤝 Contributing
I welcome contributions to improve this project!
Feel free to fork, pull request, or open issues. Ideas welcome!


--- 
## 📬 Contact
- Contact for work: **Nguyễn Công Phát** – congphatnguyen.work@gmail.com
- [Github](https://github.com/paht2005) 
