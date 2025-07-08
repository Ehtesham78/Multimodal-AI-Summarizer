# 🎥 Multimodal AI Summarizer Agent

A powerful Streamlit web application that summarizes video (YouTube & direct MP4), website content, and website-embedded videos using Google's **Gemini 1.5 Flash** and other AI tools. 
This project combines transcription, translation, web search, and large language model analysis into one unified summarization experience.

---

## 🌟 Features

- 🔗 **YouTube Video Summarization** – Enter a YouTube URL and get automatic transcription, language detection, translation, and AI-driven summarization.
- 🎞️ **Direct MP4 Video Summarization** – Upload or link to a `.mp4` video and get instant analysis.
- 🌐 **Website Content Summarization** – Extracts readable content from websites and summarizes it using Gemini.
- 📽️ **Website Video Summarization** – Finds embedded videos from websites and analyzes them.
- 🎙️ **Speech Recognition** – Audio is extracted and transcribed using `speech_recognition`.
- 🌍 **Language Detection & Translation** – Uses `langdetect` and Google Translate API for multilingual support.
- 🔍 **Web Contextual Search** – Performs DuckDuckGo search to enhance analysis with contextual knowledge.

---

## 🚀 Tech Stack

| Area | Technologies |
|------|--------------|
| Backend | Python, Streamlit |
| AI/ML | Google Gemini 1.5 Flash, Google Translate API, speech_recognition |
| Video Processing | pytube, ffmpeg-python |
| NLP & Utilities | langdetect, DuckDuckGo Search, BeautifulSoup |
| Deployment Ready | Streamlit Cloud / Localhost |

---

## 🛠️ Installation

1. **Clone the repository**

git clone https://github.com/your-username/multimodal-ai-summarizer.git
cd multimodal-ai-summarizer

2. **Create a virtual environment (optional but recommended)**
 python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install the dependencies**
   pip install -r requirements.txt

4. **Set up your environment variables**
   GOOGLE_API_KEY=your_google_api_key_here

5. **Run the app**
   streamlit run app.py

**PROJECT STRUCTURE**
.
├── app.py                # Main Streamlit application
├── .env                  # Environment variables (not committed)
├── requirements.txt      # All required Python packages
├── README.md             # This file

**REQUIREMENTS**
 Python 3.8+
 Google Cloud Translate API Key
 FFmpeg installed and added to PATH
 Stable internet connection (for API calls and video downloads)
