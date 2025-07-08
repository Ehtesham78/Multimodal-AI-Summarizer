import streamlit as st
import google.generativeai as genai
from duckduckgo_search import DDGS
import ffmpeg
import speech_recognition as sr
from langdetect import detect
from google.cloud import translate_v2 as translate
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os
from pytube import YouTube

# Load environment variables
load_dotenv()

# Configure Google API
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
def translate_text(text, target_language, api_key):
    url = f"https://translation.googleapis.com/language/translate/v2"
    params = {
        "q": text,
        "target": target_language,
        "format": "text",
        "key": api_key,
    }
    response = requests.post(url, data=params)
    result = response.json()
    return result["data"]["translations"][0]["translatedText"]
   


# Page configuration
st.set_page_config(
    page_title="Multimodal AI Summarizer",
    page_icon="🎥",
    layout="wide"
)

st.title("Video & Web Summarizer Agent 🎥🎤📜")
st.header("Powered by Gemini 1.5 Flash")

# Initialize Gemini model
@st.cache_resource
def initialize_model():
    return genai.GenerativeModel("gemini-1.5-flash")

model = initialize_model()

# Function to extract audio from video using ffmpeg
def extract_audio_ffmpeg(video_path, audio_path):
    try:
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, acodec="pcm_s16le", ar="16000", ac=1, loglevel="quiet")
        ffmpeg.run(stream)
        return audio_path
    except ffmpeg.Error as e:
        return None, f"Error extracting audio: {e}"

# Function to extract transcript from video
def extract_transcript(video_path, target_language="en"):
    try:
        # Extract audio from video
        audio_path = video_path.rsplit(".", 1)[0] + ".wav"
        result = extract_audio_ffmpeg(video_path, audio_path)
        if result is None:
            return None, "Error extracting audio from video"

        # Initialize recognizer
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)

        # Detect language
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
            detected_language = detect(audio_data.decode("utf-8", errors="ignore"))

        # Transcribe audio
        transcript = recognizer.recognize_google(audio, language=detected_language)
        
        # Translate if target language is different
        if detected_language != target_language:
            transcript = translate_text(transcript, target_language, API_KEY)


        Path(audio_path).unlink(missing_ok=True)
        return transcript, detected_language
    except Exception as e:
        return None, f"Error extracting transcript: {e}"

# Function to fetch and summarize website content
def summarize_website(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract text content
        text_content = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
        
        # Find video URLs
        video_urls = []
        for source in soup.find_all(["source", "video"], src=True):
            src = source.get("src")
            if src and (src.endswith(".mp4") or src.endswith(".mov") or src.endswith(".avi")):
                video_urls.append(urljoin(url, src))

        return text_content, video_urls
    except Exception as e:
        return None, f"Error fetching website: {e}"

# Sidebar for summarization options
st.sidebar.header("Summarization Options")
summarization_type = st.sidebar.selectbox(
    "Choose summarization type",
    ["Video Upload", "Website Content", "Website Video"]
)
language = st.sidebar.selectbox(
    "Select target language for transcript",
    ["en", "es", "fr", "de", "zh", "ja", "hi", "ar"],
    help="Select the language for transcription or translation"
)

# Main content based on summarization type
if summarization_type == "Video Upload":
    st.markdown("### Choose Video Source")

    youtube_url = st.text_input("📺 Enter a YouTube Video URL", placeholder="https://youtube.com/watch?v=...")
    direct_url = st.text_input("🎞️ Enter a Direct MP4 Video URL", placeholder="https://example.com/video.mp4")
    
    video_path = None
    selected_source = None

    if youtube_url:
        try:
            selected_source = "YouTube"
            st.video(youtube_url)
            with st.spinner("Downloading YouTube video..."):
                yt = YouTube(youtube_url)
                stream = yt.streams.filter(file_extension='mp4', progressive=True).order_by('resolution').desc().first()
                if stream:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                        stream.stream_to_buffer(temp_video)
                        video_path = temp_video.name
                else:
                    st.error("No suitable stream found for YouTube video.")
        except Exception as e:
            st.error(f"Failed to download YouTube video: {e}")

    elif direct_url:
        try:
            selected_source = "Direct"
            st.video(direct_url)
            with st.spinner("Downloading direct video..."):
                response = requests.get(direct_url, stream=True, timeout=30)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_video.write(chunk)
                    video_path = temp_video.name
        except Exception as e:
            st.error(f"Failed to download direct video: {e}")

    if video_path:
        st.video(video_path, format="video/mp4", start_time=0)

        # Extract and display transcript
        transcript, detected_language = extract_transcript(video_path, language)
        if transcript:
            st.subheader("Video Transcript")
            st.markdown(f"**Detected Language**: {detected_language}\n\n{transcript}")
        else:
            st.warning(detected_language)  # Error message

        user_query = st.text_area(
            "What insights are you seeking?",
            placeholder="Ask about the video, transcript, or additional context.",
            help="Provide specific questions or insights."
        )

        if st.button("🔍 Analyze Video", key="analyze_video_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze.")
            else:
                try:
                    with st.spinner("Processing video and gathering insights..."):
                        processed_video = genai.upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = genai.get_file(processed_video.name)

                        with DDGS() as ddgs:
                            search_results = list(ddgs.text(user_query, max_results=5))
                        search_context = "\n".join([result["body"] for result in search_results])

                        analysis_prompt = (
                            f"""
                            Analyze the uploaded video and its transcript (Source: {selected_source}):
                            Transcript (Language: {detected_language}, Translated to: {language}):
                            {transcript}

                            Additional context from web search:
                            {search_context}

                            Respond to the following query using video insights, transcript, and web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response in markdown format.
                            """
                        )

                        response = model.generate_content([
                            {"file_data": {"file_uri": processed_video.uri, "mime_type": "video/mp4"}},
                            {"text": analysis_prompt}
                        ])

                        st.subheader("Analysis Result")
                        st.markdown(response.text)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    Path(video_path).unlink(missing_ok=True)

    elif youtube_url or direct_url:
        st.warning("Video could not be processed. Please check the URL.")

        st.video(video_path, format="video/mp4", start_time=0)

        # Extract and display transcript
        transcript, detected_language = extract_transcript(video_path, language)
        if transcript:
            st.subheader("Video Transcript")
            st.markdown(f"**Detected Language**: {detected_language}\n\n{transcript}")
        else:
            st.warning(detected_language)  # Error message

        user_query = st.text_area(
            "What insights are you seeking?",
            placeholder="Ask about the video, transcript, or additional context.",
            help="Provide specific questions or insights."
        )

        if st.button("🔍 Analyze Video", key="analyze_video_button"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze.")
            else:
                try:
                    with st.spinner("Processing video and gathering insights..."):
                        # Upload and process video file
                        processed_video = genai.upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = genai.get_file(processed_video.name)

                        # Perform web search for additional context
                        with DDGS() as ddgs:
                            search_results = list(ddgs.text(user_query, max_results=5))
                        search_context = "\n".join([result["body"] for result in search_results])

                        # Prompt generation
                        analysis_prompt = (
                            f"""
                            Analyze the uploaded video and its transcript:
                            Transcript (Language: {detected_language}, Translated to: {language}):
                            {transcript}

                            Additional context from web search:
                            {search_context}

                            Respond to the following query using video insights, transcript, and web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response in markdown format.
                            """
                        )

                        # Generate response
                        response = model.generate_content([
                            {"file_data": {"file_uri": processed_video.uri, "mime_type": "video/mp4"}},
                            {"text": analysis_prompt}
                        ])

                        # Display result
                        st.subheader("Analysis Result")
                        st.markdown(response.text)

                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    Path(video_path).unlink(missing_ok=True)

elif summarization_type == "Website Content":
    url = st.text_input("Enter website URL", placeholder="https://example.com")
    user_query = st.text_area(
        "What insights are you seeking from the website?",
        placeholder="Ask about the website content.",
        help="Provide specific questions or insights."
    )

    if st.button("🔍 Summarize Website", key="summarize_website_button"):
        if not url or not user_query:
            st.warning("Please enter a valid URL and query.")
        else:
            try:
                with st.spinner("Fetching and summarizing website content..."):
                    text_content, video_urls = summarize_website(url)
                    if text_content:
                        # Perform web search for additional context
                        with DDGS() as ddgs:
                            search_results = list(ddgs.text(user_query, max_results=5))
                        search_context = "\n".join([result["body"] for result in search_results])

                        analysis_prompt = (
                            f"""
                            Analyze the following website content:
                            {text_content}

                            Additional context from web search:
                            {search_context}

                            Respond to the following query using the content and web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response in markdown format.
                            """
                        )
                        response = model.generate_content(analysis_prompt)
                        st.subheader("Website Content Summary")
                        st.markdown(response.text)
                    else:
                        st.error(video_urls)  # Error message
            except Exception as error:
                st.error(f"An error occurred: {error}")

elif summarization_type == "Website Video":
    url = st.text_input("Enter website URL with video", placeholder="https://example.com")
    user_query = st.text_area(
        "What insights are you seeking from the website's video?",
        placeholder="Ask about the video content on the website.",
        help="Provide specific questions or insights."
    )

    if st.button("🔍 Analyze Website Video", key="analyze_website_video_button"):
        if not url or not user_query:
            st.warning("Please enter a valid URL and query.")
        else:
            try:
                with st.spinner("Fetching and analyzing website video..."):
                    text_content, video_urls = summarize_website(url)
                    if video_urls:
                        video_url = video_urls[0]  # Process first video
                        response = requests.get(video_url, stream=True)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                            for chunk in response.iter_content(chunk_size=8192):
                                temp_video.write(chunk)
                            video_path = temp_video.name

                        transcript, detected_language = extract_transcript(video_path, language)
                        processed_video = genai.upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = genai.get_file(processed_video.name)

                        # Perform web search for additional context
                        with DDGS() as ddgs:
                            search_results = list(ddgs.text(user_query, max_results=5))
                        search_context = "\n".join([result["body"] for result in search_results])

                        analysis_prompt = (
                            f"""
                            Analyze the video from the website ({video_url}) and its transcript:
                            Transcript (Language: {detected_language}, Translated to: {language}):
                            {transcript}

                            Additional context from web search:
                            {search_context}

                            Respond to the following query using video insights, transcript, and web research:
                            {user_query}

                            Provide a detailed, user-friendly, and actionable response in markdown format.
                            """
                        )
                        response = model.generate_content([
                            {"file_data": {"file_uri": processed_video.uri, "mime_type": "video/mp4"}},
                            {"text": analysis_prompt}
                        ])
                        st.subheader("Website Video Analysis")
                        if transcript:
                            st.markdown(f"**Transcript**: {transcript}")
                        st.markdown(response.text)
                        Path(video_path).unlink(missing_ok=True)
                    else:
                        st.error("No videos found on the website.")
            except Exception as error:
                st.error(f"An error occurred: {error}")

# Customize UI
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 150px;
    }
    .stSelectbox {
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if summarization_type == "Video Upload" and not (youtube_url or direct_url):
    st.info("Enter a YouTube or Direct MP4 video URL to begin analysis.")
