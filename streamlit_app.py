import streamlit as st
import requests
from pydub import AudioSegment
from translate import Translator
from gtts import gTTS
import tempfile
import os

# Hugging Face API Details
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
HEADERS = {"Authorization": "Bearer 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"}

def split_text_into_chunks(text, max_length=500):
    """
    Split text into chunks of max_length characters, 
    ensuring chunks split at word boundaries when possible
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for word in text.split():
        # If adding this word would exceed max_length, save current chunk and start new one
        if current_length + len(word) + 1 > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        # Add word to current chunk
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def compress_audio(input_audio_path, target_sample_rate=16000):
    """Compress audio file to specified sample rate"""
    try:
        audio = AudioSegment.from_file(input_audio_path)
        # Convert to mono and downsample to target sample rate
        compressed_audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
        
        # Create a temporary file to save compressed audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            compressed_audio.export(temp_file.name, format="wav")
            return temp_file.name
    except Exception as e:
        st.error(f"Error during audio compression: {e}")
        return input_audio_path

def transcribe_audio(audio_path):
    """Transcribe audio using Hugging Face Whisper API"""
    try:
        with open(audio_path, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=HEADERS, data=data)
        response.raise_for_status()
        return response.json().get('text', 'No transcription available')
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return f"Transcription error: {str(e)}"

def translate_text(text, target_language='es'):
    """Translate text to target language using chunking"""
    try:
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        
        # Translate each chunk
        translator = Translator(to_lang=target_language)
        translated_chunks = []
        
        for chunk in chunks:
            try:
                translated_chunk = translator.translate(chunk)
                translated_chunks.append(translated_chunk)
            except Exception as chunk_error:
                st.warning(f"Error translating chunk: {chunk_error}")
                translated_chunks.append(chunk)  # Fallback to original chunk if translation fails
        
        # Join translated chunks
        return " ".join(translated_chunks)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return f"Translation error: {str(e)}"

def text_to_speech(text, language='es'):
    """Convert text to speech"""
    try:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(temp_file.name)
            return temp_file.name
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

def healthcare_translation_workflow(audio_file, target_language):
    """Complete workflow: compress, transcribe, translate, text-to-speech"""
    try:
        # Map language names to codes
        language_map = {
            'Spanish': 'es', 
            'French': 'fr', 
            'German': 'de', 
            'Italian': 'it', 
            'Portuguese': 'pt'
        }
        language_code = language_map.get(target_language, 'es')
        
        # Step 1: Compress Audio
        compressed_audio = compress_audio(audio_file)
        
        # Step 2: Transcribe
        transcribed_text = transcribe_audio(compressed_audio)
        
        # Step 3: Translate
        translated_text = translate_text(transcribed_text, language_code)
        
        # Step 4: Convert to Speech
        speech_output = text_to_speech(translated_text, language_code)
        
        return transcribed_text, translated_text, speech_output
    except Exception as e:
        st.error(f"Workflow error: {str(e)}")
        return str(e), "", None

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Healthcare Translation Assistant", 
        page_icon="🩺", 
        layout="centered"
    )

    # Title and description
    st.title("🩺 Healthcare Translation Assistant")
    st.write("Upload an audio file to transcribe, translate, and generate speech in your desired language.")

    # Sidebar for language selection
    st.sidebar.header("Translation Settings")
    target_language = st.sidebar.selectbox(
        "Select Target Language",
        [
            'Spanish', 
            'French', 
            'German', 
            'Italian', 
            'Portuguese'
        ]
    )

    # Audio file upload
    uploaded_file = st.file_uploader(
        "Upload Audio File", 
        type=['wav', 'mp3', 'ogg', 'm4a'],
        help="Upload an audio file for transcription and translation"
    )

    # Process button
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Translation button
        if st.button("Transcribe & Translate", type="primary"):
            # Progress indicators
            with st.spinner('Processing audio...'):
                # Run translation workflow
                transcribed_text, translated_text, speech_output = healthcare_translation_workflow(
                    temp_file_path, 
                    target_language
                )

            # Display results
            st.subheader("Results")
            
            # Original Transcription
            st.markdown("**Original Transcription:**")
            st.text_area("Original Text", value=transcribed_text, height=150, disabled=True)
            
            # Translated Text
            st.markdown(f"**Translated Text ({target_language}):**")
            st.text_area("Translated Text", value=translated_text, height=150, disabled=True)
            
            # Audio Playback
            if speech_output:
                st.audio(speech_output, format='audio/mp3')

        # Clean up temporary file
        os.unlink(temp_file_path)

if __name__ == "__main__":
    main()