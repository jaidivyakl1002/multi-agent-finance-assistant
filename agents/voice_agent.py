"""
Voice Agent for Finance Assistant
Handles Speech-to-Text (STT) and Text-to-Speech (TTS) operations
Uses OpenAI Whisper for STT and various TTS engines
"""

import os
import io
import tempfile
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import asyncio
import numpy as np

from crewai import Agent, Task  # Removed Tool import as it's not used
from crewai_tools import BaseTool
from pydantic import  PrivateAttr, Field

from langchain.chat_models import ChatOpenAI
# STT dependencies
import whisper
import soundfile as sf
from pydub import AudioSegment

# TTS dependencies
import pyttsx3
import gtts
from gtts import gTTS
import pygame

# Audio recording dependencies
import pyaudio
import wave
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioConfig:
    """Audio configuration constants"""
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    RECORD_SECONDS = 30  # Max recording time
    SILENCE_THRESHOLD = 500
    SILENCE_DURATION = 2  # Seconds of silence to stop recording

class VoiceTools:
    """Collection of voice-related tools"""
    
    @staticmethod
    def setup_whisper_model(model_size: str = "base") -> whisper.Whisper:
        """Initialize Whisper model for STT"""
        try:
            model = whisper.load_model(model_size)
            logger.info(f"Loaded Whisper model: {model_size}")
            return model
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    @staticmethod
    def setup_tts_engine() -> pyttsx3.Engine:
        """Initialize local TTS engine"""
        try:
            engine = pyttsx3.init()
            
            # Configure voice properties
            voices = engine.getProperty('voices')
            if voices:
                # Try to use a female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                else:
                    engine.setProperty('voice', voices[0].id)
            
            # Set speech rate and volume
            engine.setProperty('rate', 180)  # Words per minute
            engine.setProperty('volume', 0.9)
            
            logger.info("TTS engine initialized successfully")
            return engine
        except Exception as e:
            logger.error(f"Error initializing TTS engine: {e}")
            raise

class AudioRecorder:
    """Handles audio recording with silence detection"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.frames = []
        
    def start_recording(self) -> bytes:
        """Record audio with automatic silence detection"""
        try:
            stream = self.audio.open(
                format=AudioConfig.FORMAT,
                channels=AudioConfig.CHANNELS,
                rate=AudioConfig.SAMPLE_RATE,
                input=True,
                frames_per_buffer=AudioConfig.CHUNK_SIZE
            )
            
            logger.info("Recording started... Speak now!")
            self.frames = []
            self.is_recording = True
            
            silence_counter = 0
            
            while self.is_recording:
                data = stream.read(AudioConfig.CHUNK_SIZE)
                self.frames.append(data)
                
                # Simple silence detection
                audio_data = np.frombuffer(data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_data**2))
                
                if volume < AudioConfig.SILENCE_THRESHOLD:
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                # Stop if silence detected for specified duration
                if silence_counter > (AudioConfig.SILENCE_DURATION * AudioConfig.SAMPLE_RATE / AudioConfig.CHUNK_SIZE):
                    break
                    
                # Max recording time safety
                if len(self.frames) > (AudioConfig.RECORD_SECONDS * AudioConfig.SAMPLE_RATE / AudioConfig.CHUNK_SIZE):
                    break
            
            stream.stop_stream()
            stream.close()
            
            # Convert frames to audio bytes
            audio_data = b''.join(self.frames)
            logger.info(f"Recording completed. Captured {len(audio_data)} bytes")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            raise
        finally:
            self.is_recording = False
    
    def stop_recording(self):
        """Stop the current recording"""
        self.is_recording = False
    
    def __del__(self):
        """Cleanup audio resources"""
        if hasattr(self, 'audio'):
            self.audio.terminate()

class SpeechToTextTool(BaseTool):
    name: str = Field(default="speech_to_text")
    description: str = Field(default="Convert audio input to text using OpenAI Whisper")
    
    _whisper_model: Any = PrivateAttr()
    _recorder: Any = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._whisper_model = VoiceTools.setup_whisper_model()
        self._recorder = AudioRecorder()
    
    def _run(self, audio_input: Optional[Union[str, bytes]] = None) -> str:
        """
        Convert speech to text
        Args:
            audio_input: Either file path or audio bytes. If None, records from microphone
        """
        try:
            if audio_input is None:
                # Record from microphone
                logger.info("No audio input provided, recording from microphone...")
                audio_data = self._recorder.start_recording()

                # Save to temporary file for Whisper
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    # Convert raw audio to WAV
                    with wave.open(temp_file.name, 'wb') as wav_file:
                        wav_file.setnchannels(AudioConfig.CHANNELS)
                        wav_file.setsampwidth(self._recorder.audio.get_sample_size(AudioConfig.FORMAT))
                        wav_file.setframerate(AudioConfig.SAMPLE_RATE)
                        wav_file.writeframes(audio_data)
                    
                    audio_file_path = temp_file.name
                    
            elif isinstance(audio_input, str):
                # File path provided
                audio_file_path = audio_input
                
            elif isinstance(audio_input, bytes):
                # Audio bytes provided
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_input)
                    audio_file_path = temp_file.name
            else:
                raise ValueError("Invalid audio input type")
            
            # Transcribe using Whisper
            logger.info("Transcribing audio...")
            result = self._whisper_model.transcribe(audio_file_path)
            text = result["text"].strip()
            
            logger.info(f"Transcription completed: {text[:100]}...")
            
            # Cleanup temporary file
            if audio_input is None or isinstance(audio_input, bytes):
                os.unlink(audio_file_path)
            
            return text
            
        except Exception as e:
            logger.error(f"Error in speech-to-text conversion: {e}")
            return f"Error: Could not transcribe audio - {str(e)}"

class TextToSpeechTool(BaseTool):
    name: str = Field(default="text_to_speech")
    description: str = Field(default="Convert text to speech using TTS engines")

    _tts_engine: Any = PrivateAttr()

    def __init__(self):
        super().__init__()
        self._tts_engine = VoiceTools.setup_tts_engine()
        pygame.mixer.init()

    def _run(self, text: str, output_file: Optional[str] = None, use_gtts: bool = False) -> str:
        try:
            if not text or not text.strip():
                return "Error: No text provided for TTS"

            if use_gtts:
                return self._use_google_tts(text, output_file)
            else:
                return self._use_local_tts(text, output_file)

        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            return f"Error: Could not convert text to speech - {str(e)}"

    def _use_local_tts(self, text: str, output_file: Optional[str] = None) -> str:
        try:
            if output_file:
                self._tts_engine.save_to_file(text, output_file)
                self._tts_engine.runAndWait()
                logger.info(f"TTS audio saved to: {output_file}")
                return f"Speech generated and saved to {output_file}"
            else:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
                logger.info("TTS audio played successfully")
                return "Speech generated and played successfully"

        except Exception as e:
            logger.error(f"Error with local TTS: {e}")
            raise

    def _use_google_tts(self, text: str, output_file: Optional[str] = None) -> str:
        try:
            tts = gTTS(text=text, lang='en', slow=False)

            if output_file:
                tts.save(output_file)
                logger.info(f"Google TTS audio saved to: {output_file}")
                return f"Speech generated and saved to {output_file}"
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    tts.save(temp_file.name)
                    pygame.mixer.music.load(temp_file.name)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    os.unlink(temp_file.name)

                logger.info("Google TTS audio played successfully")
                return "Speech generated and played successfully"

        except Exception as e:
            logger.error(f"Error with Google TTS: {e}")
            raise


class VoiceAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.stt_tool = SpeechToTextTool()
        self.tts_tool = TextToSpeechTool()
        
        llm_instance = ChatOpenAI(model_name=model_name)  # ✅ ACTUAL LLM object

        self.agent = Agent(
            role="Voice Interface Specialist",
            goal="Handle all voice input/output operations for the finance assistant",
            backstory="""You are a specialized voice interface agent responsible for 
            converting speech to text and text to speech in a financial analysis system. 
            You ensure clear communication between users and the AI system through voice 
            interactions, handling audio quality issues and providing fallback options 
            when needed.""",
            tools=[self.stt_tool, self.tts_tool],
            verbose=True,
            allow_delegation=False,
            llm=llm_instance  # ✅ FIXED: Use actual LLM object, not a string
        )
    
    def create_stt_task(self, audio_input: Optional[Union[str, bytes]] = None) -> Task:
        """Create a speech-to-text task"""
        return Task(
            description=f"""
            Convert the provided audio input to text using speech recognition.
            
            Audio input: {type(audio_input).__name__ if audio_input else 'microphone recording'}
            
            Requirements:
            1. Use the speech_to_text tool to transcribe the audio
            2. Return the transcribed text clearly
            3. If transcription fails, provide a clear error message
            4. Ensure the output is clean and properly formatted
            """,
            agent=self.agent,
            expected_output="Clean, transcribed text from the audio input"
        )
    
    def create_tts_task(self, text: str, save_to_file: bool = False, 
                       use_google: bool = False) -> Task:
        """Create a text-to-speech task"""
        return Task(
            description=f"""
            Convert the following text to speech:
            
            Text: {text}
            
            Requirements:
            1. Use the text_to_speech tool to generate speech
            2. Save to file: {save_to_file}
            3. Use Google TTS: {use_google}
            4. Ensure clear, natural speech output
            5. Handle any errors gracefully
            """,
            agent=self.agent,
            expected_output="Confirmation that speech was generated successfully"
        )
    
    def process_voice_input(self, audio_input: Optional[Union[str, bytes]] = None) -> str:
        """Process voice input and return transcribed text"""
        try:
            task = self.create_stt_task(audio_input)
            result = self.agent.execute_task(task)
            return result
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            return f"Error processing voice input: {str(e)}"
    
    def generate_voice_output(self, text: str, save_to_file: bool = False, 
                            use_google: bool = False) -> str:
        """Generate voice output from text"""
        try:
            task = self.create_tts_task(text, save_to_file, use_google)
            result = self.agent.execute_task(task)
            return result
        except Exception as e:
            logger.error(f"Error generating voice output: {e}")
            return f"Error generating voice output: {str(e)}"
    
    def handle_voice_interaction(self, audio_input: Optional[Union[str, bytes]] = None) -> Dict[str, Any]:
        """Handle complete voice interaction cycle"""
        try:
            # Step 1: Convert speech to text
            logger.info("Starting voice interaction...")
            transcribed_text = self.process_voice_input(audio_input)
            
            if transcribed_text.startswith("Error"):
                return {
                    "success": False,
                    "transcribed_text": None,
                    "error": transcribed_text
                }
            
            return {
                "success": True,
                "transcribed_text": transcribed_text,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in voice interaction: {e}")
            return {
                "success": False,
                "transcribed_text": None,
                "error": str(e)
            }

# Factory function for easy agent creation
def create_voice_agent(model_name: str = "gpt-3.5-turbo") -> VoiceAgent:
    """Factory function to create a voice agent"""
    return VoiceAgent(model_name=model_name)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the voice agent
    voice_agent = create_voice_agent()
    
    # Test STT
    print("Testing Speech-to-Text...")
    # This would record from microphone in a real scenario
    stt_result = voice_agent.process_voice_input()
    print(f"STT Result: {stt_result}")
    
    # Test TTS
    print("\nTesting Text-to-Speech...")
    test_text = "Hello, this is your finance assistant. Today's market brief is ready."
    tts_result = voice_agent.generate_voice_output(test_text)
    print(f"TTS Result: {tts_result}")
    
    print("\nVoice Agent initialized successfully!")