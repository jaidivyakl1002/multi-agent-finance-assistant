import os
import io
import asyncio
import tempfile
import logging
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import hashlib
import json
from datetime import datetime, timedelta
import threading
import queue
import time

# FastAPI and async support
from fastapi import UploadFile, HTTPException
import aiofiles

# Audio processing
import whisper
import soundfile as sf
from pydub import AudioSegment
import pyaudio
import wave
import numpy as np

# TTS engines
import pyttsx3
from gtts import gTTS
import pygame

# Caching
from functools import lru_cache
import redis
from typing_extensions import Annotated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceServiceConfig:
    """Configuration for voice service"""
    
    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    MAX_RECORDING_SECONDS = 30
    SILENCE_THRESHOLD = 500
    SILENCE_DURATION = 2.0
    
    # STT settings
    WHISPER_MODEL = "base"  # base, small, medium, large
    STT_LANGUAGE = "en"
    
    # TTS settings
    TTS_RATE = 180  # Words per minute
    TTS_VOLUME = 0.9
    DEFAULT_VOICE_LANG = "en"
    
    # Caching
    CACHE_TTL = 3600  # 1 hour
    MAX_CACHE_SIZE = 100
    
    # File handling
    TEMP_DIR = "./temp_audio"
    SUPPORTED_FORMATS = [".wav", ".mp3", ".m4a", ".ogg", ".flac"]

class AudioProcessor:
    """Handles audio processing and format conversion"""
    
    @staticmethod
    def convert_to_wav(audio_data: bytes, input_format: str = "wav") -> bytes:
        """Convert audio data to WAV format"""
        try:
            # Create AudioSegment from bytes
            if input_format.lower() == "mp3":
                audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
            elif input_format.lower() == "m4a":
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format="m4a")
            elif input_format.lower() in ["ogg", "oga"]:
                audio = AudioSegment.from_ogg(io.BytesIO(audio_data))
            elif input_format.lower() == "flac":
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format="flac")
            else:
                # Assume it's already WAV
                return audio_data
            
            # Convert to WAV
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            return wav_io.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {e}")
            raise HTTPException(status_code=400, detail=f"Audio conversion failed: {str(e)}")
    
    @staticmethod
    def validate_audio_quality(audio_data: bytes) -> Dict[str, Any]:
        """Validate audio quality and provide metrics"""
        try:
            # Load audio for analysis
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            # Analyze with soundfile
            data, sample_rate = sf.read(temp_path)
            
            # Calculate metrics
            duration = len(data) / sample_rate
            rms = np.sqrt(np.mean(data**2))
            max_amplitude = np.max(np.abs(data))
            
            # Clean up
            os.unlink(temp_path)
            
            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "rms_level": float(rms),
                "max_amplitude": float(max_amplitude),
                "is_valid": duration > 0.1 and rms > 0.001,
                "quality_score": min(1.0, rms * 10)  # Simple quality metric
            }
            
        except Exception as e:
            logger.error(f"Error validating audio quality: {e}")
            return {
                "duration": 0,
                "is_valid": False,
                "error": str(e)
            }

class STTService:
    """Speech-to-Text service with caching and async support"""
    
    def __init__(self, config: VoiceServiceConfig):
        self.config = config
        self._whisper_model = None
        self._model_lock = threading.Lock()
        
        # Initialize cache
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=False)
            self.redis_client.ping()
            self.cache_enabled = True
            logger.info("Redis cache connected for STT service")
        except:
            self.cache_enabled = False
            logger.warning("Redis not available, using in-memory cache")
            self._memory_cache = {}
    
    @property
    def whisper_model(self):
        """Lazy loading of Whisper model with thread safety"""
        if self._whisper_model is None:
            with self._model_lock:
                if self._whisper_model is None:
                    logger.info(f"Loading Whisper model: {self.config.WHISPER_MODEL}")
                    self._whisper_model = whisper.load_model(self.config.WHISPER_MODEL)
        return self._whisper_model
    
    def _get_cache_key(self, audio_data: bytes) -> str:
        """Generate cache key from audio data"""
        return f"stt:{hashlib.md5(audio_data).hexdigest()}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """Get cached STT result"""
        try:
            if self.cache_enabled:
                result = self.redis_client.get(cache_key)
                if result:
                    return result.decode('utf-8')
            else:
                return self._memory_cache.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        return None
    
    def _cache_result(self, cache_key: str, result: str):
        """Cache STT result"""
        try:
            if self.cache_enabled:
                self.redis_client.setex(cache_key, self.config.CACHE_TTL, result)
            else:
                if len(self._memory_cache) >= self.config.MAX_CACHE_SIZE:
                    # Simple LRU - remove oldest
                    oldest_key = next(iter(self._memory_cache))
                    del self._memory_cache[oldest_key]
                self._memory_cache[cache_key] = result
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def transcribe_audio(self, audio_data: bytes, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Raw audio bytes
            language: Optional language code (default: auto-detect)
            
        Returns:
            Dict with transcription result and metadata
        """
        try:
            # Generate cache key
            cache_key = self._get_cache_key(audio_data)
            
            # Check cache first
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info("Returning cached STT result")
                return {
                    "text": cached_result,
                    "cached": True,
                    "timestamp": datetime.now().isoformat()
                }
            
            # Validate audio quality
            audio_metrics = AudioProcessor.validate_audio_quality(audio_data)
            if not audio_metrics.get("is_valid", False):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid audio quality: {audio_metrics.get('error', 'Unknown issue')}"
                )
            
            # Create temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Transcribe with Whisper
                logger.info("Starting audio transcription...")
                start_time = time.time()
                
                result = self.whisper_model.transcribe(
                    temp_path,
                    language=language or self.config.STT_LANGUAGE,
                    fp16=False,  # Better compatibility
                    task='transcribe'
                )
                
                transcription_time = time.time() - start_time
                text = result["text"].strip()
                
                # Cache the result
                self._cache_result(cache_key, text)
                
                logger.info(f"Transcription completed in {transcription_time:.2f}s: {text[:100]}...")
                
                return {
                    "text": text,
                    "language": result.get("language", "unknown"),
                    "cached": False,
                    "transcription_time": transcription_time,
                    "audio_duration": audio_metrics.get("duration", 0),
                    "confidence": self._calculate_confidence(result),
                    "timestamp": datetime.now().isoformat()
                }
                
            finally:
                # Clean up temp file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """Calculate confidence score from Whisper result"""
        try:
            # Whisper doesn't provide direct confidence, so we use segment info
            segments = whisper_result.get("segments", [])
            if not segments:
                return 0.8  # Default confidence
            
            # Average the probabilities if available
            probs = []
            for segment in segments:
                if "no_speech_prob" in segment:
                    probs.append(1.0 - segment["no_speech_prob"])
            
            return sum(probs) / len(probs) if probs else 0.8
        except:
            return 0.8
    
    async def transcribe_audio_async(self, audio_data: bytes, language: str = None) -> Dict[str, Any]:
        """Async wrapper for transcribe_audio"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_audio, audio_data, language)

class TTSService:
    """Text-to-Speech service with multiple engines and caching"""
    
    def __init__(self, config: VoiceServiceConfig):
        self.config = config
        self._local_engine = None
        self._engine_lock = threading.Lock()
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Initialize cache
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=False)
            self.redis_client.ping()
            self.cache_enabled = True
            logger.info("Redis cache connected for TTS service")
        except:
            self.cache_enabled = False
            logger.warning("Redis not available for TTS cache")
    
    @property
    def local_engine(self):
        """Lazy loading of local TTS engine"""
        if self._local_engine is None:
            with self._engine_lock:
                if self._local_engine is None:
                    try:
                        self._local_engine = pyttsx3.init()
                        
                        # Configure voice properties
                        voices = self._local_engine.getProperty('voices')
                        if voices:
                            # Prefer female voice
                            for voice in voices:
                                if any(keyword in voice.name.lower() 
                                      for keyword in ['female', 'zira', 'samantha']):
                                    self._local_engine.setProperty('voice', voice.id)
                                    break
                            else:
                                self._local_engine.setProperty('voice', voices[0].id)
                        
                        self._local_engine.setProperty('rate', self.config.TTS_RATE)
                        self._local_engine.setProperty('volume', self.config.TTS_VOLUME)
                        
                        logger.info("Local TTS engine initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize local TTS: {e}")
                        self._local_engine = None
        return self._local_engine
    
    def _get_cache_key(self, text: str, engine: str, voice_settings: Dict) -> str:
        """Generate cache key for TTS"""
        content = f"{text}:{engine}:{json.dumps(voice_settings, sort_keys=True)}"
        return f"tts:{hashlib.md5(content.encode()).hexdigest()}"
    
    def synthesize_speech(self, 
                         text: str, 
                         engine: str = "local",  # "local", "google", "azure"
                         output_format: str = "wav",
                         voice_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to convert to speech
            engine: TTS engine to use
            output_format: Output audio format
            voice_settings: Voice configuration options
            
        Returns:
            Dict with audio data and metadata
        """
        try:
            if not text or not text.strip():
                raise HTTPException(status_code=400, detail="No text provided for TTS")
            
            voice_settings = voice_settings or {}
            
            # Generate cache key
            cache_key = self._get_cache_key(text, engine, voice_settings)
            
            # Check cache
            if self.cache_enabled:
                try:
                    cached_audio = self.redis_client.get(cache_key)
                    if cached_audio:
                        logger.info("Returning cached TTS result")
                        return {
                            "audio_data": cached_audio,
                            "format": output_format,
                            "cached": True,
                            "text_length": len(text),
                            "timestamp": datetime.now().isoformat()
                        }
                except Exception as e:
                    logger.warning(f"TTS cache retrieval error: {e}")
            
            # Generate speech based on engine
            start_time = time.time()
            
            if engine == "google":
                audio_data = self._synthesize_google_tts(text, voice_settings)
            elif engine == "local":
                audio_data = self._synthesize_local_tts(text, voice_settings)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported TTS engine: {engine}")
            
            synthesis_time = time.time() - start_time
            
            # Cache the result
            if self.cache_enabled:
                try:
                    self.redis_client.setex(cache_key, self.config.CACHE_TTL, audio_data)
                except Exception as e:
                    logger.warning(f"TTS cache storage error: {e}")
            
            logger.info(f"TTS synthesis completed in {synthesis_time:.2f}s for {len(text)} characters")
            
            return {
                "audio_data": audio_data,
                "format": output_format,
                "cached": False,
                "synthesis_time": synthesis_time,
                "text_length": len(text),
                "engine": engine,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")
    
    def _synthesize_local_tts(self, text: str, voice_settings: Dict[str, Any]) -> bytes:
        """Synthesize using local pyttsx3 engine"""
        try:
            if not self.local_engine:
                raise Exception("Local TTS engine not available")
            
            # Apply voice settings
            if "rate" in voice_settings:
                self.local_engine.setProperty('rate', voice_settings["rate"])
            if "volume" in voice_settings:
                self.local_engine.setProperty('volume', voice_settings["volume"])
            
            # Generate to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                self.local_engine.save_to_file(text, temp_path)
                self.local_engine.runAndWait()
                
                # Read the generated audio
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()
                
                return audio_data
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Local TTS error: {e}")
            raise
    
    def _synthesize_google_tts(self, text: str, voice_settings: Dict[str, Any]) -> bytes:
        """Synthesize using Google TTS"""
        try:
            lang = voice_settings.get("language", self.config.DEFAULT_VOICE_LANG)
            slow = voice_settings.get("slow", False)
            
            tts = gTTS(text=text, lang=lang, slow=slow)
            
            # Save to temporary file and read back
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
                tts.save(temp_path)
            
            try:
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()
                
                # Convert to WAV if needed
                wav_data = AudioProcessor.convert_to_wav(audio_data, "mp3")
                return wav_data
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Google TTS error: {e}")
            raise
    
    async def synthesize_speech_async(self, text: str, **kwargs) -> Dict[str, Any]:
        """Async wrapper for synthesize_speech"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize_speech, text, **kwargs)

class VoiceService:
    """
    Main Voice Service class that provides unified STT/TTS functionality
    Integrates seamlessly with FastAPI and CrewAI agents
    """
    
    def __init__(self, config: VoiceServiceConfig = None):
        self.config = config or VoiceServiceConfig()
        
        # Ensure temp directory exists
        os.makedirs(self.config.TEMP_DIR, exist_ok=True)
        
        # Initialize services
        self.stt_service = STTService(self.config)
        self.tts_service = TTSService(self.config)
        
        logger.info("Voice Service initialized successfully")
    
    # STT Methods
    async def transcribe_upload(self, file: UploadFile) -> Dict[str, Any]:
        """Transcribe uploaded audio file"""
        try:
            # Validate file format
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in self.config.SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported format: {file_ext}. Supported: {self.config.SUPPORTED_FORMATS}"
                )
            
            # Read file content
            audio_data = await file.read()
            
            # Convert to WAV if needed
            if file_ext != ".wav":
                audio_data = AudioProcessor.convert_to_wav(audio_data, file_ext[1:])
            
            # Transcribe
            result = await self.stt_service.transcribe_audio_async(audio_data)
            result["original_filename"] = file.filename
            
            return result
            
        except Exception as e:
            logger.error(f"Upload transcription error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def transcribe_bytes(self, audio_bytes: bytes, language: str = None) -> Dict[str, Any]:
        """Transcribe raw audio bytes"""
        return self.stt_service.transcribe_audio(audio_bytes, language)
    
    async def transcribe_bytes_async(self, audio_bytes: bytes, language: str = None) -> Dict[str, Any]:
        """Async transcribe raw audio bytes"""
        return await self.stt_service.transcribe_audio_async(audio_bytes, language)
    
    # TTS Methods
    def generate_speech(self, 
                       text: str,
                       engine: str = "local",
                       voice_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate speech from text"""
        return self.tts_service.synthesize_speech(text, engine, voice_settings=voice_settings)
    
    async def generate_speech_async(self, 
                                   text: str,
                                   engine: str = "local",
                                   voice_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Async generate speech from text"""
        return await self.tts_service.synthesize_speech_async(text, engine, voice_settings=voice_settings)
    
    # Combined workflow methods
    async def process_voice_query(self, audio_input: Union[UploadFile, bytes]) -> Dict[str, Any]:
        """Process complete voice query workflow: STT -> Processing -> TTS"""
        try:
            # Step 1: Transcribe audio
            if isinstance(audio_input, UploadFile):
                stt_result = await self.transcribe_upload(audio_input)
            else:
                stt_result = await self.transcribe_bytes_async(audio_input)
            
            transcribed_text = stt_result["text"]
            
            if not transcribed_text.strip():
                raise HTTPException(status_code=400, detail="No speech detected in audio")
            
            logger.info(f"Voice query transcribed: {transcribed_text}")
            
            return {
                "success": True,
                "transcribed_text": transcribed_text,
                "stt_metadata": stt_result,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Voice query processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_voice_response(self, text: str, engine: str = "local") -> Dict[str, Any]:
        """Generate voice response for text"""
        try:
            tts_result = self.generate_speech(text, engine)
            
            return {
                "success": True,
                "audio_data": tts_result["audio_data"],
                "format": tts_result["format"],
                "tts_metadata": tts_result
            }
            
        except Exception as e:
            logger.error(f"Voice response generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Health and utility methods
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            # Test STT service
            stt_healthy = self.stt_service.whisper_model is not None
            
            # Test TTS service
            tts_healthy = True
            try:
                # Quick test synthesis
                test_result = self.tts_service.synthesize_speech("test", "local")
                tts_healthy = len(test_result.get("audio_data", b"")) > 0
            except:
                tts_healthy = False
            
            return {
                "status": "healthy" if (stt_healthy and tts_healthy) else "degraded",
                "stt_service": "healthy" if stt_healthy else "error",
                "tts_service": "healthy" if tts_healthy else "error",
                "cache_enabled": self.stt_service.cache_enabled,
                "whisper_model": self.config.WHISPER_MODEL,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get detailed service statistics"""
        try:
            stats = {
                "config": {
                    "whisper_model": self.config.WHISPER_MODEL,
                    "supported_formats": self.config.SUPPORTED_FORMATS,
                    "sample_rate": self.config.SAMPLE_RATE,
                    "cache_enabled": self.stt_service.cache_enabled
                },
                "health": self.health_check(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add cache stats if available
            if self.stt_service.cache_enabled:
                try:
                    stt_info = self.stt_service.redis_client.info()
                    stats["cache_stats"] = {
                        "redis_version": stt_info.get("redis_version"),
                        "used_memory": stt_info.get("used_memory_human"),
                        "connected_clients": stt_info.get("connected_clients")
                    }
                except:
                    pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats retrieval error: {e}")
            return {"error": str(e)}

# Factory function for easy service creation
def create_voice_service(config: Dict[str, Any] = None) -> VoiceService:
    """Factory function to create voice service with custom config"""
    if config:
        service_config = VoiceServiceConfig()
        for key, value in config.items():
            if hasattr(service_config, key.upper()):
                setattr(service_config, key.upper(), value)
        return VoiceService(service_config)
    return VoiceService()

# Global service instance for use across the application
_voice_service_instance = None

def get_voice_service() -> VoiceService:
    """Get global voice service instance (singleton pattern)"""
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = create_voice_service()
    return _voice_service_instance