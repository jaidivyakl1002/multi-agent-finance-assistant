# streamlit_app/components/audio_recorder.py
import streamlit as st
import streamlit.components.v1 as components
from typing import Optional, Dict, Any
import base64
import io
import json
import uuid
import datetime

class AudioRecorder:
    """
    Streamlit component for browser-based audio recording
    Uses Web Audio API for real-time microphone capture
    """
    
    def __init__(self):
        self.component_html_template = self._get_audio_recorder_html_template()
    
    def _get_audio_recorder_html_template(self) -> str:
        """Generate HTML/JavaScript template for audio recording component"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .audio-recorder {
                    padding: 20px;
                    border: 2px solid #f0f2f6;
                    border-radius: 10px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    text-align: center;
                    margin: 10px 0;
                }
                
                .recorder-button {
                    background: rgba(255, 255, 255, 0.2);
                    border: 2px solid rgba(255, 255, 255, 0.3);
                    border-radius: 50px;
                    color: white;
                    padding: 15px 30px;
                    font-size: 16px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    margin: 10px;
                    backdrop-filter: blur(10px);
                }
                
                .recorder-button:hover {
                    background: rgba(255, 255, 255, 0.3);
                    transform: translateY(-2px);
                    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
                }
                
                .recorder-button:active {
                    transform: translateY(0);
                }
                
                .recording {
                    background: rgba(255, 59, 48, 0.8) !important;
                    animation: pulse 2s infinite;
                }
                
                @keyframes pulse {
                    0% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0.7); }
                    70% { box-shadow: 0 0 0 20px rgba(255, 59, 48, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0); }
                }
                
                .recording-indicator {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 10px;
                    margin: 15px 0;
                    font-weight: 500;
                }
                
                .recording-dot {
                    width: 12px;
                    height: 12px;
                    background: #ff3b30;
                    border-radius: 50%;
                    animation: blink 1s infinite;
                }
                
                @keyframes blink {
                    0%, 50% { opacity: 1; }
                    51%, 100% { opacity: 0.3; }
                }
                
                .audio-visualizer {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 3px;
                    height: 40px;
                    margin: 15px 0;
                }
                
                .visualizer-bar {
                    width: 3px;
                    background: rgba(255, 255, 255, 0.6);
                    border-radius: 2px;
                    transition: height 0.1s ease;
                }
                
                .status-message {
                    margin: 10px 0;
                    font-weight: 500;
                    min-height: 24px;
                }
                
                .error-message {
                    color: #ff6b6b;
                    background: rgba(255, 107, 107, 0.1);
                    padding: 10px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                
                .timer {
                    font-size: 24px;
                    font-weight: 700;
                    margin: 10px 0;
                    font-family: 'Courier New', monospace;
                }
                
                .audio-controls {
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                    margin: 15px 0;
                    flex-wrap: wrap;
                }
                
                .control-button {
                    background: rgba(255, 255, 255, 0.15);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    border-radius: 25px;
                    color: white;
                    padding: 8px 16px;
                    font-size: 14px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                }
                
                .control-button:hover {
                    background: rgba(255, 255, 255, 0.25);
                }
                
                .waveform-container {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    padding: 10px;
                    margin: 10px 0;
                    min-height: 60px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
            </style>
        </head>
        <body>
            <div class="audio-recorder">
                <div id="status" class="status-message">🎤 Ready to record your financial query</div>
                
                <div class="audio-controls">
                    <button id="startBtn" class="recorder-button" onclick="startRecording()">
                        🎙️ Start Recording
                    </button>
                    <button id="stopBtn" class="recorder-button" onclick="stopRecording()" style="display: none;">
                        ⏹️ Stop Recording
                    </button>
                    <button id="playBtn" class="control-button" onclick="playRecording()" style="display: none;">
                        ▶️ Play
                    </button>
                    <button id="downloadBtn" class="control-button" onclick="downloadRecording()" style="display: none;">
                        💾 Download
                    </button>
                    <button id="clearBtn" class="control-button" onclick="clearRecording()" style="display: none;">
                        🗑️ Clear
                    </button>
                </div>
                
                <div id="recordingIndicator" class="recording-indicator" style="display: none;">
                    <div class="recording-dot"></div>
                    <span>Recording in progress...</span>
                </div>
                
                <div id="timer" class="timer" style="display: none;">00:00</div>
                
                <div id="visualizer" class="audio-visualizer" style="display: none;">
                    <!-- Visualizer bars will be generated by JavaScript -->
                </div>
                
                <div id="waveform" class="waveform-container" style="display: none;">
                    <canvas id="waveformCanvas" width="400" height="50"></canvas>
                </div>
                
                <audio id="audioPlayback" controls style="display: none; margin: 10px 0; width: 100%;"></audio>
                
                <div id="error" class="error-message" style="display: none;"></div>
            </div>

            <script>
                let mediaRecorder;
                let audioChunks = [];
                let stream;
                let audioContext;
                let analyser;
                let microphone;
                let dataArray;
                let animationId;
                let recordingStartTime;
                let timerInterval;
                let recordedBlob;
                let componentKey = "{COMPONENT_KEY}";

                // Initialize visualizer
                function initVisualizer() {
                    const visualizer = document.getElementById('visualizer');
                    for (let i = 0; i < 20; i++) {
                        const bar = document.createElement('div');
                        bar.className = 'visualizer-bar';
                        bar.style.height = '5px';
                        visualizer.appendChild(bar);
                    }
                }

                // Update visualizer with audio data
                function updateVisualizer() {
                    if (!analyser) return;
                    
                    analyser.getByteFrequencyData(dataArray);
                    const bars = document.querySelectorAll('.visualizer-bar');
                    
                    for (let i = 0; i < bars.length; i++) {
                        const value = dataArray[i * 4] || 0;
                        const height = Math.max(3, (value / 255) * 40);
                        bars[i].style.height = height + 'px';
                    }
                    
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        animationId = requestAnimationFrame(updateVisualizer);
                    }
                }

                // Start timer
                function startTimer() {
                    recordingStartTime = Date.now();
                    timerInterval = setInterval(() => {
                        const elapsed = Date.now() - recordingStartTime;
                        const minutes = Math.floor(elapsed / 60000);
                        const seconds = Math.floor((elapsed % 60000) / 1000);
                        document.getElementById('timer').textContent = 
                            `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
                    }, 100);
                }

                // Stop timer
                function stopTimer() {
                    if (timerInterval) {
                        clearInterval(timerInterval);
                        timerInterval = null;
                    }
                }

                // Start recording
                async function startRecording() {
                    try {
                        // Clear any previous errors
                        document.getElementById('error').style.display = 'none';
                        
                        // Request microphone access
                        stream = await navigator.mediaDevices.getUserMedia({ 
                            audio: {
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true,
                                sampleRate: 44100
                            } 
                        });

                        // Set up audio analysis
                        audioContext = new (window.AudioContext || window.webkitAudioContext)();
                        analyser = audioContext.createAnalyser();
                        microphone = audioContext.createMediaStreamSource(stream);
                        microphone.connect(analyser);
                        analyser.fftSize = 256;
                        dataArray = new Uint8Array(analyser.frequencyBinCount);

                        // Set up media recorder
                        const options = {
                            mimeType: 'audio/webm;codecs=opus',
                            audioBitsPerSecond: 128000
                        };
                        
                        mediaRecorder = new MediaRecorder(stream, options);
                        audioChunks = [];

                        mediaRecorder.ondataavailable = event => {
                            if (event.data.size > 0) {
                                audioChunks.push(event.data);
                            }
                        };

                        mediaRecorder.onstop = () => {
                            recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                            const audioUrl = URL.createObjectURL(recordedBlob);
                            
                            // Set up audio playback
                            const audioPlayback = document.getElementById('audioPlayback');
                            audioPlayback.src = audioUrl;
                            audioPlayback.style.display = 'block';
                            
                            // Show control buttons
                            document.getElementById('playBtn').style.display = 'inline-block';
                            document.getElementById('downloadBtn').style.display = 'inline-block';
                            document.getElementById('clearBtn').style.display = 'inline-block';
                            
                            // Send audio data to Streamlit
                            sendAudioToStreamlit(recordedBlob);
                        };

                        // Start recording
                        mediaRecorder.start(100); // Collect data every 100ms
                        
                        // Update UI
                        document.getElementById('startBtn').style.display = 'none';
                        document.getElementById('stopBtn').style.display = 'inline-block';
                        document.getElementById('startBtn').classList.add('recording');
                        document.getElementById('recordingIndicator').style.display = 'flex';
                        document.getElementById('timer').style.display = 'block';
                        document.getElementById('visualizer').style.display = 'flex';
                        document.getElementById('status').textContent = '🔴 Recording your financial query...';
                        
                        // Start visualizer and timer
                        updateVisualizer();
                        startTimer();

                    } catch (error) {
                        console.error('Error starting recording:', error);
                        showError('Error accessing microphone: ' + error.message);
                    }
                }

                // Stop recording
                function stopRecording() {
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                    }
                    
                    // Stop all tracks
                    if (stream) {
                        stream.getTracks().forEach(track => track.stop());
                    }
                    
                    // Close audio context
                    if (audioContext) {
                        audioContext.close();
                    }
                    
                    // Stop animation and timer
                    if (animationId) {
                        cancelAnimationFrame(animationId);
                    }
                    stopTimer();
                    
                    // Update UI
                    document.getElementById('startBtn').style.display = 'inline-block';
                    document.getElementById('stopBtn').style.display = 'none';
                    document.getElementById('startBtn').classList.remove('recording');
                    document.getElementById('recordingIndicator').style.display = 'none';
                    document.getElementById('visualizer').style.display = 'none';
                    document.getElementById('status').textContent = '✅ Recording completed! Processing...';
                }

                // Play recorded audio
                function playRecording() {
                    const audioPlayback = document.getElementById('audioPlayback');
                    if (audioPlayback.paused) {
                        audioPlayback.play();
                        document.getElementById('playBtn').textContent = '⏸️ Pause';
                    } else {
                        audioPlayback.pause();
                        document.getElementById('playBtn').textContent = '▶️ Play';
                    }
                }

                // Download recording
                function downloadRecording() {
                    if (recordedBlob) {
                        const url = URL.createObjectURL(recordedBlob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `financial_query_${new Date().toISOString().slice(0, 19)}.webm`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                    }
                }

                // Clear recording
                function clearRecording() {
                    recordedBlob = null;
                    audioChunks = [];
                    
                    // Hide controls and audio player
                    document.getElementById('audioPlayback').style.display = 'none';
                    document.getElementById('playBtn').style.display = 'none';
                    document.getElementById('downloadBtn').style.display = 'none';
                    document.getElementById('clearBtn').style.display = 'none';
                    document.getElementById('waveform').style.display = 'none';
                    
                    // Reset timer
                    document.getElementById('timer').textContent = '00:00';
                    document.getElementById('timer').style.display = 'none';
                    
                    // Reset status
                    document.getElementById('status').textContent = '🎤 Ready to record your financial query';
                    
                    // Store cleared state in session storage with component key
                    try {
                        sessionStorage.setItem('streamlit_audio_' + componentKey, JSON.stringify(null));
                    } catch (e) {
                        console.log('Session storage not available');
                    }
                }

                // Send audio data to Streamlit
                function sendAudioToStreamlit(blob) {
                    const reader = new FileReader();
                    reader.onload = function() {
                        const base64Audio = reader.result.split(',')[1];
                        const audioData = {
                            audio_data: base64Audio,
                            mime_type: blob.type,
                            size: blob.size,
                            duration: recordingStartTime ? (Date.now() - recordingStartTime) / 1000 : 0,
                            timestamp: new Date().toISOString(),
                            component_key: componentKey
                        };
                        
                        // Store in session storage for Streamlit access
                        try {
                            sessionStorage.setItem('streamlit_audio_' + componentKey, JSON.stringify(audioData));
                        } catch (e) {
                            console.log('Session storage not available, using postMessage');
                        }
                        
                        // Also try postMessage as fallback
                        try {
                            window.parent.postMessage({
                                type: 'streamlit:componentValue',
                                value: audioData
                            }, '*');
                        } catch (e) {
                            console.log('PostMessage not available');
                        }
                        
                        document.getElementById('status').textContent = '📤 Audio ready for processing';
                    };
                    reader.readAsDataURL(blob);
                }

                // Show error message
                function showError(message) {
                    const errorDiv = document.getElementById('error');
                    errorDiv.textContent = message;
                    errorDiv.style.display = 'block';
                    document.getElementById('status').textContent = '❌ Recording failed';
                }

                // Initialize on load
                window.onload = function() {
                    initVisualizer();
                    
                    // Check for browser support
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        showError('Your browser does not support audio recording. Please use a modern browser like Chrome, Firefox, or Safari.');
                        document.getElementById('startBtn').disabled = true;
                    }
                    
                    // Handle audio playback events
                    const audioPlayback = document.getElementById('audioPlayback');
                    audioPlayback.addEventListener('ended', () => {
                        document.getElementById('playBtn').textContent = '▶️ Play';
                    });
                    
                    audioPlayback.addEventListener('pause', () => {
                        document.getElementById('playBtn').textContent = '▶️ Play';
                    });
                    
                    audioPlayback.addEventListener('play', () => {
                        document.getElementById('playBtn').textContent = '⏸️ Pause';
                    });
                };

                // Handle keyboard shortcuts
                document.addEventListener('keydown', function(event) {
                    if (event.code === 'Space' && event.ctrlKey) {
                        event.preventDefault();
                        const startBtn = document.getElementById('startBtn');
                        const stopBtn = document.getElementById('stopBtn');
                        
                        if (startBtn.style.display !== 'none') {
                            startRecording();
                        } else if (stopBtn.style.display !== 'none') {
                            stopRecording();
                        }
                    }
                });
            </script>
        </body>
        </html>
        """

    def render(self, key: str = "audio_recorder", height: int = 400) -> Optional[Dict[str, Any]]:
        """
        Render the audio recorder component
        
        Args:
            key: Unique key for this component instance
            height: Height of the component in pixels
            
        Returns:
            Dict containing audio data if recording is available, None otherwise
        """
        # Generate unique component key
        component_key = f"{key}_{hash(key) % 10000}"
        
        # Replace placeholder in HTML template
        component_html = self.component_html_template.replace("{COMPONENT_KEY}", component_key)
        
        # Render the component without the key parameter
        components.html(
            component_html,
            height=height
        )
        
        # Try to get audio data from session state or use alternative approach
        session_key = f"audio_data_{component_key}"
        
        # Check if we have audio data in session state
        if session_key in st.session_state:
            return st.session_state[session_key]
        
        return None

    @staticmethod
    def process_audio_data(audio_data: Dict[str, Any]) -> Optional[bytes]:
        """
        Process audio data returned from the component
        
        Args:
            audio_data: Dictionary containing base64 audio data and metadata
            
        Returns:
            Raw audio bytes or None if processing fails
        """
        if not audio_data or 'audio_data' not in audio_data:
            return None
            
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data['audio_data'])
            return audio_bytes
        except Exception as e:
            st.error(f"Error processing audio data: {str(e)}")
            return None

    @staticmethod
    def create_wav_file(audio_bytes: bytes, filename: str = "recording.wav") -> io.BytesIO:
        """
        Create a WAV file from audio bytes
        
        Args:
            audio_bytes: Raw audio data
            filename: Name for the audio file
            
        Returns:
            BytesIO object containing WAV file data
        """
        try:
            # Create a BytesIO buffer
            wav_buffer = io.BytesIO()
            
            # Note: This is a simplified WAV creation
            # In production, you might want to use a proper audio library
            wav_buffer.write(audio_bytes)
            wav_buffer.seek(0)
            
            return wav_buffer
        except Exception as e:
            st.error(f"Error creating WAV file: {str(e)}")
            return io.BytesIO()

    def render_with_controls(self, key: str = "audio_recorder") -> tuple[Optional[Dict[str, Any]], bool]:
        """
        Render audio recorder with additional Streamlit controls
        
        Args:
            key: Unique key for this component instance
            
        Returns:
            Tuple of (audio_data, process_button_clicked)
        """
        st.markdown("### 🎙️ Voice Recorder")
        st.markdown("*Click 'Start Recording' to record your financial query*")
        
        # Render the component
        audio_data = self.render(key=key)
        
        # Additional controls
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            process_button = st.button("🚀 Process Recording", key=f"{key}_process")
        
        with col2:
            if st.button("🔄 Reset", key=f"{key}_reset"):
                # Clear session state
                session_key = f"audio_data_{key}_{hash(key) % 10000}"
                if session_key in st.session_state:
                    del st.session_state[session_key]
                st.rerun()
        
        with col3:
            if audio_data:
                st.success(f"📡 Recording ready ({audio_data.get('duration', 0):.1f}s)")
            else:
                st.info("🎤 No recording available")
        
        # Alternative text input for testing
        st.markdown("---")
        st.markdown("**Alternative: Upload Audio File**")
        uploaded_file = st.file_uploader(
            "Upload an audio file", 
            type=['wav', 'mp3', 'webm', 'ogg'],
            key=f"{key}_upload"
        )
        
        if uploaded_file:
            st.audio(uploaded_file)
            # Convert uploaded file to our audio_data format
            audio_bytes = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            audio_data = {
                'audio_data': base64.b64encode(audio_bytes).decode('utf-8'),
                'mime_type': uploaded_file.type,
                'size': len(audio_bytes),
                'duration': 0,  # We don't know duration from upload
                'timestamp': datetime.now().isoformat(),
                'source': 'upload'
            }
        
        # Display recording metadata if available
        if audio_data:
            with st.expander("📊 Recording Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duration", f"{audio_data.get('duration', 0):.1f}s")
                    st.metric("Size", f"{audio_data.get('size', 0) / 1024:.1f} KB")
                with col2:
                    st.metric("Format", audio_data.get('mime_type', 'Unknown'))
                    st.metric("Source", audio_data.get('source', 'recording'))
        
        return audio_data, process_button

def create_audio_recorder_demo():
    """
    Demo function to showcase the audio recorder
    """
    st.title("🎙️ Audio Recorder Demo")
    
    recorder = AudioRecorder()
    audio_data, process_clicked = recorder.render_with_controls("demo_recorder")
    
    if process_clicked and audio_data:
        st.success("Processing audio...")
        
        # Process the audio data
        audio_bytes = AudioRecorder.process_audio_data(audio_data)
        
        if audio_bytes:
            st.success(f"Audio processed successfully! ({len(audio_bytes)} bytes)")
            
            # Create download button
            st.download_button(
                label="💾 Download Recording",
                data=audio_bytes,
                file_name=f"recording_{audio_data.get('timestamp', 'unknown')[:19]}.webm",
                mime="audio/webm"
            )
        else:
            st.error("Failed to process audio data")

if __name__ == "__main__":
    create_audio_recorder_demo()