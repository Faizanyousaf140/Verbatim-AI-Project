"""
Real-time Audio and Video Recording Module
Modern web-based recording using MediaRecorder API and Streamlit
"""

import streamlit as st
import os
import base64
from datetime import datetime
from typing import Optional, Dict, Any
import tempfile
import uuid

# Handle config import gracefully
try:
    from config import RECORDING_CONFIG
except ImportError:
    # Fallback configuration
    RECORDING_CONFIG = {
        'sample_rate': 44100,
        'channels': 2,
        'max_duration': 3600,
        'audio_format': 'webm',
        'video_format': 'webm',
        'video_quality': 'high',
        'auto_save': True
    }

class WebRecorder:
    """Modern web-based audio and video recorder using MediaRecorder API"""
    
    def __init__(self):
        self.recordings_dir = "meetings"
        self.ensure_recordings_directory()
    
    def ensure_recordings_directory(self):
        """Ensure recordings directory exists"""
        os.makedirs(self.recordings_dir, exist_ok=True)
    
    def save_recording(self, recording_data: bytes, recording_type: str, filename: str = None) -> str:
        """Save recording data to file"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                extension = RECORDING_CONFIG.get(f'{recording_type}_format', 'webm')
                filename = f"{recording_type}_recording_{timestamp}_{unique_id}.{extension}"
            
            file_path = os.path.join(self.recordings_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(recording_data)
            
            return file_path
            
        except Exception as e:
            st.error(f"Error saving {recording_type} recording: {e}")
            return None
    
    def get_recording_stats(self, file_path: str) -> Dict[str, Any]:
        """Get recording file statistics"""
        try:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                created = datetime.fromtimestamp(os.path.getctime(file_path))
                return {
                    'file_path': file_path,
                    'size_mb': round(size / (1024 * 1024), 2),
                    'created': created.strftime("%Y-%m-%d %H:%M:%S"),
                    'exists': True
                }
        except Exception as e:
            st.error(f"Error getting file stats: {e}")
        
        return {'exists': False}

def get_audio_recorder_html():
    """Generate HTML for audio recording using MediaRecorder API"""
    return """
    <div id="audio-recorder" style="padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; margin: 10px 0;">
        <h3>🎙️ Audio Recorder</h3>
        <div style="margin: 10px 0;">
            <button id="start-audio" onclick="startAudioRecording()" style="background: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
                🔴 Start Audio Recording
            </button>
            <button id="pause-audio" onclick="pauseAudioRecording()" disabled style="background: #FF9800; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
                ⏸️ Pause
            </button>
            <button id="stop-audio" onclick="stopAudioRecording()" disabled style="background: #f44336; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
                ⏹️ Stop Recording
            </button>
        </div>
        <div id="audio-status" style="margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px;">
            Status: Ready to record
        </div>
        <div id="audio-timer" style="margin: 10px 0; font-weight: bold; font-size: 18px; color: #333;">
            Duration: 00:00
        </div>
        <audio id="audio-playback" controls style="width: 100%; margin: 10px 0;" hidden></audio>
        <div id="audio-download" style="margin: 10px 0;" hidden>
            <a id="audio-download-link" download="audio-recording.webm" style="background: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                📥 Download Audio Recording
            </a>
        </div>
    </div>

    <script>
    let audioRecorder = null;
    let audioChunks = [];
    let audioTimer = null;
    let audioStartTime = null;
    let audioPaused = false;
    let audioPauseTime = 0;

    function updateAudioTimer() {
        if (audioStartTime && !audioPaused) {
            const elapsed = Math.floor((Date.now() - audioStartTime - audioPauseTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            document.getElementById('audio-timer').textContent = 
                `Duration: ${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    async function startAudioRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100
                } 
            });
            
            audioRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            audioChunks = [];
            audioStartTime = Date.now();
            audioPauseTime = 0;
            audioPaused = false;
            
            audioRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            audioRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const audioUrl = URL.createObjectURL(audioBlob);
                
                document.getElementById('audio-playback').src = audioUrl;
                document.getElementById('audio-playback').hidden = false;
                document.getElementById('audio-download-link').href = audioUrl;
                document.getElementById('audio-download').hidden = false;
                
                // Convert to base64 and send to Streamlit
                const reader = new FileReader();
                reader.onload = function() {
                    const base64Data = reader.result.split(',')[1];
                    window.parent.postMessage({
                        type: 'audio_recording',
                        data: base64Data,
                        filename: `audio_${new Date().getTime()}.webm`
                    }, '*');
                };
                reader.readAsDataURL(audioBlob);
                
                stream.getTracks().forEach(track => track.stop());
            };
            
            audioRecorder.start(1000); // Collect data every second
            
            document.getElementById('start-audio').disabled = true;
            document.getElementById('pause-audio').disabled = false;
            document.getElementById('stop-audio').disabled = false;
            document.getElementById('audio-status').textContent = 'Status: 🔴 Recording...';
            document.getElementById('audio-playback').hidden = true;
            document.getElementById('audio-download').hidden = true;
            
            audioTimer = setInterval(updateAudioTimer, 1000);
            
        } catch (error) {
            document.getElementById('audio-status').textContent = 'Error: Could not access microphone - ' + error.message;
        }
    }

    function pauseAudioRecording() {
        if (audioRecorder && audioRecorder.state === 'recording') {
            audioRecorder.pause();
            audioPaused = true;
            audioPauseTime += Date.now() - audioStartTime;
            document.getElementById('audio-status').textContent = 'Status: ⏸️ Paused';
            document.getElementById('pause-audio').textContent = '▶️ Resume';
            document.getElementById('pause-audio').onclick = resumeAudioRecording;
        }
    }

    function resumeAudioRecording() {
        if (audioRecorder && audioRecorder.state === 'paused') {
            audioRecorder.resume();
            audioPaused = false;
            audioStartTime = Date.now();
            document.getElementById('audio-status').textContent = 'Status: 🔴 Recording...';
            document.getElementById('pause-audio').textContent = '⏸️ Pause';
            document.getElementById('pause-audio').onclick = pauseAudioRecording;
        }
    }

    function stopAudioRecording() {
        if (audioRecorder && audioRecorder.state !== 'inactive') {
            audioRecorder.stop();
            clearInterval(audioTimer);
            
            document.getElementById('start-audio').disabled = false;
            document.getElementById('pause-audio').disabled = true;
            document.getElementById('stop-audio').disabled = true;
            document.getElementById('pause-audio').textContent = '⏸️ Pause';
            document.getElementById('pause-audio').onclick = pauseAudioRecording;
            document.getElementById('audio-status').textContent = 'Status: ✅ Recording completed';
        }
    }
    </script>
    """

def get_video_recorder_html():
    """Generate HTML for video recording using MediaRecorder API"""
    return """
    <div id="video-recorder" style="padding: 20px; border: 2px solid #2196F3; border-radius: 10px; margin: 10px 0;">
        <h3>🎥 Video Recorder</h3>
        <div style="margin: 10px 0;">
            <button id="start-video" onclick="startVideoRecording()" style="background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
                🔴 Start Video Recording
            </button>
            <button id="pause-video" onclick="pauseVideoRecording()" disabled style="background: #FF9800; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
                ⏸️ Pause
            </button>
            <button id="stop-video" onclick="stopVideoRecording()" disabled style="background: #f44336; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer;">
                ⏹️ Stop Recording
            </button>
        </div>
        <div id="video-status" style="margin: 10px 0; padding: 10px; background: #f0f0f0; border-radius: 5px;">
            Status: Ready to record
        </div>
        <div id="video-timer" style="margin: 10px 0; font-weight: bold; font-size: 18px; color: #333;">
            Duration: 00:00
        </div>
        <div style="display: flex; gap: 20px; margin: 10px 0;">
            <div style="flex: 1;">
                <h4>📹 Live Preview</h4>
                <video id="camera-preview" autoplay muted playsinline style="width: 100%; max-width: 400px; border: 1px solid #ccc; border-radius: 5px;"></video>
            </div>
            <div style="flex: 1;">
                <h4>▶️ Recording Playback</h4>
                <video id="video-playback" controls style="width: 100%; max-width: 400px; border: 1px solid #ccc; border-radius: 5px;" hidden></video>
                <div id="video-download" style="margin: 10px 0;" hidden>
                    <a id="video-download-link" download="video-recording.webm" style="background: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">
                        📥 Download Video Recording
                    </a>
                </div>
            </div>
        </div>
    </div>

    <script>
    let videoRecorder = null;
    let videoChunks = [];
    let videoTimer = null;
    let videoStartTime = null;
    let videoPaused = false;
    let videoPauseTime = 0;
    let cameraStream = null;

    function updateVideoTimer() {
        if (videoStartTime && !videoPaused) {
            const elapsed = Math.floor((Date.now() - videoStartTime - videoPauseTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            document.getElementById('video-timer').textContent = 
                `Duration: ${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    async function startVideoRecording() {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ 
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                },
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100
                }
            });
            
            document.getElementById('camera-preview').srcObject = cameraStream;
            
            videoRecorder = new MediaRecorder(cameraStream, {
                mimeType: 'video/webm;codecs=vp9,opus'
            });
            
            videoChunks = [];
            videoStartTime = Date.now();
            videoPauseTime = 0;
            videoPaused = false;
            
            videoRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    videoChunks.push(event.data);
                }
            };
            
            videoRecorder.onstop = () => {
                const videoBlob = new Blob(videoChunks, { type: 'video/webm' });
                const videoUrl = URL.createObjectURL(videoBlob);
                
                document.getElementById('video-playback').src = videoUrl;
                document.getElementById('video-playback').hidden = false;
                document.getElementById('video-download-link').href = videoUrl;
                document.getElementById('video-download').hidden = false;
                
                // Convert to base64 and send to Streamlit
                const reader = new FileReader();
                reader.onload = function() {
                    const base64Data = reader.result.split(',')[1];
                    window.parent.postMessage({
                        type: 'video_recording',
                        data: base64Data,
                        filename: `video_${new Date().getTime()}.webm`
                    }, '*');
                };
                reader.readAsDataURL(videoBlob);
                
                // Stop camera
                if (cameraStream) {
                    cameraStream.getTracks().forEach(track => track.stop());
                    document.getElementById('camera-preview').srcObject = null;
                }
            };
            
            videoRecorder.start(1000); // Collect data every second
            
            document.getElementById('start-video').disabled = true;
            document.getElementById('pause-video').disabled = false;
            document.getElementById('stop-video').disabled = false;
            document.getElementById('video-status').textContent = 'Status: 🔴 Recording...';
            document.getElementById('video-playback').hidden = true;
            document.getElementById('video-download').hidden = true;
            
            videoTimer = setInterval(updateVideoTimer, 1000);
            
        } catch (error) {
            document.getElementById('video-status').textContent = 'Error: Could not access camera/microphone - ' + error.message;
        }
    }

    function pauseVideoRecording() {
        if (videoRecorder && videoRecorder.state === 'recording') {
            videoRecorder.pause();
            videoPaused = true;
            videoPauseTime += Date.now() - videoStartTime;
            document.getElementById('video-status').textContent = 'Status: ⏸️ Paused';
            document.getElementById('pause-video').textContent = '▶️ Resume';
            document.getElementById('pause-video').onclick = resumeVideoRecording;
        }
    }

    function resumeVideoRecording() {
        if (videoRecorder && videoRecorder.state === 'paused') {
            videoRecorder.resume();
            videoPaused = false;
            videoStartTime = Date.now();
            document.getElementById('video-status').textContent = 'Status: 🔴 Recording...';
            document.getElementById('pause-video').textContent = '⏸️ Pause';
            document.getElementById('pause-video').onclick = pauseVideoRecording;
        }
    }

    function stopVideoRecording() {
        if (videoRecorder && videoRecorder.state !== 'inactive') {
            videoRecorder.stop();
            clearInterval(videoTimer);
            
            document.getElementById('start-video').disabled = false;
            document.getElementById('pause-video').disabled = true;
            document.getElementById('stop-video').disabled = true;
            document.getElementById('pause-video').textContent = '⏸️ Pause';
            document.getElementById('pause-video').onclick = pauseVideoRecording;
            document.getElementById('video-status').textContent = 'Status: ✅ Recording completed';
        }
    }

    // Listen for messages from the recorder
    window.addEventListener('message', function(event) {
        if (event.data.type === 'audio_recording' || event.data.type === 'video_recording') {
            console.log('Recording data received:', event.data.type);
        }
    });
    </script>
    """

def show_real_time_recording():
    """Modern Streamlit interface for web-based recording"""
    st.title("🎙️🎥 Modern Meeting Recording")
    st.markdown("### Record audio and video directly in your browser using MediaRecorder API")
    
    # Initialize web recorder
    if 'web_recorder' not in st.session_state:
        st.session_state.web_recorder = WebRecorder()
    
    recorder = st.session_state.web_recorder
    
    # Recording options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎙️ Audio Recording")
        st.info("Record high-quality audio directly from your microphone")
        
        # Audio recorder HTML
        st.components.v1.html(get_audio_recorder_html(), height=400)
    
    with col2:
        st.markdown("#### 🎥 Video Recording")
        st.info("Record video with audio from your camera")
        
        # Video recorder HTML  
        st.components.v1.html(get_video_recorder_html(), height=600)
    
    # Handle recording data from JavaScript
    if st.button("🔄 Refresh to Check for New Recordings"):
        st.rerun()
    
    # File upload section for recorded files
    st.markdown("---")
    st.subheader("📁 Upload and Transcribe Recordings")
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.markdown("#### 🎵 Upload Audio File")
        st.success("📁 **File Upload Limit**: 1GB (1024MB) - Configured for large meeting recordings")
        st.info("📁 **Maximum file size: 1GB (1024MB)** - Supported formats: WAV, MP3, WEBM, M4A, OGG")
        audio_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3', 'webm', 'm4a', 'ogg'],
            help="Upload audio files to transcribe (Max size: 1GB)"
        )
        
        if audio_file is not None:
            # Check file size (1GB = 1024MB limit)
            file_size_mb = audio_file.size / 1024 / 1024
            
            if file_size_mb > 1024:  # 1GB limit
                st.error(f"⚠️ File size ({file_size_mb:.1f} MB) exceeds the 1GB (1024MB) limit. Please upload a smaller file.")
                return
            
            st.audio(audio_file, format='audio/wav')
            st.info(f"📁 File size: {file_size_mb:.1f} MB")
            
            if st.button("🚀 Transcribe Audio", key="transcribe_audio"):
                with st.spinner("Transcribing audio..."):
                    try:
                        # Save uploaded file
                        saved_path = recorder.save_recording(
                            audio_file.read(), 
                            "audio", 
                            f"uploaded_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{audio_file.name.split('.')[-1]}"
                        )
                        
                        if saved_path:
                            st.success(f"Audio saved: {saved_path}")
                            
                            # Trigger transcription if VerbatimAI is available
                            if 'verbatim_ai' in st.session_state:
                                # Reset file pointer
                                audio_file.seek(0)
                                transcript = st.session_state.verbatim_ai.transcribe_audio(audio_file)
                                if transcript:
                                    st.session_state.verbatim_ai.transcript_data = transcript
                                    st.success("✅ Audio transcribed successfully!")
                                    st.info("📊 Go to 'Analytics Dashboard' to view results.")
                                else:
                                    st.error("❌ Transcription failed")
                            else:
                                st.info("💡 Initialize VerbatimAI to enable transcription")
                        else:
                            st.error("❌ Failed to save audio file")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing audio: {e}")
    
    with col_upload2:
        st.markdown("#### 🎬 Upload Video File")
        st.success("📁 **File Upload Limit**: 1GB (1024MB) - Configured for large meeting recordings")
        st.info("📁 **Maximum file size: 1GB (1024MB)** - Supported formats: MP4, WEBM, AVI, MOV")
        video_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'webm', 'avi', 'mov'],
            help="Upload video files to extract and transcribe audio (Max size: 1GB)"
        )
        
        if video_file is not None:
            # Check file size (1GB = 1024MB limit)
            file_size_mb = video_file.size / 1024 / 1024
            
            if file_size_mb > 1024:  # 1GB limit
                st.error(f"⚠️ File size ({file_size_mb:.1f} MB) exceeds the 1GB (1024MB) limit. Please upload a smaller file.")
                return
            
            st.video(video_file)
            st.info(f"📁 File size: {file_size_mb:.1f} MB")
            
            if st.button("🚀 Transcribe Video Audio", key="transcribe_video"):
                with st.spinner("Processing video and transcribing audio..."):
                    try:
                        # Save uploaded file
                        saved_path = recorder.save_recording(
                            video_file.read(),
                            "video",
                            f"uploaded_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{video_file.name.split('.')[-1]}"
                        )
                        
                        if saved_path:
                            st.success(f"Video saved: {saved_path}")
                            st.info("🔄 Video audio extraction and transcription coming soon!")
                            # TODO: Add video audio extraction and transcription
                        else:
                            st.error("❌ Failed to save video file")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing video: {e}")
    
    # Show saved recordings
    st.markdown("---")
    st.subheader("📂 Saved Recordings")
    
    try:
        recordings_files = []
        if os.path.exists(recorder.recordings_dir):
            for file in os.listdir(recorder.recordings_dir):
                if file.endswith(('.wav', '.webm', '.mp4', '.avi', '.mp3')):
                    file_path = os.path.join(recorder.recordings_dir, file)
                    stats = recorder.get_recording_stats(file_path)
                    if stats['exists']:
                        recordings_files.append((file, stats))
        
        if recordings_files:
            for filename, stats in recordings_files[-5:]:  # Show last 5 recordings
                with st.expander(f"📁 {filename}"):
                    col_info, col_actions = st.columns([2, 1])
                    
                    with col_info:
                        st.write(f"**File:** {filename}")
                        st.write(f"**Size:** {stats['size_mb']} MB")
                        st.write(f"**Created:** {stats['created']}")
                        st.write(f"**Path:** {stats['file_path']}")
                    
                    with col_actions:
                        if filename.endswith(('.wav', '.mp3', '.webm')):
                            if st.button(f"🎵 Play Audio", key=f"play_{filename}"):
                                try:
                                    with open(stats['file_path'], 'rb') as f:
                                        st.audio(f.read())
                                except Exception as e:
                                    st.error(f"Error playing audio: {e}")
                        
                        elif filename.endswith(('.mp4', '.avi', '.webm')):
                            if st.button(f"🎬 Play Video", key=f"play_video_{filename}"):
                                try:
                                    st.video(stats['file_path'])
                                except Exception as e:
                                    st.error(f"Error playing video: {e}")
                        
                        if st.button(f"🚀 Transcribe", key=f"transcribe_{filename}"):
                            st.info("Select this file for transcription in the upload section above")
        else:
            st.info("No recordings found. Create some recordings using the tools above!")
            
    except Exception as e:
        st.error(f"Error listing recordings: {e}")
    
    # Recording tips
    st.markdown("---")
    st.subheader("💡 Recording Tips")
    
    col_tip1, col_tip2 = st.columns(2)
    
    with col_tip1:
        st.markdown("""
        **🎙️ Audio Recording:**
        - Make sure your microphone is working
        - Use a quiet environment for better quality
        - Chrome/Edge browsers work best
        - Audio is saved in WebM format
        """)
    
    with col_tip2:
        st.markdown("""
        **🎥 Video Recording:**
        - Allow camera and microphone permissions
        - Good lighting improves video quality
        - Video includes both video and audio
        - Downloads are available after recording
        """)

# Utility functions for file management
def list_saved_recordings(recordings_dir: str = "meetings") -> list:
    """List all saved recordings"""
    recordings = []
    
    if os.path.exists(recordings_dir):
        for file in os.listdir(recordings_dir):
            if file.endswith(('.wav', '.webm', '.mp4', '.avi', '.mp3', '.m4a')):
                file_path = os.path.join(recordings_dir, file)
                try:
                    size = os.path.getsize(file_path)
                    created = datetime.fromtimestamp(os.path.getctime(file_path))
                    recordings.append({
                        'filename': file,
                        'path': file_path,
                        'size_mb': round(size / (1024 * 1024), 2),
                        'created': created.strftime("%Y-%m-%d %H:%M:%S"),
                        'type': 'audio' if file.endswith(('.wav', '.mp3', '.m4a')) else 'video'
                    })
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    return sorted(recordings, key=lambda x: x['created'], reverse=True)

def cleanup_old_recordings(recordings_dir: str = "meetings", days_old: int = 30):
    """Clean up recordings older than specified days"""
    try:
        if not os.path.exists(recordings_dir):
            return
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        for file in os.listdir(recordings_dir):
            file_path = os.path.join(recordings_dir, file)
            if os.path.getctime(file_path) < cutoff_time:
                os.remove(file_path)
                cleaned_count += 1
        
        if cleaned_count > 0:
            st.info(f"Cleaned up {cleaned_count} old recordings")
            
    except Exception as e:
        st.error(f"Error cleaning up recordings: {e}")

def main():
    """Main function for standalone execution"""
    st.set_page_config(
        page_title="Modern Recording",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("🎙️🎥 Modern Meeting Recording")
    st.markdown("### Web-based audio and video recording with MediaRecorder API")
    
    # Show the recording interface
    show_real_time_recording()
    
    # Configuration info
    with st.expander("⚙️ Recording Configuration"):
        st.json(RECORDING_CONFIG)
    
    # Cleanup option
    if st.button("🧹 Cleanup Old Recordings (30+ days)"):
        cleanup_old_recordings()

# Alias for backward compatibility
RealTimeRecorder = WebRecorder

if __name__ == "__main__":
    main()
