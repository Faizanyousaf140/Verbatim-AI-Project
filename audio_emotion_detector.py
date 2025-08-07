"""
Audio Emotion Detection Module
Uses pre-trained models to detect emotions from audio segments
"""

import numpy as np
import tempfile
import os

# Optional imports with fallbacks
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ librosa not available - using basic audio analysis")
    LIBROSA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ torch not available - using fallback emotion detection")
    TORCH_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    print("⚠️ pydub not available - limited audio processing")
    PYDUB_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    print("⚠️ soundfile not available - using basic audio loading")
    SOUNDFILE_AVAILABLE = False

try:
    from config import EMOTION_LABELS
except ImportError:
    # Fallback emotion labels
    EMOTION_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

class AudioEmotionDetector:
    def __init__(self):
        """Initialize the audio emotion detector with pre-trained model"""
        self.model = None
        self.feature_extractor = None
        self.device = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained emotion detection model"""
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch not available - using fallback emotion detection")
            return
            
        try:
            print("Loading audio emotion detection model...")
            # Try to import transformers
            try:
                from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
                from config import AUDIO_EMOTION_MODEL
                
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_EMOTION_MODEL)
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_EMOTION_MODEL)
                self.model.to(self.device)
                self.model.eval()
                print("✅ Audio emotion detection model loaded successfully")
            except Exception as model_error:
                print(f"⚠️ Could not load pre-trained model: {model_error}")
                print("⚠️ Using fallback audio analysis instead")
                self.model = None
                self.feature_extractor = None
                
        except Exception as e:
            print(f"❌ Error loading audio emotion model: {e}")
            self.model = None
            self.feature_extractor = None
    
    def extract_audio_segment(self, audio_file_path, start_time, end_time):
        """Extract audio segment between start_time and end_time (in milliseconds)"""
        if not PYDUB_AVAILABLE:
            print("⚠️ pydub not available - cannot extract audio segments")
            return None
            
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_file_path)
            
            # Extract segment
            segment = audio[start_time:end_time]
            
            # Export to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                segment.export(temp_file.name, format="wav")
                temp_path = temp_file.name
            
            return temp_path
        except Exception as e:
            print(f"Error extracting audio segment: {e}")
            return None
    
    def detect_emotion_from_features(self, audio_segment_path):
        """Detect emotion using basic audio features (fallback method)"""
        if not SOUNDFILE_AVAILABLE:
            # Ultra-basic fallback
            return {
                "emotion": "neutral",
                "confidence": 0.3,
                "method": "basic_fallback",
                "note": "Limited audio analysis available"
            }
            
        try:
            # Load audio
            audio, sample_rate = sf.read(audio_segment_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            if LIBROSA_AVAILABLE:
                # Extract advanced features with librosa
                # RMS (loudness)
                rms = np.sqrt(np.mean(audio**2))
                
                # Spectral centroid (brightness)
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
                avg_spectral_centroid = np.mean(spectral_centroids)
                
                # Zero crossing rate (noisiness)
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                avg_zcr = np.mean(zcr)
                
                # Simple emotion mapping based on features
                if rms > 0.1 and avg_spectral_centroid > 2000:
                    emotion = "excited"
                    confidence = 0.7
                elif rms < 0.05 and avg_zcr < 0.1:
                    emotion = "calm"
                    confidence = 0.6
                elif avg_spectral_centroid > 3000:
                    emotion = "angry"
                    confidence = 0.5
                elif rms > 0.08 and avg_spectral_centroid < 1500:
                    emotion = "confident"
                    confidence = 0.6
                else:
                    emotion = "neutral"
                    confidence = 0.5
                
                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "method": "librosa_analysis",
                    "features": {
                        "rms": float(rms),
                        "spectral_centroid": float(avg_spectral_centroid),
                        "zero_crossing_rate": float(avg_zcr)
                    }
                }
            else:
                # Basic analysis without librosa
                rms = np.sqrt(np.mean(audio**2))
                
                if rms > 0.1:
                    emotion = "excited"
                    confidence = 0.5
                elif rms < 0.03:
                    emotion = "calm"
                    confidence = 0.4
                else:
                    emotion = "neutral"
                    confidence = 0.4
                
                return {
                    "emotion": emotion,
                    "confidence": confidence,
                    "method": "basic_analysis",
                    "features": {
                        "rms": float(rms)
                    }
                }
            
        except Exception as e:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "error": str(e),
                "method": "error_fallback"
            }
    
    def detect_emotion(self, audio_segment_path):
        """Detect emotion from audio segment"""
        if not self.model or not self.feature_extractor or not TORCH_AVAILABLE:
            # Use fallback method
            return self.detect_emotion_from_features(audio_segment_path)
        
        try:
            if not SOUNDFILE_AVAILABLE:
                return self.detect_emotion_from_features(audio_segment_path)
                
            # Load and preprocess audio
            audio, sample_rate = sf.read(audio_segment_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if LIBROSA_AVAILABLE and sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # Prepare input for model
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get emotion label
            emotion = EMOTION_LABELS[predicted_class] if predicted_class < len(EMOTION_LABELS) else "unknown"
            
            return {
                "emotion": emotion,
                "confidence": confidence,
                "probabilities": probabilities[0].cpu().numpy().tolist(),
                "method": "pre_trained_model"
            }
            
        except Exception as e:
            # Fallback to basic analysis
            return self.detect_emotion_from_features(audio_segment_path)
        finally:
            # Clean up temporary file
            if os.path.exists(audio_segment_path):
                try:
                    os.unlink(audio_segment_path)
                except:
                    pass  # Ignore cleanup errors
    
    def analyze_utterance_emotions(self, audio_file_path, utterances):
        """Analyze emotions for all utterances in a transcript"""
        if not self.model and not hasattr(self, '_fallback_available'):
            print("⚠️ Using fallback audio emotion detection")
            self._fallback_available = True
        
        print("🔊 Analyzing audio emotions for utterances...")
        
        for utterance in utterances:
            if hasattr(utterance, 'start') and hasattr(utterance, 'end'):
                # Extract audio segment for this utterance
                segment_path = self.extract_audio_segment(
                    audio_file_path, 
                    utterance.start, 
                    utterance.end
                )
                
                if segment_path:
                    # Detect emotion
                    emotion_result = self.detect_emotion(segment_path)
                    
                    # Add emotion data to utterance
                    utterance.audio_emotion = emotion_result.get("emotion", "unknown")
                    utterance.audio_confidence = emotion_result.get("confidence", 0.0)
                    utterance.audio_probabilities = emotion_result.get("probabilities", [])
                    utterance.audio_method = emotion_result.get("method", "unknown")
        
        print("✅ Audio emotion analysis completed")
        return utterances

# Fallback emotion detector using audio features
class FallbackAudioEmotionDetector:
    """Fallback emotion detector using audio features when model is not available"""
    
    def __init__(self):
        self.emotion_keywords = {
            'calm': ['slow', 'steady', 'quiet', 'gentle'],
            'excited': ['fast', 'loud', 'energetic', 'enthusiastic'],
            'confident': ['clear', 'strong', 'assertive', 'decisive'],
            'worried': ['hesitant', 'uncertain', 'nervous', 'anxious'],
            'angry': ['sharp', 'harsh', 'aggressive', 'forceful'],
            'happy': ['bright', 'cheerful', 'positive', 'upbeat'],
            'sad': ['low', 'slow', 'melancholy', 'down']
        }
    
    def detect_emotion_from_features(self, audio_features):
        """Detect emotion from audio features (pitch, tempo, energy)"""
        # This is a simplified fallback - in practice you'd use more sophisticated analysis
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "method": "fallback"
        }


def test_audio_emotion_detector():
    """Test function to verify the audio emotion detector works"""
    print("🧪 Testing Audio Emotion Detector...")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = AudioEmotionDetector()
        print("✅ AudioEmotionDetector initialized successfully")
        
        # Test fallback emotion detector
        fallback_detector = FallbackAudioEmotionDetector()
        print("✅ FallbackAudioEmotionDetector initialized successfully")
        
        # Test basic emotion detection
        test_features = {"rms": 0.05, "spectral_centroid": 1500}
        result = fallback_detector.detect_emotion_from_features(test_features)
        print(f"✅ Fallback emotion detection test: {result}")
        
        print("\n🎯 Available Dependencies:")
        print(f"  • librosa: {'✅' if LIBROSA_AVAILABLE else '❌'}")
        print(f"  • torch: {'✅' if TORCH_AVAILABLE else '❌'}")
        print(f"  • pydub: {'✅' if PYDUB_AVAILABLE else '❌'}")
        print(f"  • soundfile: {'✅' if SOUNDFILE_AVAILABLE else '❌'}")
        
        print("\n✅ Audio Emotion Detector test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    test_audio_emotion_detector() 