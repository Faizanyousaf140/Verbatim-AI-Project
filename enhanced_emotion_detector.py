"""
Enhanced Emotion Detection System
Uses HuggingFace models (YAMNet, CREMA-D) for advanced emotion recognition
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import tempfile
import os

# Optional imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ torch not available - using basic emotion detection")
    TORCH_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    print("⚠️ torchaudio not available - using alternative audio processing")
    TORCHAUDIO_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("⚠️ librosa not available - limited audio processing")
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    print("⚠️ soundfile not available - basic audio loading only")
    SOUNDFILE_AVAILABLE = False

try:
    from transformers import (
        pipeline, 
        AutoFeatureExtractor, 
        AutoModelForAudioClassification,
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2FeatureExtractor
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ transformers not available - using fallback emotion detection")
    TRANSFORMERS_AVAILABLE = False

try:
    from config import EMOTION_DETECTION_CONFIG
except ImportError:
    # Fallback configuration
    EMOTION_DETECTION_CONFIG = {
        'models': ['text_fallback'],
        'confidence_threshold': 0.5,
        'ensemble_voting': True
    }

class EnhancedEmotionDetector:
    """Enhanced emotion detection using multiple models"""
    
    def __init__(self):
        """Initialize the enhanced emotion detector"""
        self.models = {}
        self.feature_extractors = {}
        self.device = None
        
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_models()
    
    def _load_models(self):
        """Load various emotion detection models"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers not available - using fallback emotion detection")
            self._setup_fallback_models()
            return
            
        try:
            print("🔄 Loading enhanced emotion detection models...")
            
            # Load YAMNet for audio classification
            try:
                if TORCH_AVAILABLE:
                    self.models['yamnet'] = pipeline(
                        "audio-classification",
                        model="harshit345/xlsr-wav2vec-speech-emotion-recognition",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    print("✅ YAMNet emotion model loaded")
                else:
                    print("⚠️ YAMNet model requires PyTorch")
            except Exception as e:
                print(f"⚠️ YAMNet model not available: {e}")
                self.models['yamnet'] = None
            
            # Load CREMA-D model
            try:
                if TORCH_AVAILABLE:
                    self.models['crema_d'] = pipeline(
                        "audio-classification",
                        model="harshit345/xlsr-wav2vec-speech-emotion-recognition",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    print("✅ CREMA-D emotion model loaded")
                else:
                    print("⚠️ CREMA-D model requires PyTorch")
            except Exception as e:
                print(f"⚠️ CREMA-D model not available: {e}")
                self.models['crema_d'] = None
            
            # Load Wav2Vec2 for speech emotion recognition
            try:
                if TORCH_AVAILABLE:
                    self.models['wav2vec2'] = pipeline(
                        "audio-classification",
                        model="superb/wav2vec2-base-superb-ks",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    print("✅ Wav2Vec2 emotion model loaded")
                else:
                    print("⚠️ Wav2Vec2 model requires PyTorch")
            except Exception as e:
                print(f"⚠️ Wav2Vec2 model not available: {e}")
                self.models['wav2vec2'] = None
            
            # Load text-based emotion detection
            try:
                if TORCH_AVAILABLE:
                    self.models['text_emotion'] = pipeline(
                        "text-classification",
                        model="j-hartmann/emotion-english-distilroberta-base",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    print("✅ Text emotion model loaded")
                else:
                    print("⚠️ Text emotion model requires PyTorch")
            except Exception as e:
                print(f"⚠️ Text emotion model not available: {e}")
                self.models['text_emotion'] = None
            
            # Setup fallback if no models loaded
            if not any(self.models.values()):
                self._setup_fallback_models()
            else:
                print("✅ Enhanced emotion detection models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading emotion detection models: {e}")
            self._setup_fallback_models()
    
    def _setup_fallback_models(self):
        """Setup fallback emotion detection methods"""
        print("🔄 Setting up fallback emotion detection...")
        
        # Basic text-based emotion keywords
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'good'],
            'sad': ['sad', 'unhappy', 'disappointed', 'sorry', 'unfortunate', 'regret', 'terrible', 'awful'],
            'angry': ['angry', 'mad', 'furious', 'frustrated', 'annoyed', 'upset', 'irritated', 'outraged'],
            'fearful': ['worried', 'concerned', 'afraid', 'scared', 'nervous', 'anxious', 'uncertain'],
            'surprised': ['surprised', 'shocked', 'amazed', 'unexpected', 'wow', 'incredible', 'unbelievable'],
            'disgusted': ['disgusted', 'awful', 'terrible', 'horrible', 'disgusting', 'unacceptable'],
            'neutral': ['okay', 'fine', 'normal', 'standard', 'regular', 'typical']
        }
        
        self.models['text_fallback'] = True
        print("✅ Fallback emotion detection ready")
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocess audio for emotion detection"""
        if not SOUNDFILE_AVAILABLE:
            print("⚠️ soundfile not available - cannot preprocess audio")
            return None
            
        try:
            # Load audio file
            audio, sample_rate = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to 16kHz if needed
            if LIBROSA_AVAILABLE and sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            elif sample_rate != 16000:
                print("⚠️ librosa not available - cannot resample audio")
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None
    
    def detect_emotion_yamnet(self, audio_path: str) -> Dict:
        """Detect emotion using YAMNet model"""
        if not self.models.get('yamnet'):
            return self._fallback_audio_emotion(audio_path, 'yamnet')
            
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            if audio is None:
                return {"error": "Failed to preprocess audio"}
            
            # Save preprocessed audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                if SOUNDFILE_AVAILABLE:
                    sf.write(temp_file.name, audio, 16000)
                    temp_path = temp_file.name
                else:
                    return {"error": "soundfile not available for audio processing"}
            
            try:
                # Run emotion detection
                results = self.models['yamnet'](temp_path)
                
                # Process results
                emotions = []
                for result in results:
                    emotions.append({
                        'emotion': result['label'],
                        'confidence': result['score'],
                        'model': 'yamnet'
                    })
                
                # Get top emotion
                top_emotion = max(emotions, key=lambda x: x['confidence'])
                
                return {
                    'emotion': top_emotion['emotion'],
                    'confidence': top_emotion['confidence'],
                    'all_emotions': emotions,
                    'model': 'yamnet'
                }
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            return {"error": f"YAMNet emotion detection failed: {str(e)}"}
    
    def detect_emotion_crema_d(self, audio_path: str) -> Dict:
        """Detect emotion using CREMA-D model"""
        if not self.models.get('crema_d'):
            return self._fallback_audio_emotion(audio_path, 'crema_d')
            
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            if audio is None:
                return {"error": "Failed to preprocess audio"}
            
            # Save preprocessed audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                if SOUNDFILE_AVAILABLE:
                    sf.write(temp_file.name, audio, 16000)
                    temp_path = temp_file.name
                else:
                    return {"error": "soundfile not available for audio processing"}
            
            try:
                # Run emotion detection
                results = self.models['crema_d'](temp_path)
                
                # Process results
                emotions = []
                for result in results:
                    emotions.append({
                        'emotion': result['label'],
                        'confidence': result['score'],
                        'model': 'crema_d'
                    })
                
                # Get top emotion
                top_emotion = max(emotions, key=lambda x: x['confidence'])
                
                return {
                    'emotion': top_emotion['emotion'],
                    'confidence': top_emotion['confidence'],
                    'all_emotions': emotions,
                    'model': 'crema_d'
                }
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            return {"error": f"CREMA-D emotion detection failed: {str(e)}"}
    
    def detect_emotion_wav2vec2(self, audio_path: str) -> Dict:
        """Detect emotion using Wav2Vec2 model"""
        if not self.models.get('wav2vec2'):
            return self._fallback_audio_emotion(audio_path, 'wav2vec2')
            
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            if audio is None:
                return {"error": "Failed to preprocess audio"}
            
            # Save preprocessed audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                if SOUNDFILE_AVAILABLE:
                    sf.write(temp_file.name, audio, 16000)
                    temp_path = temp_file.name
                else:
                    return {"error": "soundfile not available for audio processing"}
            
            try:
                # Run emotion detection
                results = self.models['wav2vec2'](temp_path)
                
                # Process results
                emotions = []
                for result in results:
                    emotions.append({
                        'emotion': result['label'],
                        'confidence': result['score'],
                        'model': 'wav2vec2'
                    })
                
                # Get top emotion
                top_emotion = max(emotions, key=lambda x: x['confidence'])
                
                return {
                    'emotion': top_emotion['emotion'],
                    'confidence': top_emotion['confidence'],
                    'all_emotions': emotions,
                    'model': 'wav2vec2'
                }
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            return {"error": f"Wav2Vec2 emotion detection failed: {str(e)}"}
    
    def detect_text_emotion(self, text: str) -> Dict:
        """Detect emotion from text using NLP models"""
        if not self.models.get('text_emotion'):
            return self._fallback_text_emotion(text)
            
        try:
            # Run text emotion detection
            results = self.models['text_emotion'](text)
            
            # Process results
            emotions = []
            for result in results:
                emotions.append({
                    'emotion': result['label'],
                    'confidence': result['score'],
                    'model': 'text_emotion'
                })
            
            # Get top emotion
            top_emotion = max(emotions, key=lambda x: x['confidence'])
            
            return {
                'emotion': top_emotion['emotion'],
                'confidence': top_emotion['confidence'],
                'all_emotions': emotions,
                'model': 'text_emotion'
            }
            
        except Exception as e:
            return {"error": f"Text emotion detection failed: {str(e)}"}
    
    def _fallback_audio_emotion(self, audio_path: str, model_name: str) -> Dict:
        """Fallback audio emotion detection"""
        return {
            'emotion': 'neutral',
            'confidence': 0.3,
            'model': f'{model_name}_fallback',
            'note': f'{model_name} model not available - using fallback'
        }
    
    def _fallback_text_emotion(self, text: str) -> Dict:
        """Fallback text emotion detection using keywords"""
        if not hasattr(self, 'emotion_keywords'):
            return {
                'emotion': 'neutral',
                'confidence': 0.3,
                'model': 'text_fallback',
                'note': 'Basic fallback emotion detection'
            }
        
        text_lower = text.lower()
        emotion_scores = {}
        
        # Score each emotion based on keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            
            if score > 0:
                emotion_scores[emotion] = score / len(keywords)
        
        if emotion_scores:
            # Get emotion with highest score
            best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            confidence = min(0.8, best_emotion[1] * 2)  # Cap at 0.8
            
            return {
                'emotion': best_emotion[0],
                'confidence': confidence,
                'model': 'text_fallback',
                'keyword_matches': emotion_scores
            }
        else:
            return {
                'emotion': 'neutral',
                'confidence': 0.4,
                'model': 'text_fallback',
                'note': 'No emotion keywords detected'
            }
    
    def detect_emotion_ensemble(self, audio_path: str, text: str = None) -> Dict:
        """Detect emotion using ensemble of multiple models"""
        try:
            results = {}
            
            # Audio-based emotion detection
            yamnet_result = self.detect_emotion_yamnet(audio_path)
            if 'error' not in yamnet_result:
                results['yamnet'] = yamnet_result
            
            crema_result = self.detect_emotion_crema_d(audio_path)
            if 'error' not in crema_result:
                results['crema_d'] = crema_result
            
            wav2vec_result = self.detect_emotion_wav2vec2(audio_path)
            if 'error' not in wav2vec_result:
                results['wav2vec2'] = wav2vec_result
            
            # Text-based emotion detection
            if text:
                text_result = self.detect_text_emotion(text)
                if 'error' not in text_result:
                    results['text'] = text_result
            
            if not results:
                return {"error": "No emotion detection models available"}
            
            # Ensemble voting
            emotion_votes = {}
            total_confidence = 0
            
            for model_name, result in results.items():
                emotion = result['emotion']
                confidence = result['confidence']
                
                if emotion not in emotion_votes:
                    emotion_votes[emotion] = 0
                
                emotion_votes[emotion] += confidence
                total_confidence += confidence
            
            # Get most voted emotion
            if emotion_votes:
                ensemble_emotion = max(emotion_votes.items(), key=lambda x: x[1])
                
                return {
                    'emotion': ensemble_emotion[0],
                    'confidence': ensemble_emotion[1] / total_confidence,
                    'model_results': results,
                    'ensemble_votes': emotion_votes,
                    'model': 'ensemble'
                }
            else:
                return {"error": "No emotion detected by any model"}
                
        except Exception as e:
            return {"error": f"Ensemble emotion detection failed: {str(e)}"}
    
    def analyze_utterance_emotions(self, audio_file_path: str, utterances: List) -> List:
        """Analyze emotions for all utterances using enhanced models"""
        try:
            print("🔊 Analyzing emotions with enhanced models...")
            
            enhanced_utterances = []
            
            for utterance in utterances:
                if hasattr(utterance, 'start') and hasattr(utterance, 'end'):
                    # Extract audio segment for this utterance
                    audio_segment_path = self._extract_audio_segment(
                        audio_file_path, 
                        utterance.start, 
                        utterance.end
                    )
                    
                    if audio_segment_path:
                        # Detect emotion using ensemble
                        emotion_result = self.detect_emotion_ensemble(
                            audio_segment_path, 
                            utterance.text
                        )
                        
                        # Add emotion data to utterance
                        if 'error' not in emotion_result:
                            utterance.enhanced_emotion = emotion_result['emotion']
                            utterance.enhanced_confidence = emotion_result['confidence']
                            utterance.enhanced_model_results = emotion_result.get('model_results', {})
                            utterance.ensemble_votes = emotion_result.get('ensemble_votes', {})
                        else:
                            utterance.enhanced_emotion = "unknown"
                            utterance.enhanced_confidence = 0.0
                            utterance.enhanced_model_results = {}
                            utterance.ensemble_votes = {}
                        
                        enhanced_utterances.append(utterance)
                        
                        # Clean up segment file
                        if os.path.exists(audio_segment_path):
                            os.unlink(audio_segment_path)
            
            print("✅ Enhanced emotion analysis completed")
            return enhanced_utterances
            
        except Exception as e:
            print(f"Error in enhanced emotion analysis: {e}")
            return utterances
    
    def _extract_audio_segment(self, audio_file_path: str, start_time: int, end_time: int) -> Optional[str]:
        """Extract audio segment between start_time and end_time (in milliseconds)"""
        if not LIBROSA_AVAILABLE or not SOUNDFILE_AVAILABLE:
            print("⚠️ librosa or soundfile not available - cannot extract audio segments")
            return None
            
        try:
            # Load audio file
            audio, sample_rate = librosa.load(audio_file_path, sr=None)
            
            # Convert times to samples
            start_sample = int(start_time * sample_rate / 1000)
            end_sample = int(end_time * sample_rate / 1000)
            
            # Extract segment
            segment = audio[start_sample:end_sample]
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                sf.write(temp_file.name, segment, sample_rate)
                return temp_file.name
                
        except Exception as e:
            print(f"Error extracting audio segment: {e}")
            return None
    
    def get_emotion_statistics(self, utterances: List) -> Dict:
        """Get emotion statistics from analyzed utterances"""
        try:
            emotion_counts = {}
            model_agreement = {}
            
            for utterance in utterances:
                if hasattr(utterance, 'enhanced_emotion'):
                    emotion = utterance.enhanced_emotion
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    # Check model agreement
                    if hasattr(utterance, 'ensemble_votes'):
                        votes = utterance.ensemble_votes
                        if len(votes) > 1:
                            max_vote = max(votes.values())
                            agreement = max_vote / sum(votes.values())
                            model_agreement[emotion] = model_agreement.get(emotion, []) + [agreement]
            
            # Calculate statistics
            total_utterances = len([u for u in utterances if hasattr(u, 'enhanced_emotion')])
            
            emotion_percentages = {}
            for emotion, count in emotion_counts.items():
                emotion_percentages[emotion] = (count / total_utterances * 100) if total_utterances > 0 else 0
            
            avg_agreement = {}
            for emotion, agreements in model_agreement.items():
                avg_agreement[emotion] = np.mean(agreements) if agreements else 0
            
            return {
                'emotion_counts': emotion_counts,
                'emotion_percentages': emotion_percentages,
                'total_utterances': total_utterances,
                'model_agreement': avg_agreement,
                'dominant_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
            }
            
        except Exception as e:
            print(f"Error calculating emotion statistics: {e}")
            return {}

# Utility functions
def format_emotion_result(result: Dict) -> str:
    """Format emotion detection result for display"""
    if 'error' in result:
        return f"Error: {result['error']}"
    
    return f"{result['emotion'].title()} ({result['confidence']:.2%})"

def get_emotion_color(emotion: str) -> str:
    """Get color for emotion display"""
    emotion_colors = {
        'happy': '#28a745',
        'sad': '#6c757d',
        'angry': '#dc3545',
        'fear': '#ffc107',
        'surprise': '#17a2b8',
        'disgust': '#6f42c1',
        'neutral': '#6c757d',
        'excited': '#fd7e14',
        'calm': '#20c997',
        'confident': '#007bff',
        'worried': '#e83e8c'
    }
    return emotion_colors.get(emotion.lower(), '#6c757d')


def test_enhanced_emotion_detector():
    """Test function to verify the enhanced emotion detector works"""
    print("🧪 Testing Enhanced Emotion Detector...")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = EnhancedEmotionDetector()
        print("✅ EnhancedEmotionDetector initialized successfully")
        
        # Test text emotion detection with sample texts
        test_texts = [
            "I am so happy and excited about this project!",
            "This is really disappointing and makes me sad.",
            "I'm angry about this situation!",
            "This is a normal meeting discussion."
        ]
        
        for text in test_texts:
            result = detector.detect_text_emotion(text)
            print(f"📝 Text: '{text[:30]}...'")
            print(f"   Emotion: {result.get('emotion', 'unknown')} (confidence: {result.get('confidence', 0):.2f})")
        
        print("\n🎯 Available Dependencies:")
        print(f"  • torch: {'✅' if TORCH_AVAILABLE else '❌'}")
        print(f"  • torchaudio: {'✅' if TORCHAUDIO_AVAILABLE else '❌'}")
        print(f"  • transformers: {'✅' if TRANSFORMERS_AVAILABLE else '❌'}")
        print(f"  • librosa: {'✅' if LIBROSA_AVAILABLE else '❌'}")
        print(f"  • soundfile: {'✅' if SOUNDFILE_AVAILABLE else '❌'}")
        
        print("\n🤖 Available Models:")
        available_models = [model for model, loaded in detector.models.items() if loaded]
        if available_models:
            for model in available_models:
                print(f"  • {model}: ✅")
        else:
            print("  • No advanced models loaded - using fallback methods")
        
        print("\n✅ Enhanced Emotion Detector test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    test_enhanced_emotion_detector() 