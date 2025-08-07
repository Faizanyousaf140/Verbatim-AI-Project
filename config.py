import os
from dotenv import load_dotenv
load_dotenv()

# AssemblyAI Configuration
ASSEMBLYAI_API_KEY = "13f4b3cec3804da394f357ce6ebde48e"  # User's API key
# Fallback to environment variable if needed
if not ASSEMBLYAI_API_KEY or ASSEMBLYAI_API_KEY == "your_assemblyai_api_key_here":
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Application Configuration
APP_TITLE = "VerbatimAI - Smart Meeting Intelligence"
APP_ICON = "🎙️"

# File upload configuration
ALLOWED_EXTENSIONS = ['mp3', 'wav', 'mp4', 'avi', 'mov', 'm4a', 'opus', 'mkv', 'flv', 'webm', 'ogg', 'aac', 'wma']
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB

# NLP Configuration
DECISION_KEYWORDS = [
    'decided', 'decision', 'agreed', 'approved', 'chosen', 'selected',
    'determined', 'resolved', 'settled', 'concluded'
]

ACTION_KEYWORDS = [
    'will', 'going to', 'need to', 'should', 'must', 'action', 'task',
    'assign', 'responsible', 'deadline', 'follow up', 'review'
]

# Export Configuration
EXPORT_FORMATS = ['DOCX', 'PDF']

# New configurations for advanced features
# Email settings
EMAIL_SMTP_SERVER = "smtp.gmail.com"
EMAIL_SMTP_PORT = 587
EMAIL_SENDER = "faizanyousaf140@gmail.com"
EMAIL_PASSWORD = "ptph ssjy qvky ffih"

# Validate email settings
# if not EMAIL_SENDER or EMAIL_SENDER == "faizanyousaf140@gmail.com":
#     print("⚠️ Warning: EMAIL_SENDER not configured. Email functionality will be disabled.")
#     EMAIL_SENDER = None

# if not EMAIL_PASSWORD or EMAIL_PASSWORD == "ptphssjyqvkyffih":
#     print("⚠️ Warning: EMAIL_PASSWORD not configured. Email functionality will be disabled.")
#     EMAIL_PASSWORD = None

# Audio emotion detection settings
AUDIO_EMOTION_MODEL = "harshit345/xlsr-wav2vec-speech-emotion-recognition"
EMOTION_LABELS = ['angry', 'calm', 'confident', 'excited', 'happy', 'sad', 'worried']

# Advanced emotion detection config
EMOTION_DETECTION_CONFIG = {
    'enable_audio_emotion': True,
    'enable_text_emotion': True,
    'emotion_models': {
        'audio': 'harshit345/xlsr-wav2vec-speech-emotion-recognition',
        'text': 'j-hartmann/emotion-english-distilroberta-base'
    },
    'confidence_threshold': 0.7,
    'emotion_labels': ['angry', 'calm', 'confident', 'excited', 'happy', 'sad', 'worried']
}

# Semantic search settings
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_TYPE = "IndexFlatIP"  # Inner product for cosine similarity

# UI Theme settings - Enhanced colors
UI_THEME = {
    "primaryColor": "#4A90E2",  # Professional blue
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F8F9FA",  # Light gray
    "textColor": "#2C3E50",  # Dark blue-gray
    "accentColor": "#E74C3C",  # Red for alerts
    "successColor": "#27AE60",  # Green for success
    "warningColor": "#F39C12",  # Orange for warnings
    "font": "sans serif"
}

# Real-time recording configuration
RECORDING_CONFIG = {
    'sample_rate': 16000,
    'channels': 1,
    'chunk_size': 1024,
    'format': 'wav',
    'max_duration': 3600,  # 1 hour max
    'auto_save_interval': 300,  # 5 minutes
    'enable_vad': True,  # Voice Activity Detection
    'silence_threshold': 0.01,
    'silence_duration': 2.0  # seconds
} 