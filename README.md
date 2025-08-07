# 🎙️ VerbatimAI - Enterprise-Grade Meeting Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Redis](https://img.shields.io/badge/Redis-7.0%2B-red.svg)](https://redis.io/)
[![Celery](https://img.shields.io/badge/Celery-5.3%2B-green.svg)](https://docs.celeryproject.org/)

A revolutionary AI-powered meeting intelligence platform that transforms audio/video meetings into actionable insights with enterprise-grade analytics, Google/Microsoft-quality UI, and advanced semantic understanding.

## 🌟 Enterprise Features

### 🎙️🎥 Advanced Media Processing
- **Multi-format Support**: Upload MP3, WAV, MP4, AVI, MOV, M4A files (up to 1GB)
- **Real-time Recording**: Professional audio + video recording with live preview
- **Speaker Diarization**: Advanced speaker identification with confidence scoring
- **High-accuracy Transcription**: Powered by AssemblyAI with 95%+ accuracy
- **Meeting Type Optimization**: Specialized processing for interviews, standups, reviews
- **Batch Processing**: Handle multiple files simultaneously

### 🧠 AI-Powered Intelligence Engine
- **Smart Key Point Extraction**: ML-powered identification of decisions, action items, questions
- **Advanced Sentiment Analysis**: Real-time emotion detection with speaker-specific insights
- **Semantic Understanding**: Vector embeddings for deep content comprehension
- **AI Content Mapping**: Intelligent detection and categorization of technical discussions
- **Topic Clustering**: Automatic theme discovery and progression analysis
- **Entity Recognition**: Extract names, organizations, dates, and key entities

### 📊 Google/Microsoft-Quality Analytics Dashboard
- **Modern Glassmorphism UI**: Professional design with Inter fonts and 20-color palette
- **Interactive Multi-tab Analytics**: Speaker, Sentiment, Semantic, Engagement, AI, Reports
- **Real-time Metrics**: Live engagement scoring and participation balance
- **Advanced Visualizations**: Plotly-powered charts with professional styling
- **Heatmaps & Timelines**: Sentiment evolution and topic progression tracking
- **Export-ready Charts**: Publication-quality visualizations

### 🔍 Enterprise Search & Discovery
- **Semantic Search Engine**: Vector-based similarity search using sentence transformers
- **Advanced Filtering**: Multi-dimensional filtering by speaker, sentiment, time, topics
- **AI Content Discovery**: Specialized search for technical and AI-related discussions
- **Topic Clustering**: Unsupervised learning for theme identification
- **Similarity Analysis**: Find related content across meetings
- **Search Analytics**: Query performance and relevance scoring

### 📄 Professional Export & Automation
- **Comprehensive Reports**: Multi-format export (PDF, HTML, DOCX, CSV, JSON)
- **Automated Email Summaries**: Scheduled delivery with customizable templates
- **Analytics Export**: CSV/Excel exports with detailed metrics
- **API Integration**: RESTful endpoints for enterprise integration
- **Batch Export**: Bulk data processing and export capabilities

### ⚡ Enterprise Infrastructure
- **Async Task Queue**: Celery + Redis for scalable background processing
- **Progress Monitoring**: Real-time task tracking with WebSocket updates
- **Horizontal Scaling**: Multi-worker support for enterprise workloads
- **Task Routing**: Intelligent queue management for optimal performance
- **Health Monitoring**: Comprehensive system health checks and alerts

### 🎨 Premium User Experience
- **Google/Microsoft Design Language**: Professional UI with modern design patterns
- **Responsive Design**: Optimized for desktop, tablet, and mobile
- **Dark/Light Themes**: Adaptive theming with user preferences
- **Accessibility**: WCAG 2.1 compliant interface
- **Loading States**: Smooth animations and progress indicators
- **Error Handling**: Graceful error management with user-friendly messages

## 🛠️ Technical Architecture

### Core Technologies
- **Frontend**: Streamlit with custom CSS/JavaScript
- **Backend**: Python 3.10+ with FastAPI integration
- **AI/ML**: AssemblyAI, sentence-transformers, scikit-learn
- **Database**: Redis for caching and task queues
- **Processing**: Celery for distributed task processing
- **Visualization**: Plotly for interactive charts

### Scalability Features
- **Microservices Architecture**: Modular design for enterprise deployment
- **Container Support**: Docker and Kubernetes ready
- **Load Balancing**: Multi-instance support with session affinity
- **Caching Layer**: Redis-based intelligent caching
- **Background Processing**: Queue-based async operations

## 🚀 Quick Start

### System Requirements
- **Python**: 3.10 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Stable internet for AI API calls
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+

### Prerequisites
- Python 3.10 or higher
- AssemblyAI API key (free tier available)
- For video recording: Camera permissions
- For background tasks: Redis server (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd verbatim-ai
   ```

2. **Install Redis Server** (Required for async processing)
   
   **Windows:**
   ```bash
   # Install via Chocolatey
   choco install redis-64
   
   # Or download from GitHub releases
   # https://github.com/tporadowski/redis/releases
   ```
   
   **macOS:**
   ```bash
   brew install redis
   brew services start redis
   ```
   
   **Ubuntu/Linux:**
   ```bash
   sudo apt update
   sudo apt install redis-server
   sudo systemctl start redis-server
   ```

3. **Install core dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install advanced ML dependencies**
   ```bash
   pip install sentence-transformers
   pip install scikit-learn
   pip install faiss-cpu  # or faiss-gpu for CUDA support
   ```

5. **Install video recording support (optional)**
   ```bash
   python install_video_support.py
   ```
   Or manually:
   ```bash
   pip install opencv-python pyaudio
   ```

6. **Set up environment variables**
   Create a `.env` file:
   ```env
   ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
   REDIS_URL=redis://localhost:6379
   CELERY_BROKER_URL=redis://localhost:6379
   ```
   Get your free API key from: https://www.assemblyai.com/

7. **Start background workers** (New terminal)
   ```bash
   # Start Celery worker
   celery -A celery_app worker --loglevel=info
   
   # Start Celery beat scheduler (optional, for scheduled tasks)
   celery -A celery_app beat --loglevel=info
   ```

8. **Run the application**
   ```bash
   streamlit run app.py
   ```

9. **Open your browser**
   Navigate to `http://localhost:8501`

### Enterprise Deployment

For production deployment with multiple workers:

```bash
# Start multiple workers for different queues
celery -A celery_app worker --loglevel=info --queues=high_priority,cpu_intensive
celery -A celery_app worker --loglevel=info --queues=analysis,search
celery -A celery_app worker --loglevel=info --queues=reports,batch_operations

# Monitor tasks
celery -A celery_app flower
```

## 🛠️ Compatibility & Troubleshooting

### Known Compatibility Issues

#### 1. **Python Version Conflicts**
- **Issue**: VerbatimAI requires Python 3.10+ for optimal performance
- **Solution**: Use pyenv or conda to manage Python versions
- **Command**: `pyenv install 3.10.12 && pyenv local 3.10.12`

#### 2. **NumPy/Pandas Version Conflicts**
- **Issue**: Newer NumPy versions may conflict with some dependencies
- **Solution**: Use pinned versions from requirements.txt
- **Fix**: `python fix_numpy_pandas.py`

#### 3. **AssemblyAI API Rate Limits**
- **Issue**: Free tier has limited concurrent requests
- **Solution**: Implement request queuing or upgrade to paid plan
- **Workaround**: Use async processing with Celery

#### 4. **FFmpeg Missing for Video Processing**
- **Issue**: Video file processing fails without FFmpeg
- **Solution**: Install FFmpeg system-wide
- **Windows**: `choco install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

#### 5. **Redis Connection Issues**
- **Issue**: Celery tasks fail without Redis
- **Solution**: Ensure Redis is running and accessible
- **Check**: `redis-cli ping` should return "PONG"

#### 6. **Memory Issues with Large Files**
- **Issue**: Files >500MB may cause memory errors
- **Solution**: Implement chunked processing
- **Workaround**: Use cloud storage and streaming

#### 7. **Streamlit Session State Conflicts**
- **Issue**: Session state corruption with multiple tabs
- **Solution**: Implement proper state management
- **Fix**: Clear browser cache and restart application

#### 8. **WebRTC Audio Recording Limitations**
- **Issue**: Browser-based recording has format limitations
- **Solution**: Use downloadable recordings for better quality
- **Alternative**: Desktop recording tools integration

### Platform-Specific Issues

#### Windows
- **PyAudio Installation**: May require Visual C++ Build Tools
- **Permission Issues**: Run as administrator for system-level dependencies
- **Path Issues**: Ensure Python and pip are in system PATH

#### macOS
- **M1/M2 Chip Compatibility**: Some ML libraries need ARM-specific builds
- **Solution**: `pip install --upgrade pip setuptools wheel`
- **Tensorflow**: Use `tensorflow-macos` for Apple Silicon

#### Linux
- **Audio System**: May require additional ALSA/PulseAudio packages
- **Dependencies**: `sudo apt install python3-dev libasound2-dev portaudio19-dev`

### Performance Optimization

#### For Large Scale Deployment
1. **Use Redis Cluster** for distributed caching
2. **Implement CDN** for static assets
3. **Database Optimization** for meeting storage
4. **Load Balancing** with nginx/Apache
5. **Container Orchestration** with Kubernetes

#### Memory Optimization
```python
# Add to config.py for large file handling
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
ENABLE_STREAMING = True
```

### Debugging Tools

#### Enable Debug Mode
```bash
export STREAMLIT_LOGGER_LEVEL=debug
export CELERY_LOG_LEVEL=DEBUG
streamlit run app.py
```

#### Health Check Endpoints
- **Application**: `http://localhost:8501/health`
- **Celery Worker**: `celery -A celery_app inspect active`
- **Redis**: `redis-cli monitor`

## 📖 Usage Guide

### 1. 📁 Upload & Configure Meeting

1. **Select Meeting Type**: Choose from 12 meeting types (General, Standup, Interview, etc.)
2. **Set Duration**: Select expected time frame (0-10 min to 2+ hours)
3. **Upload File**: Support for files up to 1GB in multiple formats
4. **Configure Options**: 
   - Speaker detection and language settings
   - Real-time sentiment analysis
   - Auto punctuation and profanity filtering
   - Auto highlights detection

### 2. 🚀 Advanced Transcription

The enhanced transcription process includes:
- **Progress Tracking**: Real-time progress for large files (>100MB)
- **Meeting Context**: Analysis optimized for meeting type
- **Async Processing**: Background processing for enterprise workloads
- **Quality Assurance**: Confidence scoring and validation

### 3. 📊 Multi-Tab Analytics Dashboard

#### Speaker Analytics Tab
- **Comprehensive Metrics**: Speaking time, word count, participation balance
- **Interactive Charts**: Scatter plots and performance comparisons
- **Performance Table**: Detailed speaker statistics with WPM calculations

#### Sentiment Analysis Tab
- **Real-time Sentiment**: Timeline showing emotional progression
- **Emotion Heatmaps**: Distribution of emotions across speakers
- **Mood Analytics**: Overall meeting sentiment with confidence scores

#### Semantic Insights Tab
- **Vector Search**: AI-powered semantic search across content
- **Topic Clustering**: Automatic theme discovery and keyword extraction
- **Content Hierarchy**: Sunburst charts showing topic relationships

#### Engagement Metrics Tab
- **Participation Flow**: Visual timeline of speaker engagement
- **Interaction Patterns**: Speaking transitions and momentum analysis
- **Balance Scoring**: Quantified engagement balance metrics

#### AI Content Analysis Tab
- **Technical Discussion Detection**: Identify AI/tech conversations
- **Topic Distribution**: Frequency analysis of technical topics
- **Timeline Mapping**: When AI content was discussed

#### Detailed Reports Tab
- **Custom Reports**: Select sections and formats (PDF, HTML, CSV, JSON)
- **Quick Exports**: One-click data export in multiple formats
- **Visualization Export**: Download charts and graphs

### 4. 🔍 Enhanced Search & Discovery

#### Semantic Search Engine
- **Vector Similarity**: Find conceptually similar content
- **Multi-language Support**: Cross-language semantic understanding
- **Relevance Scoring**: Confidence-based result ranking

#### Advanced Filtering
- **Multi-dimensional**: Filter by speaker, sentiment, time, topics
- **AI Content Focus**: Specialized technical content discovery
- **Custom Queries**: Complex search expressions

### 5. 📋 Professional Export Options

#### Comprehensive Reports
- **HTML Reports**: Modern, responsive design with interactive elements
- **PDF Generation**: Professional layouts with charts and analytics
- **CSV Exports**: Detailed data for further analysis
- **JSON API**: Structured data for system integration

#### Automated Workflows
- **Email Integration**: Scheduled summary delivery
- **Webhook Support**: Real-time notifications
- **API Endpoints**: Enterprise integration capabilities

## 🎯 Meeting Type Optimization

VerbatimAI optimizes analysis based on meeting type:

### **Client Interviews**
- Enhanced sentiment tracking
- Question/answer identification
- Engagement scoring
- Decision point detection

### **Technical Discussions**
- AI content highlighting
- Technical term extraction
- Code/algorithm detection
- Architecture discussion mapping

### **Weekly Standups**
- Action item extraction
- Blocker identification
- Progress tracking
- Team engagement metrics

### **Performance Reviews**
- Goal tracking
- Feedback analysis
- Achievement highlighting
- Development area identification

## 🏗️ Advanced Architecture

### Microservices Design
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Celery Workers │────│  Redis Queue    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  File Storage   │    │   AI Services   │    │   Analytics DB  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Task Queue Architecture
- **High Priority**: Health checks, quick summaries
- **CPU Intensive**: Transcription, semantic analysis
- **Analysis**: Sentiment, meeting insights
- **Reports**: Export generation, email delivery
- **Search**: Semantic queries, similarity analysis
- **Batch**: Multi-file processing

### Scalability Features
- **Horizontal Scaling**: Multiple worker instances
- **Load Balancing**: Intelligent task distribution
- **Caching Layer**: Redis-based performance optimization
- **Resource Management**: Memory and CPU optimization
- Navigate to "🎙️ Upload & Transcribe"
- Upload your audio/video file (supports multiple formats)
- Configure transcription settings (speaker diarization, highlights)
- Click "Start Transcription"
- Wait for processing (typically 1-3 minutes depending on file size)

### 2. 🔴 Real-time Recording
- Navigate to "🔴 Real-time Recording"
- **Audio Only**: Click "🔴 Start Recording"
- **Audio + Video**: Check "🎥 Enable Video Recording" then start
- Monitor live camera preview (video mode)
- Use pause/resume controls as needed
- Click "⏹️ Stop Recording" when finished
- Transcribe recorded content directly

### 3. 📊 Analytics Dashboard
Comprehensive meeting analysis including:
- **Speaker Analysis**: Time distribution, engagement scores, word counts
- **Engagement Metrics**: Participation-based scoring (not sentiment)
- **Key Points Summary**: Decisions, action items, questions
- **Sentiment Trends**: Emotional analysis with interactive charts
- **Speaker Comparison**: Multi-dimensional analytics

### 4. 🔍 Semantic Search
- AI-powered content discovery using sentence transformers
- Search through meeting transcripts semantically
- Find relevant content even with different wording
- Build searchable knowledge base of meetings

### 5. 📚 Meeting Library
- Auto-saves all transcribed meetings
- Browse and search past meetings
- Quick access to analytics and insights
- Export individual meetings

### 6. 📄 Export & Share
- **DOCX**: Professional formatted documents
- **PDF**: Clean reports with charts and analytics
- **Email**: Automated meeting summaries
- **JSON**: Raw data for integration

## 🛠️ Technical Architecture

### Core Technologies
- **Frontend**: Streamlit 1.28.1 (Python web framework)
- **Transcription**: AssemblyAI API with speaker diarization
- **AI/ML**: 
  - Sentence Transformers for semantic search
  - NLTK for text processing
  - scikit-learn for analytics
- **Video**: OpenCV for camera capture and recording
- **Audio**: PyAudio for real-time audio recording
- **Visualization**: Plotly for interactive charts
- **Export**: python-docx, reportlab for document generation
- **Background Processing**: Celery + Redis for async tasks

### Key Components
- `app.py`: Main Streamlit application with all UI components
- `real_time_recorder.py`: Audio/video recording with live preview
- `semantic_search.py`: AI-powered content discovery
- `async_task_queue.py`: Background task processing
- `email_summary.py`: Automated email functionality
- `meeting_summarizer.py`: Advanced meeting analysis
- `enhanced_emotion_detector.py`: Sentiment analysis
- `config.py`: Configuration and settings

### Data Flow
1. **Input**: Audio/Video file or real-time recording
2. **Processing**: AssemblyAI transcription + speaker diarization
3. **Analysis**: AI-powered extraction of insights and metrics
4. **Storage**: Meeting library with metadata
5. **Search**: Semantic indexing for content discovery
6. **Output**: Analytics dashboard + export options

## 🎯 Use Cases

### Business Meetings
- Automatic meeting minutes generation
- Action item tracking and follow-up
- Speaker engagement analysis
- Decision documentation

### Interviews & Research
- Interview transcription with speaker identification
- Key insight extraction
- Sentiment analysis of responses
- Research note organization

### Educational Content
- Lecture transcription and summarization
- Student participation analysis
- Key concept identification
- Study material generation

### Legal & Compliance
- Accurate meeting documentation
- Speaker attribution for statements
- Searchable transcript archive
- Professional report generation

## 🔧 Configuration

### Environment Variables
```env
# Required
ASSEMBLYAI_API_KEY=your_api_key

# Optional - Email functionality
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# Optional - Background processing
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### Recording Settings
```python
RECORDING_CONFIG = {
    'sample_rate': 16000,
    'channels': 1,
    'chunk_size': 1024,
    'max_duration': 3600,  # 1 hour
    'video_fps': 30,
    'video_resolution': (640, 480),
    'video_codec': 'XVID'
}
```

## 📦 Dependencies

### Core Requirements (minimal_requirements.txt)
```
numpy==1.24.3
pandas==2.1.3
streamlit==1.28.1
plotly==5.17.0
assemblyai==0.21.0
requests==2.31.0
python-dotenv==1.0.0
redis
celery
```

### Extended Features (requirements.txt)
- Video recording: `opencv-python`, `pyaudio`
- AI/ML: `sentence-transformers`, `faiss-cpu`, `transformers`
- Audio processing: `librosa`, `soundfile`, `pydub`
- NLP: `nltk`, `spacy`
- Export: `python-docx`, `reportlab`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AssemblyAI** for excellent transcription API
- **Streamlit** for the amazing web framework
- **OpenAI/Anthropic** for AI assistance in development
- **OpenCV** community for video processing capabilities
- **Sentence Transformers** for semantic search functionality

## 📞 Support

- 📧 Email: support@verbatimai.com
- 📝 Issues: [GitHub Issues](https://github.com/your-repo/verbatimai/issues)
- 📖 Documentation: [Wiki](https://github.com/your-repo/verbatimai/wiki)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/verbatimai/discussions)

---

**Made with ❤️ for better meeting intelligence**