"""
Celery Tasks Module Init
Import all task modules and setup task discovery
"""

# Import all task modules to register tasks with Celery
from .transcription_tasks import *
from .analysis_tasks import *
from .semantic_tasks import *
from .export_tasks import *

# List of all available tasks for easy reference
AVAILABLE_TASKS = [
    # Transcription Tasks
    'transcribe_audio_async',
    'batch_transcribe',
    'transcription_health_check',
    
    # Analysis Tasks
    'analyze_sentiment_async',
    'generate_semantic_embeddings',
    'analyze_meeting_insights',
    'export_analysis_data',
    
    # Semantic Tasks
    'semantic_search_query',
    'map_ai_content_blocks',
    'similarity_analysis',
    'topic_clustering',
    
    # Export Tasks
    'generate_comprehensive_report',
    'export_analytics_csv',
    'generate_quick_summary'
]

# Task routing configuration
TASK_ROUTES = {
    # High-priority, quick tasks
    'transcription_health_check': {'queue': 'high_priority'},
    'generate_quick_summary': {'queue': 'high_priority'},
    
    # CPU-intensive tasks
    'transcribe_audio_async': {'queue': 'cpu_intensive'},
    'generate_semantic_embeddings': {'queue': 'cpu_intensive'},
    'similarity_analysis': {'queue': 'cpu_intensive'},
    
    # Analysis tasks
    'analyze_sentiment_async': {'queue': 'analysis'},
    'analyze_meeting_insights': {'queue': 'analysis'},
    'map_ai_content_blocks': {'queue': 'analysis'},
    'topic_clustering': {'queue': 'analysis'},
    
    # Export and reporting tasks
    'generate_comprehensive_report': {'queue': 'reports'},
    'export_analytics_csv': {'queue': 'reports'},
    'export_analysis_data': {'queue': 'reports'},
    
    # Search tasks
    'semantic_search_query': {'queue': 'search'},
    
    # Batch operations
    'batch_transcribe': {'queue': 'batch_operations'}
}
