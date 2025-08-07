"""
Celery Application for VerbatimAI
Handles async task processing with Redis backend
"""

from celery import Celery
import os
from datetime import timedelta

# Redis configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)

# Create Celery app
app = Celery(
    'verbatimai',
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=[
        'celery_tasks.transcription_tasks',
        'celery_tasks.analysis_tasks',
        'celery_tasks.semantic_tasks',
        'celery_tasks.export_tasks'
    ]
)

# Configuration
app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task routing
    task_routes={
        'celery_tasks.transcription_tasks.transcribe_audio': {'queue': 'transcription'},
        'celery_tasks.analysis_tasks.analyze_sentiment': {'queue': 'analysis'},
        'celery_tasks.semantic_tasks.build_semantic_index': {'queue': 'semantic'},
        'celery_tasks.export_tasks.generate_report': {'queue': 'export'},
    },
    
    # Task execution
    task_always_eager=False,  # Set to True for testing without Redis
    task_eager_propagates=True,
    task_ignore_result=False,
    
    # Results
    result_expires=3600,  # 1 hour
    result_persistent=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-results': {
            'task': 'celery_tasks.maintenance_tasks.cleanup_old_results',
            'schedule': timedelta(hours=6),
        },
        'health-check': {
            'task': 'celery_tasks.maintenance_tasks.health_check',
            'schedule': timedelta(minutes=30),
        },
    },
    beat_scheduler='django_celery_beat.schedulers:DatabaseScheduler',
)

# Task retry settings
app.conf.task_default_retry_delay = 60  # 1 minute
app.conf.task_max_retries = 3

if __name__ == '__main__':
    app.start()
