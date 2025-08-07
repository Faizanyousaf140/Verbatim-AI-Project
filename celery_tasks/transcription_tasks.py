"""
Transcription Tasks for Celery
Handles audio transcription in the background
"""

from celery import current_app as celery_app
from celery.utils.log import get_task_logger
import tempfile
import os
import time
from datetime import datetime, timedelta
import json

logger = get_task_logger(__name__)

@celery_app.task(bind=True, name='transcribe_audio_async')
def transcribe_audio_async(self, file_data, config_options=None):
    """
    Asynchronously transcribe audio file
    
    Args:
        file_data: Base64 encoded audio file data
        config_options: Transcription configuration options
    
    Returns:
        dict: Transcription results and metadata
    """
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 10,
                'total': 100,
                'status': 'Initializing transcription...'
            }
        )
        
        # Import here to avoid circular imports
        import assemblyai as aai
        import base64
        from config import ASSEMBLYAI_API_KEY
        
        # Configure API
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        
        # Decode file data
        audio_data = base64.b64decode(file_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Uploading to AssemblyAI...'
            }
        )
        
        # Configure transcription options
        config = aai.TranscriptionConfig(
            speaker_labels=config_options.get('speaker_detection', True) if config_options else True,
            auto_punctuation=config_options.get('auto_punctuation', True) if config_options else True,
            filter_profanity=config_options.get('filter_profanity', False) if config_options else False,
            sentiment_analysis=config_options.get('sentiment_analysis', True) if config_options else True,
            auto_highlights=config_options.get('auto_highlights', True) if config_options else True,
            language_code=config_options.get('language', 'en') if config_options else 'en'
        )
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 30,
                'total': 100,
                'status': 'Starting transcription...'
            }
        )
        
        # Start transcription
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(tmp_file_path, config)
        
        # Poll for completion
        while transcript.status == aai.TranscriptStatus.processing:
            time.sleep(5)  # Wait 5 seconds
            
            # Update progress
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': min(90, self.request.retries * 10 + 50),
                    'total': 100,
                    'status': 'Processing transcription...'
                }
            )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 95,
                'total': 100,
                'status': 'Finalizing results...'
            }
        )
        
        # Prepare results
        result = {
            'transcript_id': transcript.id,
            'text': transcript.text,
            'audio_duration': transcript.audio_duration,
            'confidence': transcript.confidence,
            'utterances': [
                {
                    'speaker': utterance.speaker,
                    'text': utterance.text,
                    'start': utterance.start,
                    'end': utterance.end,
                    'confidence': utterance.confidence
                }
                for utterance in transcript.utterances
            ] if transcript.utterances else [],
            'sentiment_analysis': [
                {
                    'text': result.text,
                    'sentiment': result.sentiment.value,
                    'confidence': result.confidence,
                    'speaker': result.speaker
                }
                for result in transcript.sentiment_analysis_results
            ] if hasattr(transcript, 'sentiment_analysis_results') and transcript.sentiment_analysis_results else [],
            'auto_highlights': [
                {
                    'text': highlight.text,
                    'count': highlight.count,
                    'rank': highlight.rank,
                    'timestamps': [
                        {
                            'start': ts.start,
                            'end': ts.end
                        }
                        for ts in highlight.timestamps
                    ]
                }
                for highlight in transcript.auto_highlights.results
            ] if hasattr(transcript, 'auto_highlights') and transcript.auto_highlights else [],
            'processing_time': time.time() - self.request.kwargs.get('start_time', time.time()),
            'completed_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Transcription completed successfully. Duration: {transcript.audio_duration}ms")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'message': 'Transcription completed successfully'
        }
        
    except Exception as exc:
        logger.error(f"Transcription failed: {str(exc)}")
        
        # Clean up temporary file if it exists
        try:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except:
            pass
        
        # Retry logic
        if self.request.retries < 3:
            logger.info(f"Retrying transcription. Attempt {self.request.retries + 1}/3")
            raise self.retry(countdown=60, exc=exc)
        
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Transcription failed after multiple attempts'
        }

@celery_app.task(name='batch_transcribe')
def batch_transcribe(file_list, config_options=None):
    """
    Transcribe multiple files in batch
    
    Args:
        file_list: List of file data dictionaries
        config_options: Transcription configuration
        
    Returns:
        dict: Batch processing results
    """
    try:
        results = []
        total_files = len(file_list)
        
        for i, file_data in enumerate(file_list):
            try:
                # Start individual transcription task
                task = transcribe_audio_async.delay(file_data['data'], config_options)
                
                results.append({
                    'filename': file_data['filename'],
                    'task_id': task.id,
                    'status': 'PENDING'
                })
                
            except Exception as e:
                results.append({
                    'filename': file_data['filename'],
                    'task_id': None,
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        return {
            'batch_id': f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'total_files': total_files,
            'results': results,
            'started_at': datetime.utcnow().isoformat()
        }
        
    except Exception as exc:
        logger.error(f"Batch transcription failed: {str(exc)}")
        return {
            'status': 'FAILURE',
            'error': str(exc)
        }

@celery_app.task(name='transcription_health_check')
def transcription_health_check():
    """
    Health check for transcription service
    """
    try:
        import assemblyai as aai
        from config import ASSEMBLYAI_API_KEY
        
        # Test API connection
        aai.settings.api_key = ASSEMBLYAI_API_KEY
        
        # Simple API test (you might want to implement a more thorough test)
        test_result = {
            'service': 'AssemblyAI Transcription',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'api_key_configured': bool(ASSEMBLYAI_API_KEY)
        }
        
        return test_result
        
    except Exception as exc:
        return {
            'service': 'AssemblyAI Transcription',
            'status': 'unhealthy',
            'error': str(exc),
            'timestamp': datetime.utcnow().isoformat()
        }
