"""
Analysis Tasks for Celery
Handles heavy analysis operations in the background
"""

from celery import current_app as celery_app
from celery.utils.log import get_task_logger
from datetime import datetime
import json
import numpy as np

logger = get_task_logger(__name__)

@celery_app.task(bind=True, name='analyze_sentiment_async')
def analyze_sentiment_async(self, transcript_data):
    """
    Asynchronously analyze sentiment of transcript
    
    Args:
        transcript_data: Transcript text or structured data
        
    Returns:
        dict: Sentiment analysis results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Initializing sentiment analysis...'
            }
        )
        
        # Import here to avoid circular imports
        from enhanced_emotion_detector import EnhancedEmotionDetector
        
        detector = EnhancedEmotionDetector()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 40,
                'total': 100,
                'status': 'Processing emotions...'
            }
        )
        
        # Extract text if structured data
        if isinstance(transcript_data, dict):
            text = transcript_data.get('text', '')
            utterances = transcript_data.get('utterances', [])
        else:
            text = transcript_data
            utterances = []
        
        # Analyze overall sentiment
        overall_emotions = detector.analyze_emotions(text)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 70,
                'total': 100,
                'status': 'Analyzing speaker emotions...'
            }
        )
        
        # Analyze speaker-specific emotions
        speaker_emotions = {}
        if utterances:
            for utterance in utterances:
                speaker = utterance.get('speaker', 'Unknown')
                utterance_text = utterance.get('text', '')
                
                if speaker not in speaker_emotions:
                    speaker_emotions[speaker] = []
                
                emotions = detector.analyze_emotions(utterance_text)
                speaker_emotions[speaker].append({
                    'text': utterance_text,
                    'emotions': emotions,
                    'start': utterance.get('start'),
                    'end': utterance.get('end')
                })
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 90,
                'total': 100,
                'status': 'Generating insights...'
            }
        )
        
        # Generate insights
        insights = detector.generate_emotion_insights(overall_emotions, speaker_emotions)
        
        result = {
            'overall_emotions': overall_emotions,
            'speaker_emotions': speaker_emotions,
            'insights': insights,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'total_speakers': len(speaker_emotions),
            'total_utterances': len(utterances)
        }
        
        logger.info(f"Sentiment analysis completed. Analyzed {len(utterances)} utterances for {len(speaker_emotions)} speakers")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'message': 'Sentiment analysis completed successfully'
        }
        
    except Exception as exc:
        logger.error(f"Sentiment analysis failed: {str(exc)}")
        
        if self.request.retries < 2:
            raise self.retry(countdown=30, exc=exc)
        
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Sentiment analysis failed'
        }

@celery_app.task(bind=True, name='generate_semantic_embeddings')
def generate_semantic_embeddings(self, transcript_data):
    """
    Generate semantic embeddings for transcript
    
    Args:
        transcript_data: Transcript text or structured data
        
    Returns:
        dict: Embedding results and analytics
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 15,
                'total': 100,
                'status': 'Loading semantic models...'
            }
        )
        
        # Import here to avoid circular imports
        from semantic_search import SemanticSearch
        
        semantic_engine = SemanticSearch()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 30,
                'total': 100,
                'status': 'Processing text segments...'
            }
        )
        
        # Extract text segments
        if isinstance(transcript_data, dict):
            text = transcript_data.get('text', '')
            utterances = transcript_data.get('utterances', [])
        else:
            text = transcript_data
            utterances = []
        
        # Split into segments for embedding
        segments = []
        if utterances:
            segments = [utterance.get('text', '') for utterance in utterances]
        else:
            # Split by sentences if no utterances
            import re
            segments = re.split(r'[.!?]+', text)
            segments = [seg.strip() for seg in segments if seg.strip()]
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 50,
                'total': 100,
                'status': 'Generating embeddings...'
            }
        )
        
        # Generate embeddings
        embeddings = semantic_engine.generate_embeddings(segments)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 70,
                'total': 100,
                'status': 'Analyzing semantic themes...'
            }
        )
        
        # Find semantic themes
        themes = semantic_engine.find_semantic_themes(segments, embeddings)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 85,
                'total': 100,
                'status': 'Generating analytics...'
            }
        )
        
        # Generate analytics
        analytics = semantic_engine.get_embedding_analytics(segments, embeddings)
        
        # Topic progression analysis
        topic_progression = semantic_engine.analyze_topic_progression(segments, embeddings)
        
        result = {
            'embeddings': embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            'segments': segments,
            'themes': themes,
            'analytics': analytics,
            'topic_progression': topic_progression,
            'embedding_dimension': embeddings.shape[1] if hasattr(embeddings, 'shape') else len(embeddings[0]),
            'total_segments': len(segments),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Semantic embeddings generated for {len(segments)} segments")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'message': 'Semantic embeddings generated successfully'
        }
        
    except Exception as exc:
        logger.error(f"Semantic embedding generation failed: {str(exc)}")
        
        if self.request.retries < 2:
            raise self.retry(countdown=45, exc=exc)
        
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Semantic embedding generation failed'
        }

@celery_app.task(name='analyze_meeting_insights')
def analyze_meeting_insights(transcript_data, meeting_metadata=None):
    """
    Generate comprehensive meeting insights
    
    Args:
        transcript_data: Complete transcript data
        meeting_metadata: Meeting type, duration, participants etc.
        
    Returns:
        dict: Meeting insights and recommendations
    """
    try:
        # Import analysis modules
        from meeting_summarizer import AdvancedMeetingSummarizer
        import statistics
        
        summarizer = AdvancedMeetingSummarizer()
        
        # Extract basic data
        if isinstance(transcript_data, dict):
            text = transcript_data.get('text', '')
            utterances = transcript_data.get('utterances', [])
            duration = transcript_data.get('audio_duration', 0)
        else:
            text = transcript_data
            utterances = []
            duration = 0
        
        # Generate summary using the comprehensive method
        transcript_data_dict = {
            'text': text,
            'utterances': utterances,
            'audio_duration': duration,
            'title': 'Meeting Transcript'
        }
        summary_result = summarizer.generate_comprehensive_summary(transcript_data_dict)
        summary = summary_result.get('summary', 'No summary generated')
        
        # Analyze participation
        speaker_stats = {}
        if utterances:
            for utterance in utterances:
                speaker = utterance.get('speaker', 'Unknown')
                word_count = len(utterance.get('text', '').split())
                
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        'word_count': 0,
                        'utterance_count': 0,
                        'total_time': 0
                    }
                
                speaker_stats[speaker]['word_count'] += word_count
                speaker_stats[speaker]['utterance_count'] += 1
                speaker_stats[speaker]['total_time'] += utterance.get('end', 0) - utterance.get('start', 0)
        
        # Calculate engagement metrics
        total_words = sum(stats['word_count'] for stats in speaker_stats.values())
        engagement_balance = 1.0 - (max(stats['word_count'] for stats in speaker_stats.values()) / total_words) if total_words > 0 else 0
        
        # Meeting type insights
        meeting_type = meeting_metadata.get('type', 'general') if meeting_metadata else 'general'
        
        insights = {
            'summary': summary,
            'speaker_statistics': speaker_stats,
            'engagement_metrics': {
                'total_speakers': len(speaker_stats),
                'total_words': total_words,
                'engagement_balance': engagement_balance,
                'average_words_per_speaker': total_words / len(speaker_stats) if speaker_stats else 0
            },
            'meeting_metadata': {
                'type': meeting_type,
                'duration_ms': duration,
                'duration_minutes': duration / 60000 if duration > 0 else 0,
                'analyzed_at': datetime.utcnow().isoformat()
            },
            'recommendations': []
        }
        
        # Generate recommendations based on meeting type
        if meeting_type == 'standup':
            if insights['engagement_metrics']['engagement_balance'] < 0.3:
                insights['recommendations'].append("Consider encouraging quieter team members to share more")
        elif meeting_type == 'interview':
            interviewer_words = max(stats['word_count'] for stats in speaker_stats.values())
            if interviewer_words / total_words > 0.7:
                insights['recommendations'].append("Interviewer dominated conversation - consider more open-ended questions")
        
        logger.info(f"Meeting insights generated for {meeting_type} meeting with {len(speaker_stats)} speakers")
        
        return {
            'status': 'SUCCESS',
            'result': insights,
            'message': 'Meeting insights generated successfully'
        }
        
    except Exception as exc:
        logger.error(f"Meeting insights analysis failed: {str(exc)}")
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Meeting insights analysis failed'
        }

@celery_app.task(name='export_analysis_data')
def export_analysis_data(analysis_data, export_format='csv'):
    """
    Export analysis data to various formats
    
    Args:
        analysis_data: Analysis results to export
        export_format: Format (csv, json, xlsx)
        
    Returns:
        dict: Export results with download link
    """
    try:
        import pandas as pd
        import tempfile
        import os
        from datetime import datetime
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Prepare data for export
        export_data = []
        
        if 'speaker_emotions' in analysis_data:
            # Sentiment analysis export
            for speaker, emotions_list in analysis_data['speaker_emotions'].items():
                for emotion_data in emotions_list:
                    export_data.append({
                        'speaker': speaker,
                        'text': emotion_data['text'],
                        'primary_emotion': max(emotion_data['emotions'], key=emotion_data['emotions'].get),
                        'confidence': max(emotion_data['emotions'].values()),
                        'start_time': emotion_data.get('start', 0),
                        'end_time': emotion_data.get('end', 0)
                    })
        
        elif 'speaker_statistics' in analysis_data:
            # Meeting insights export
            for speaker, stats in analysis_data['speaker_statistics'].items():
                export_data.append({
                    'speaker': speaker,
                    'word_count': stats['word_count'],
                    'utterance_count': stats['utterance_count'],
                    'total_time_ms': stats['total_time']
                })
        
        # Create DataFrame
        df = pd.DataFrame(export_data)
        
        # Generate filename
        filename = f"verbatim_analysis_{timestamp}.{export_format}"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{export_format}') as tmp_file:
            if export_format == 'csv':
                df.to_csv(tmp_file.name, index=False)
            elif export_format == 'json':
                df.to_json(tmp_file.name, orient='records', indent=2)
            elif export_format == 'xlsx':
                df.to_excel(tmp_file.name, index=False)
            
            # In a real application, you'd upload this to cloud storage
            # For now, we'll return the file path
            return {
                'status': 'SUCCESS',
                'filename': filename,
                'file_path': tmp_file.name,
                'record_count': len(export_data),
                'export_format': export_format,
                'generated_at': datetime.utcnow().isoformat()
            }
    
    except Exception as exc:
        logger.error(f"Export failed: {str(exc)}")
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Data export failed'
        }
