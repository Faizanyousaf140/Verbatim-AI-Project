import os
import re
import tempfile
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
from config import MAX_FILE_SIZE

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"

def format_timestamp(milliseconds: int) -> str:
    """Format timestamp in milliseconds to MM:SS format"""
    seconds = milliseconds / 1000
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"

def validate_audio_file(file) -> Dict[str, Any]:
    """Validate uploaded audio file"""
    if file is None:
        return {"valid": False, "error": "No file uploaded"}
    
    # Check file size (use MAX_FILE_SIZE from config)
    if file.size > MAX_FILE_SIZE:
        return {"valid": False, "error": f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB"}
    
    # Check file extension
    allowed_extensions = ['mp3', 'wav', 'mp4', 'avi', 'mov', 'm4a']
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension not in allowed_extensions:
        return {"valid": False, "error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"}
    
    return {"valid": True, "file_extension": file_extension}

def extract_speaker_stats(utterances) -> Dict[str, Any]:
    """Extract speaker statistics from utterances"""
    speaker_stats = {}
    
    for utterance in utterances:
        speaker = f"Speaker {utterance.speaker}"
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'utterance_count': 0,
                'total_duration': 0,
                'word_count': 0,
                'avg_sentiment': {'positive': 0, 'negative': 0, 'neutral': 0}
            }
        
        # Update stats
        speaker_stats[speaker]['utterance_count'] += 1
        duration = (utterance.end - utterance.start) / 1000
        speaker_stats[speaker]['total_duration'] += duration
        speaker_stats[speaker]['word_count'] += len(utterance.text.split())
        
        # Sentiment (if available)
        if hasattr(utterance, 'sentiment') and utterance.sentiment:
            speaker_stats[speaker]['avg_sentiment']['positive'] += utterance.sentiment.positive
            speaker_stats[speaker]['avg_sentiment']['negative'] += utterance.sentiment.negative
            speaker_stats[speaker]['avg_sentiment']['neutral'] += utterance.sentiment.neutral
    
    # Calculate averages
    for speaker in speaker_stats:
        count = speaker_stats[speaker]['utterance_count']
        if count > 0:
            speaker_stats[speaker]['avg_sentiment']['positive'] /= count
            speaker_stats[speaker]['avg_sentiment']['negative'] /= count
            speaker_stats[speaker]['avg_sentiment']['neutral'] /= count
    
    return speaker_stats

def create_transcript_dataframe(utterances) -> pd.DataFrame:
    """Create a pandas DataFrame from utterances"""
    data = []
    
    for utterance in utterances:
        data.append({
            'Speaker': f"Speaker {utterance.speaker}",
            'Text': utterance.text,
            'Start Time': format_timestamp(utterance.start),
            'End Time': format_timestamp(utterance.end),
            'Duration (s)': (utterance.end - utterance.start) / 1000,
            'Word Count': len(utterance.text.split())
        })
    
    return pd.DataFrame(data)

def extract_meeting_summary(transcript) -> Dict[str, Any]:
    """Extract meeting summary statistics"""
    total_duration = transcript.audio_duration / 1000 / 60  # minutes
    total_words = len(transcript.text.split())
    total_utterances = len(transcript.utterances)
    unique_speakers = len(set([u.speaker for u in transcript.utterances]))
    
    # Calculate speaking rate
    speaking_rate = total_words / total_duration if total_duration > 0 else 0
    
    # Calculate average utterance length
    avg_utterance_length = total_words / total_utterances if total_utterances > 0 else 0
    
    return {
        'total_duration_minutes': total_duration,
        'total_words': total_words,
        'total_utterances': total_utterances,
        'unique_speakers': unique_speakers,
        'speaking_rate_wpm': speaking_rate,
        'avg_utterance_length': avg_utterance_length
    }

def clean_text_for_export(text: str) -> str:
    """Clean text for export formatting"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues in exports
    text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)]', '', text)
    return text.strip()

def create_meeting_metadata(transcript, filename: str = None) -> Dict[str, Any]:
    """Create meeting metadata for exports"""
    summary = extract_meeting_summary(transcript)
    
    return {
        'filename': filename or 'Unknown',
        'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'meeting_duration': f"{summary['total_duration_minutes']:.1f} minutes",
        'total_words': summary['total_words'],
        'total_utterances': summary['total_utterances'],
        'unique_speakers': summary['unique_speakers'],
        'speaking_rate': f"{summary['speaking_rate_wpm']:.1f} words per minute"
    }

def find_key_phrases(text: str, keywords: List[str]) -> List[str]:
    """Find sentences containing key phrases"""
    sentences = re.split(r'[.!?]+', text)
    key_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower().strip()
        if any(keyword.lower() in sentence_lower for keyword in keywords):
            key_sentences.append(sentence.strip())
    
    return key_sentences

def calculate_sentiment_trends(sentiment_data: Dict[str, Any]) -> pd.DataFrame:
    """Calculate sentiment trends over time"""
    trend_data = []
    
    for speaker, data in sentiment_data.items():
        for utterance in data['utterances']:
            sentiment_scores = utterance['sentiment']
            dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            
            trend_data.append({
                'Speaker': speaker,
                'Timestamp': utterance['timestamp'],
                'Dominant_Sentiment': dominant_sentiment,
                'Positive_Score': sentiment_scores['positive'],
                'Negative_Score': sentiment_scores['negative'],
                'Neutral_Score': sentiment_scores['neutral']
            })
    
    df_trends = pd.DataFrame(trend_data)
    df_trends['Time_Minutes'] = df_trends['Timestamp'] / 60
    
    return df_trends

def generate_insights_report(transcript, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate insights report from transcript and analysis"""
    summary = extract_meeting_summary(transcript)
    key_points = analysis_results['key_points']
    
    insights = {
        'meeting_overview': {
            'duration': f"{summary['total_duration_minutes']:.1f} minutes",
            'participants': summary['unique_speakers'],
            'engagement_level': 'High' if summary['speaking_rate_wpm'] > 150 else 'Medium' if summary['speaking_rate_wpm'] > 100 else 'Low'
        },
        'key_findings': {
            'decisions_made': len(key_points['decisions']),
            'action_items': len(key_points['action_items']),
            'questions_asked': len(key_points['questions']),
            'highlights': len(key_points['highlights'])
        },
        'recommendations': []
    }
    
    # Generate recommendations based on analysis
    if len(key_points['action_items']) == 0:
        insights['recommendations'].append("Consider adding more specific action items for better follow-up")
    
    if summary['unique_speakers'] < 2:
        insights['recommendations'].append("Meeting had limited participation - consider encouraging more engagement")
    
    if summary['speaking_rate_wpm'] > 200:
        insights['recommendations'].append("High speaking rate detected - consider slowing down for better clarity")
    
    return insights 