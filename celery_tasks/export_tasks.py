"""
Export Tasks for Celery
Handles data export and report generation
"""

from celery import current_app as celery_app
from celery.utils.log import get_task_logger
from datetime import datetime
import json
import tempfile
import os

logger = get_task_logger(__name__)

@celery_app.task(bind=True, name='generate_comprehensive_report')
def generate_comprehensive_report(self, session_data, export_format='pdf'):
    """
    Generate comprehensive meeting report
    
    Args:
        session_data: Complete session data including transcript, analytics, etc.
        export_format: Export format (pdf, docx, html)
        
    Returns:
        dict: Report generation results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 10,
                'total': 100,
                'status': 'Initializing report generation...'
            }
        )
        
        # Extract data components
        transcript_data = session_data.get('transcript', {})
        sentiment_data = session_data.get('sentiment_analysis', {})
        semantic_data = session_data.get('semantic_analysis', {})
        meeting_insights = session_data.get('meeting_insights', {})
        ai_content = session_data.get('ai_content_mapping', {})
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Preparing report structure...'
            }
        )
        
        # Build report structure
        report_data = {
            'title': 'VerbatimAI Meeting Report',
            'generated_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'session_info': {
                'meeting_type': session_data.get('meeting_type', 'General Meeting'),
                'duration': transcript_data.get('audio_duration', 0),
                'participants': len(sentiment_data.get('speaker_emotions', {})),
                'total_words': sum([
                    stats.get('word_count', 0) 
                    for stats in meeting_insights.get('speaker_statistics', {}).values()
                ])
            },
            'executive_summary': {
                'key_points': meeting_insights.get('summary', {}).get('key_points', []),
                'action_items': meeting_insights.get('summary', {}).get('action_items', []),
                'decisions': meeting_insights.get('summary', {}).get('decisions', [])
            },
            'participant_analysis': {},
            'sentiment_overview': {},
            'ai_content_summary': {},
            'semantic_insights': {}
        }
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 40,
                'total': 100,
                'status': 'Analyzing participant data...'
            }
        )
        
        # Process participant analysis
        speaker_stats = meeting_insights.get('speaker_statistics', {})
        speaker_emotions = sentiment_data.get('speaker_emotions', {})
        
        for speaker in speaker_stats.keys():
            stats = speaker_stats[speaker]
            emotions = speaker_emotions.get(speaker, [])
            
            # Calculate dominant emotion
            all_emotions = {}
            for emotion_entry in emotions:
                for emotion, score in emotion_entry.get('emotions', {}).items():
                    all_emotions[emotion] = all_emotions.get(emotion, 0) + score
            
            dominant_emotion = max(all_emotions, key=all_emotions.get) if all_emotions else 'neutral'
            
            report_data['participant_analysis'][speaker] = {
                'word_count': stats['word_count'],
                'utterance_count': stats['utterance_count'],
                'participation_percentage': (stats['word_count'] / report_data['session_info']['total_words'] * 100) if report_data['session_info']['total_words'] > 0 else 0,
                'dominant_emotion': dominant_emotion,
                'emotion_variety': len(all_emotions)
            }
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 60,
                'total': 100,
                'status': 'Processing sentiment analysis...'
            }
        )
        
        # Process sentiment overview
        overall_emotions = sentiment_data.get('overall_emotions', {})
        report_data['sentiment_overview'] = {
            'overall_mood': max(overall_emotions, key=overall_emotions.get) if overall_emotions else 'neutral',
            'emotion_distribution': overall_emotions,
            'sentiment_insights': sentiment_data.get('insights', [])
        }
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 75,
                'total': 100,
                'status': 'Compiling AI content analysis...'
            }
        )
        
        # Process AI content summary
        ai_summary = ai_content.get('summary', {})
        report_data['ai_content_summary'] = {
            'total_mentions': ai_summary.get('total_ai_mentions', 0),
            'high_relevance_mentions': ai_summary.get('high_relevance_mentions', 0),
            'top_keywords': ai_summary.get('top_keywords', []),
            'ai_engagement_by_speaker': {
                speaker: data['mentions'] 
                for speaker, data in ai_content.get('speaker_engagement', {}).items()
            }
        }
        
        # Process semantic insights
        themes = semantic_data.get('themes', [])
        report_data['semantic_insights'] = {
            'discovered_themes': themes[:5],  # Top 5 themes
            'topic_progression': semantic_data.get('topic_progression', {}),
            'semantic_diversity': len(themes)
        }
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 85,
                'total': 100,
                'status': f'Generating {export_format.upper()} report...'
            }
        )
        
        # Generate report file
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"verbatim_report_{timestamp}.{export_format}"
        
        if export_format == 'json':
            # JSON export
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                json.dump(report_data, tmp_file, indent=2, default=str)
                file_path = tmp_file.name
        
        elif export_format == 'html':
            # HTML export
            html_content = generate_html_report(report_data)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html', encoding='utf-8') as tmp_file:
                tmp_file.write(html_content)
                file_path = tmp_file.name
        
        elif export_format == 'csv':
            # CSV export (flattened data)
            import pandas as pd
            
            # Flatten participant data for CSV
            csv_data = []
            for speaker, data in report_data['participant_analysis'].items():
                csv_data.append({
                    'speaker': speaker,
                    'word_count': data['word_count'],
                    'utterance_count': data['utterance_count'],
                    'participation_percentage': data['participation_percentage'],
                    'dominant_emotion': data['dominant_emotion'],
                    'emotion_variety': data['emotion_variety']
                })
            
            df = pd.DataFrame(csv_data)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                file_path = tmp_file.name
        
        else:
            # Default to JSON if unsupported format
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                json.dump(report_data, tmp_file, indent=2, default=str)
                file_path = tmp_file.name
        
        result = {
            'filename': filename,
            'file_path': file_path,
            'export_format': export_format,
            'report_summary': {
                'total_participants': len(report_data['participant_analysis']),
                'meeting_duration_minutes': report_data['session_info']['duration'] / 60000 if report_data['session_info']['duration'] > 0 else 0,
                'total_words': report_data['session_info']['total_words'],
                'ai_mentions': report_data['ai_content_summary']['total_mentions'],
                'themes_discovered': report_data['semantic_insights']['semantic_diversity']
            },
            'generated_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Comprehensive report generated: {filename}")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'message': 'Comprehensive report generated successfully'
        }
        
    except Exception as exc:
        logger.error(f"Report generation failed: {str(exc)}")
        
        if self.request.retries < 2:
            raise self.retry(countdown=30, exc=exc)
        
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Report generation failed'
        }

def generate_html_report(report_data):
    """Generate HTML report from report data"""
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_data['title']}</title>
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .report-container {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }}
        
        .header {{
            text-align: center;
            border-bottom: 2px solid #e1e5e9;
            padding-bottom: 30px;
            margin-bottom: 40px;
        }}
        
        .header h1 {{
            color: #1e3a8a;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #1e3a8a;
            border-left: 4px solid #3b82f6;
            padding-left: 20px;
            font-size: 1.5em;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f8fafc, #e2e8f0);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #1e3a8a;
        }}
        
        .metric-label {{
            color: #64748b;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .participant-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        .participant-table th,
        .participant-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        .participant-table th {{
            background: #f1f5f9;
            color: #1e3a8a;
            font-weight: 600;
        }}
        
        .emotion-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        
        .emotion-positive {{ background: #dcfce7; color: #166534; }}
        .emotion-negative {{ background: #fef2f2; color: #dc2626; }}
        .emotion-neutral {{ background: #f1f5f9; color: #475569; }}
        
        .keyword-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 15px 0;
        }}
        
        .keyword {{
            background: #dbeafe;
            color: #1e40af;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.85em;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <h1>{report_data['title']}</h1>
            <p>Generated on {report_data['generated_at']}</p>
        </div>
        
        <div class="section">
            <h2>Meeting Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{report_data['session_info']['participants']}</div>
                    <div class="metric-label">Participants</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report_data['session_info']['duration'] // 60000:.0f}m</div>
                    <div class="metric-label">Duration</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report_data['session_info']['total_words']:,}</div>
                    <div class="metric-label">Total Words</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report_data['ai_content_summary']['total_mentions']}</div>
                    <div class="metric-label">AI Mentions</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Participant Analysis</h2>
            <table class="participant-table">
                <thead>
                    <tr>
                        <th>Speaker</th>
                        <th>Words</th>
                        <th>Participation %</th>
                        <th>Dominant Emotion</th>
                        <th>Utterances</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add participant rows
    for speaker, data in report_data['participant_analysis'].items():
        emotion_class = 'emotion-positive' if data['dominant_emotion'] in ['joy', 'happiness', 'excitement'] else 'emotion-negative' if data['dominant_emotion'] in ['anger', 'sadness', 'fear'] else 'emotion-neutral'
        
        html_template += f"""
                    <tr>
                        <td><strong>{speaker}</strong></td>
                        <td>{data['word_count']:,}</td>
                        <td>{data['participation_percentage']:.1f}%</td>
                        <td><span class="emotion-badge {emotion_class}">{data['dominant_emotion'].title()}</span></td>
                        <td>{data['utterance_count']}</td>
                    </tr>
        """
    
    html_template += f"""
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>Sentiment Overview</h2>
            <p><strong>Overall Mood:</strong> <span class="emotion-badge emotion-neutral">{report_data['sentiment_overview']['overall_mood'].title()}</span></p>
        </div>
        
        <div class="section">
            <h2>AI Content Analysis</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{report_data['ai_content_summary']['total_mentions']}</div>
                    <div class="metric-label">Total AI Mentions</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{report_data['ai_content_summary']['high_relevance_mentions']}</div>
                    <div class="metric-label">High Relevance</div>
                </div>
            </div>
            
            <h3>Top AI Keywords</h3>
            <div class="keyword-list">
    """
    
    # Add AI keywords
    for keyword in report_data['ai_content_summary']['top_keywords'][:10]:
        html_template += f'<span class="keyword">{keyword}</span>'
    
    html_template += f"""
            </div>
        </div>
        
        <div class="section">
            <h2>Semantic Insights</h2>
            <p><strong>Themes Discovered:</strong> {report_data['semantic_insights']['semantic_diversity']}</p>
        </div>
        
        <div class="footer">
            <p>Report generated by VerbatimAI - Advanced Meeting Analytics Platform</p>
        </div>
    </div>
</body>
</html>
    """
    
    return html_template

@celery_app.task(name='export_analytics_csv')
def export_analytics_csv(analytics_data, export_type='all'):
    """
    Export analytics data to CSV format
    
    Args:
        analytics_data: Analytics data to export
        export_type: Type of export (all, sentiment, semantic, participants)
        
    Returns:
        dict: Export results
    """
    try:
        import pandas as pd
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        if export_type == 'sentiment':
            # Export sentiment analysis data
            data = []
            for speaker, emotions_list in analytics_data.get('speaker_emotions', {}).items():
                for emotion_entry in emotions_list:
                    data.append({
                        'speaker': speaker,
                        'text': emotion_entry['text'],
                        'primary_emotion': max(emotion_entry['emotions'], key=emotion_entry['emotions'].get),
                        'confidence': max(emotion_entry['emotions'].values()),
                        'start_time': emotion_entry.get('start', 0),
                        'end_time': emotion_entry.get('end', 0)
                    })
            
            filename = f"sentiment_analysis_{timestamp}.csv"
            
        elif export_type == 'participants':
            # Export participant statistics
            data = []
            for speaker, stats in analytics_data.get('speaker_statistics', {}).items():
                data.append({
                    'speaker': speaker,
                    'word_count': stats['word_count'],
                    'utterance_count': stats['utterance_count'],
                    'total_time_ms': stats['total_time']
                })
            
            filename = f"participant_stats_{timestamp}.csv"
            
        else:
            # Export all analytics data (flattened)
            data = [analytics_data]  # Single row with all data
            filename = f"complete_analytics_{timestamp}.csv"
        
        # Create DataFrame and export
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            
            return {
                'status': 'SUCCESS',
                'filename': filename,
                'file_path': tmp_file.name,
                'record_count': len(data),
                'export_type': export_type,
                'generated_at': datetime.utcnow().isoformat()
            }
    
    except Exception as exc:
        logger.error(f"CSV export failed: {str(exc)}")
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'CSV export failed'
        }

@celery_app.task(name='generate_quick_summary')
def generate_quick_summary(session_data):
    """
    Generate a quick summary for immediate viewing
    
    Args:
        session_data: Session data
        
    Returns:
        dict: Quick summary
    """
    try:
        transcript_data = session_data.get('transcript', {})
        sentiment_data = session_data.get('sentiment_analysis', {})
        
        # Calculate basic metrics
        total_words = len(transcript_data.get('text', '').split())
        total_speakers = len(sentiment_data.get('speaker_emotions', {}))
        duration_minutes = transcript_data.get('audio_duration', 0) / 60000
        
        # Extract top emotions
        overall_emotions = sentiment_data.get('overall_emotions', {})
        dominant_emotion = max(overall_emotions, key=overall_emotions.get) if overall_emotions else 'neutral'
        
        summary = {
            'meeting_type': session_data.get('meeting_type', 'General'),
            'duration_minutes': round(duration_minutes, 1),
            'total_speakers': total_speakers,
            'total_words': total_words,
            'words_per_minute': round(total_words / duration_minutes, 1) if duration_minutes > 0 else 0,
            'dominant_emotion': dominant_emotion,
            'emotion_scores': overall_emotions,
            'quick_insights': [
                f"Meeting lasted {round(duration_minutes, 1)} minutes",
                f"{total_speakers} participants contributed",
                f"Overall mood was {dominant_emotion}",
                f"Average speaking pace: {round(total_words / duration_minutes, 1) if duration_minutes > 0 else 0} words/minute"
            ],
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return {
            'status': 'SUCCESS',
            'result': summary,
            'message': 'Quick summary generated successfully'
        }
        
    except Exception as exc:
        logger.error(f"Quick summary generation failed: {str(exc)}")
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Quick summary generation failed'
        }
