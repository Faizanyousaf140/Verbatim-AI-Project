"""
Email Summary Module
Generates and sends weekly meeting summaries with engagement scores
"""
import smtplib
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from config import EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT, EMAIL_SENDER, EMAIL_PASSWORD

# Optional pandas import
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except (ImportError, ValueError) as e:
    print(f"⚠️ Pandas not available in email_summary: {e}")
    PANDAS_AVAILABLE = False
    # Simple mock DataFrame
    class MockDataFrame:
        def __init__(self, data=None):
            self.data = data or []
        def to_html(self, *args, **kwargs):
            return "<table><tr><td>Data not available (pandas not installed)</td></tr></table>"
    pd = type('MockPandas', (), {'DataFrame': MockDataFrame})()

class EmailSummaryGenerator:
    def __init__(self):
        """Initialize the email summary generator"""
        self.smtp_server = EMAIL_SMTP_SERVER
        self.smtp_port = EMAIL_SMTP_PORT
        self.sender_email = EMAIL_SENDER
        self.sender_password = EMAIL_PASSWORD
        
        # Validate email configuration
        if not self.sender_email or not self.sender_password:
            print("⚠️ Email configuration incomplete. Email functionality will be disabled.")
            self.sender_email = None
            self.sender_password = None
        else:
            print("✅ Email configuration loaded successfully")
    
    def calculate_engagement_score(self, transcript_data: Dict) -> Dict:
        """Calculate engagement score based on various metrics"""
        try:
            # Extract metrics
            total_duration = transcript_data.get('duration_minutes', 0)
            total_words = transcript_data.get('total_words', 0)
            speakers = transcript_data.get('speakers', [])
            utterances = transcript_data.get('utterances', [])
            
            # Calculate participation score based on speaking distribution
            participation_score = 0
            if speakers and utterances:
                # Calculate based on how evenly distributed the speaking time is
                speaker_times = {}
                for utterance in utterances:
                    speaker = utterance.speaker
                    duration = (utterance.end - utterance.start) / 1000 / 60  # minutes
                    speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
                
                if speaker_times:
                    # Calculate participation balance (how evenly distributed)
                    total_time = sum(speaker_times.values())
                    if total_time > 0:
                        # Calculate standard deviation of speaking times
                        mean_time = total_time / len(speaker_times)
                        variance = sum((time - mean_time) ** 2 for time in speaker_times.values()) / len(speaker_times)
                        std_dev = variance ** 0.5
                        
                        # Participation score based on balance (lower std dev = higher score)
                        max_std = mean_time * 2  # Maximum expected deviation
                        balance_score = max(0, 1 - (std_dev / max_std)) if max_std > 0 else 0
                        participation_score = balance_score * 25
                    else:
                        participation_score = 0
            
            # Calculate interaction score
            interaction_score = 0
            if utterances:
                avg_utterances_per_minute = len(utterances) / max(total_duration, 1)
                interaction_score = min(avg_utterances_per_minute / 10, 1.0) * 25
            
            # Calculate sentiment score
            sentiment_score = 0
            positive_utterances = 0
            total_sentiment_utterances = 0
            
            for utterance in utterances:
                if hasattr(utterance, 'sentiment') and utterance.sentiment:
                    total_sentiment_utterances += 1
                    if utterance.sentiment.get('sentiment', '').lower() in ['positive', 'enthusiastic']:
                        positive_utterances += 1
            
            if total_sentiment_utterances > 0:
                sentiment_score = (positive_utterances / total_sentiment_utterances) * 25
            
            # Calculate action items score
            action_score = 0
            key_points = transcript_data.get('key_points', {})
            action_items = key_points.get('action_items', [])
            if action_items:
                action_score = min(len(action_items) / 5, 1.0) * 25
            
            # Total engagement score
            total_score = participation_score + interaction_score + sentiment_score + action_score
            
            return {
                'total_score': round(total_score, 1),
                'participation_score': round(participation_score, 1),
                'interaction_score': round(interaction_score, 1),
                'sentiment_score': round(sentiment_score, 1),
                'action_score': round(action_score, 1),
                'level': self._get_engagement_level(total_score)
            }
            
        except Exception as e:
            print(f"Error calculating engagement score: {e}")
            return {
                'total_score': 0,
                'participation_score': 0,
                'interaction_score': 0,
                'sentiment_score': 0,
                'action_score': 0,
                'level': 'Unknown'
            }
    
    def _get_engagement_level(self, score: float) -> str:
        """Get engagement level based on score"""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Moderate"
        elif score >= 20:
            return "Low"
        else:
            return "Poor"
    
    def generate_weekly_summary(self, meetings_data: List[Dict]) -> str:
        """Generate weekly summary HTML content"""
        if not meetings_data:
            return "<p>No meetings found for this week.</p>"
        
        # Calculate weekly statistics
        total_meetings = len(meetings_data)
        total_participants = sum(len(m.get('speakers', [])) for m in meetings_data)
        
        # Generate HTML content
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f2f6; padding: 20px; border-radius: 10px; }}
                .stats {{ display: flex; justify-content: space-between; margin: 20px 0; }}
                .stat-box {{ background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; margin: 0 10px; text-align: center; }}
                .meeting-list {{ margin-top: 20px; }}
                .meeting-item {{ background-color: #ffffff; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Weekly Meeting Summary</h1>
                <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>📅 Total Meetings</h3>
                    <h2>{total_meetings}</h2>
                </div>
                <div class="stat-box">
                    <h3>👥 Participants</h3>
                    <h2>{total_participants}</h2>
                </div>
            </div>
            
            <div class="meeting-list">
                <h2>📋 Meeting Details</h2>
        """
        
        for i, meeting in enumerate(meetings_data, 1):
            speakers = meeting.get('speakers', [])
            key_points = meeting.get('key_points', {})
            
            html_content += f"""
                <div class="meeting-item">
                    <h3>Meeting {i}: {meeting.get('title', 'Untitled Meeting')}</h3>
                    <p><strong>Participants:</strong> {len(speakers)} speakers</p>
                    <p><strong>Key Points:</strong></p>
                    <ul>
                        <li>Decisions: {len(key_points.get('decisions', []))}</li>
                        <li>Action Items: {len(key_points.get('action_items', []))}</li>
                        <li>Questions: {len(key_points.get('questions', []))}</li>
                    </ul>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content

    def send_email(self, recipient_email: str, subject: str, html_content: str) -> bool:
        """Send email with HTML content"""
        if not self.sender_email or not self.sender_password:
            print("❌ Email configuration not available. Cannot send email.")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            print(f"✅ Email sent successfully to {recipient_email}")
            return True
            
        except Exception as e:
            import traceback
            print("❌ Exception occurred while sending email:")
            traceback.print_exc()
            return False

    def send_weekly_summary(self, recipient_email: str, meetings_data: List[Dict]) -> bool:
        """Send weekly summary email"""
        subject = f"Weekly Meeting Summary - {datetime.now().strftime('%B %d, %Y')}"
        html_content = self.generate_weekly_summary(meetings_data)
        
        # 🔍 Debug print of the generated HTML
        print("📄 Generated HTML content:")
        print(html_content)
        
        return self.send_email(recipient_email, subject, html_content)

# Utility functions
def get_weekly_meetings(meetings_data: List[Dict], days: int = 7) -> List[Dict]:
    """Get meetings from the last N days"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    weekly_meetings = []
    for meeting in meetings_data:
        meeting_date = meeting.get('date', datetime.now())
        if isinstance(meeting_date, str):
            try:
                meeting_date = datetime.strptime(meeting_date, '%Y-%m-%d')
            except:
                meeting_date = datetime.now()
        
        if meeting_date >= cutoff_date:
            weekly_meetings.append(meeting)
    
    return weekly_meetings

def format_engagement_report(engagement_data: Dict) -> str:
    """Format engagement data for display"""
    return f"""
    **Engagement Score: {engagement_data['total_score']}% ({engagement_data['level']})**
    
    - Participation: 0%
    - Interaction: {engagement_data['interaction_score']}%
    - Sentiment: {engagement_data['sentiment_score']}%
    - Action Items: {engagement_data['action_score']}%
    """
