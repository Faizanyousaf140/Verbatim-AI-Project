"""
Meeting Library/History Management Module
Handles storage, retrieval, and management of meeting transcripts and summaries
"""

import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import re

class MeetingLibrary:
    def __init__(self, meetings_dir: str = "meetings"):
        self.meetings_dir = meetings_dir
        self.ensure_directory_exists()
        self.metadata_file = os.path.join(meetings_dir, "meeting_metadata.json")
        self.metadata = self.load_metadata()
    
    def ensure_directory_exists(self):
        """Ensure the meetings directory exists"""
        os.makedirs(self.meetings_dir, exist_ok=True)
    
    def load_metadata(self) -> Dict:
        """Load meeting metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return {}
        return {}
    
    def save_metadata(self):
        """Save meeting metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def add_meeting(self, meeting_data: Dict) -> str:
        """Add a new meeting to the library"""
        try:
            # Generate unique meeting ID
            meeting_id = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add timestamp and ID to meeting data
            meeting_data['id'] = meeting_id
            meeting_data['created_at'] = datetime.now().isoformat()
            meeting_data['last_updated'] = datetime.now().isoformat()
            
            # Save the meeting data
            meeting_file = os.path.join(self.meetings_dir, f"{meeting_id}.json")
            with open(meeting_file, 'w', encoding='utf-8') as f:
                json.dump(meeting_data, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            self.metadata[meeting_id] = {
                'title': meeting_data.get('title', 'Untitled Meeting'),
                'date': meeting_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'duration': meeting_data.get('duration', 0),
                'participants': meeting_data.get('participants', []),
                'file_path': meeting_file,
                'created_at': meeting_data['created_at'],
                'tags': meeting_data.get('tags', []),
                'summary_available': bool(meeting_data.get('summary', '')),
                'transcript_length': len(meeting_data.get('transcript', ''))
            }
            
            self.save_metadata()
            return meeting_id
            
        except Exception as e:
            print(f"Error adding meeting: {e}")
            return ""
    
    def get_meeting(self, meeting_id: str) -> Optional[Dict]:
        """Retrieve a specific meeting by ID"""
        try:
            meeting_file = os.path.join(self.meetings_dir, f"{meeting_id}.json")
            if os.path.exists(meeting_file):
                with open(meeting_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error retrieving meeting {meeting_id}: {e}")
        return None
    
    def get_all_meetings(self) -> List[Dict]:
        """Get all meetings metadata"""
        meetings = []
        for meeting_id, metadata in self.metadata.items():
            metadata['id'] = meeting_id
            meetings.append(metadata)
        
        # Sort by creation date (newest first)
        meetings.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return meetings
    
    def search_meetings(self, query: str, search_field: str = 'all') -> List[Dict]:
        """Search meetings by title, content, or participants"""
        results = []
        query_lower = query.lower()
        
        for meeting_id, metadata in self.metadata.items():
            match = False
            
            if search_field in ['all', 'title']:
                if query_lower in metadata.get('title', '').lower():
                    match = True
            
            if search_field in ['all', 'participants']:
                participants = metadata.get('participants', [])
                if any(query_lower in p.lower() for p in participants):
                    match = True
            
            if search_field in ['all', 'content']:
                # Search in full meeting content
                meeting_data = self.get_meeting(meeting_id)
                if meeting_data:
                    transcript = meeting_data.get('transcript', '')
                    summary = meeting_data.get('summary', '')
                    if query_lower in transcript.lower() or query_lower in summary.lower():
                        match = True
            
            if match:
                metadata_copy = metadata.copy()
                metadata_copy['id'] = meeting_id
                results.append(metadata_copy)
        
        return results
    
    def delete_meeting(self, meeting_id: str) -> bool:
        """Delete a meeting from the library"""
        try:
            # Remove from metadata
            if meeting_id in self.metadata:
                del self.metadata[meeting_id]
                self.save_metadata()
            
            # Remove meeting file
            meeting_file = os.path.join(self.meetings_dir, f"{meeting_id}.json")
            if os.path.exists(meeting_file):
                os.remove(meeting_file)
            
            return True
        except Exception as e:
            print(f"Error deleting meeting {meeting_id}: {e}")
            return False
    
    def update_meeting(self, meeting_id: str, updated_data: Dict) -> bool:
        """Update an existing meeting"""
        try:
            meeting_data = self.get_meeting(meeting_id)
            if not meeting_data:
                return False
            
            # Update the data
            meeting_data.update(updated_data)
            meeting_data['last_updated'] = datetime.now().isoformat()
            
            # Save updated data
            meeting_file = os.path.join(self.meetings_dir, f"{meeting_id}.json")
            with open(meeting_file, 'w', encoding='utf-8') as f:
                json.dump(meeting_data, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            self.metadata[meeting_id].update({
                'title': meeting_data.get('title', 'Untitled Meeting'),
                'date': meeting_data.get('date', ''),
                'duration': meeting_data.get('duration', 0),
                'participants': meeting_data.get('participants', []),
                'tags': meeting_data.get('tags', []),
                'summary_available': bool(meeting_data.get('summary', '')),
                'transcript_length': len(meeting_data.get('transcript', ''))
            })
            
            self.save_metadata()
            return True
            
        except Exception as e:
            print(f"Error updating meeting {meeting_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get library statistics"""
        meetings = self.get_all_meetings()
        
        if not meetings:
            return {
                'total_meetings': 0,
                'meetings_this_week': 0,
                'meetings_this_month': 0
            }
        
        # Calculate statistics
        total_meetings = len(meetings)
        
        # Time-based statistics
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        meetings_this_week = 0
        meetings_this_month = 0
        
        for meeting in meetings:
            try:
                created_at = datetime.fromisoformat(meeting['created_at'])
                if created_at >= week_ago:
                    meetings_this_week += 1
                if created_at >= month_ago:
                    meetings_this_month += 1
            except:
                continue
        
        return {
            'total_meetings': total_meetings,
            'meetings_this_week': meetings_this_week,
            'meetings_this_month': meetings_this_month
        }

def show_meeting_library():
    """Streamlit UI for meeting library"""
    st.header("📚 Meeting Library & History")
    
    # Initialize library
    if 'meeting_library' not in st.session_state:
        st.session_state.meeting_library = MeetingLibrary()
    
    library = st.session_state.meeting_library
    
    # Statistics section
    st.subheader("📊 Library Statistics")
    stats = library.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Meetings", stats['total_meetings'])
    with col2:
        st.metric("This Week", stats['meetings_this_week'])
    with col3:
        st.metric("This Month", stats['meetings_this_month'])
    
    # Search and filter section
    st.subheader("🔍 Search & Filter")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search meetings...", placeholder="Enter title, participant name, or content")
    with col2:
        search_field = st.selectbox("Search in:", ['all', 'title', 'participants', 'content'])
    
    # Get meetings (either all or filtered)
    if search_query:
        meetings = library.search_meetings(search_query, search_field)
        st.info(f"Found {len(meetings)} meetings matching '{search_query}'")
    else:
        meetings = library.get_all_meetings()
    
    # Display meetings
    if not meetings:
        st.info("No meetings found. Transcribe a meeting to see it here.")
        return
    
    st.subheader(f"📋 Meetings ({len(meetings)})")
    
    # Display meetings in a table format
    for i, meeting in enumerate(meetings):
        with st.expander(f"📄 {meeting.get('title', 'Untitled Meeting')} - {meeting.get('date', 'No date')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Created:** {meeting.get('created_at', 'Unknown')[:16]}")
                
                participants = meeting.get('participants', [])
                if participants:
                    st.write(f"**Participants:** {', '.join(participants)}")
                
                tags = meeting.get('tags', [])
                if tags:
                    st.write(f"**Tags:** {', '.join(tags)}")
                
                if meeting.get('summary_available'):
                    st.success("✅ Summary available")
                else:
                    st.warning("⚠️ No summary")
                
                st.write(f"**Transcript length:** {meeting.get('transcript_length', 0)} characters")
            
            with col2:
                # Action buttons
                if st.button("👁️ View", key=f"view_{i}"):
                    st.session_state.selected_meeting = meeting['id']
                    st.session_state.page = "view_meeting"
                
                if st.button("📝 Edit", key=f"edit_{i}"):
                    st.session_state.selected_meeting = meeting['id']
                    st.session_state.page = "edit_meeting"
                
                if st.button("🗑️ Delete", key=f"delete_{i}"):
                    if st.session_state.get(f"confirm_delete_{i}"):
                        if library.delete_meeting(meeting['id']):
                            st.success("Meeting deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete meeting")
                    else:
                        st.session_state[f"confirm_delete_{i}"] = True
                        st.warning("Click again to confirm deletion")
    
    # Handle page navigation
    if st.session_state.get('page') == 'view_meeting':
        show_meeting_details()
    elif st.session_state.get('page') == 'edit_meeting':
        show_meeting_editor()

def show_meeting_details():
    """Show detailed view of a selected meeting"""
    if 'selected_meeting' not in st.session_state:
        st.error("No meeting selected")
        return
    
    library = st.session_state.meeting_library
    meeting_data = library.get_meeting(st.session_state.selected_meeting)
    
    if not meeting_data:
        st.error("Meeting not found")
        return
    
    st.header(f"📄 {meeting_data.get('title', 'Untitled Meeting')}")
    
    if st.button("← Back to Library"):
        st.session_state.page = None
        st.rerun()
    
    # Meeting details
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Date:** {meeting_data.get('date', 'Unknown')}")
        st.write(f"**Duration:** {meeting_data.get('duration', 0):.1f} minutes")
    with col2:
        st.write(f"**Created:** {meeting_data.get('created_at', 'Unknown')[:16]}")
        st.write(f"**Last Updated:** {meeting_data.get('last_updated', 'Unknown')[:16]}")
    
    # Participants
    participants = meeting_data.get('participants', [])
    if participants:
        st.write(f"**Participants:** {', '.join(participants)}")
    
    # Summary
    summary = meeting_data.get('summary', '')
    if summary:
        st.subheader("📋 Summary")
        st.write(summary)
    
    # Transcript
    transcript = meeting_data.get('transcript', '')
    if transcript:
        st.subheader("📜 Full Transcript")
        st.text_area("Transcript", transcript, height=300, disabled=True)

def show_meeting_editor():
    """Show editor for meeting metadata"""
    if 'selected_meeting' not in st.session_state:
        st.error("No meeting selected")
        return
    
    library = st.session_state.meeting_library
    meeting_data = library.get_meeting(st.session_state.selected_meeting)
    
    if not meeting_data:
        st.error("Meeting not found")
        return
    
    st.header(f"📝 Edit Meeting: {meeting_data.get('title', 'Untitled Meeting')}")
    
    if st.button("← Back to Library"):
        st.session_state.page = None
        st.rerun()
    
    # Edit form
    with st.form("edit_meeting_form"):
        title = st.text_input("Title", value=meeting_data.get('title', ''))
        date = st.date_input("Date", value=datetime.now().date())
        duration = st.number_input("Duration (minutes)", value=meeting_data.get('duration', 0), min_value=0.0)
        
        participants_str = ', '.join(meeting_data.get('participants', []))
        participants = st.text_input("Participants (comma-separated)", value=participants_str)
        
        tags_str = ', '.join(meeting_data.get('tags', []))
        tags = st.text_input("Tags (comma-separated)", value=tags_str)
        
        summary = st.text_area("Summary", value=meeting_data.get('summary', ''), height=200)
        
        if st.form_submit_button("💾 Save Changes"):
            updated_data = {
                'title': title,
                'date': date.isoformat(),
                'duration': duration,
                'participants': [p.strip() for p in participants.split(',') if p.strip()],
                'tags': [t.strip() for t in tags.split(',') if t.strip()],
                'summary': summary
            }
            
            if library.update_meeting(st.session_state.selected_meeting, updated_data):
                st.success("Meeting updated successfully!")
                st.session_state.page = None
                st.rerun()
            else:
                st.error("Failed to update meeting")

if __name__ == "__main__":
    show_meeting_library()
