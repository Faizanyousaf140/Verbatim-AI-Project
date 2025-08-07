"""
Interactive Transcript Editor
Provides drag-to-highlight and annotation capabilities for transcripts
"""

import streamlit as st
import json
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Optional imports with fallbacks
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not available - visualizations will be disabled")
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("⚠️ Pandas not available - some features will be limited")
    pd = None
    PANDAS_AVAILABLE = False

class TranscriptAnnotation:
    """Represents an annotation on transcript text"""
    
    def __init__(self, start_pos: int, end_pos: int, annotation_type: str, text: str, user: str = None):
        self.id = str(uuid.uuid4())
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.annotation_type = annotation_type
        self.text = text
        self.user = user
        self.created_at = datetime.now()
        self.color = self._get_color_for_type(annotation_type)
    
    def _get_color_for_type(self, annotation_type: str) -> str:
        """Get color for annotation type"""
        colors = {
            'highlight': '#ffeb3b',
            'comment': '#2196f3',
            'action_item': '#4caf50',
            'decision': '#ff9800',
            'question': '#9c27b0',
            'important': '#f44336',
            'note': '#607d8b'
        }
        return colors.get(annotation_type, '#cccccc')
    
    def to_dict(self) -> Dict:
        """Convert annotation to dictionary"""
        return {
            'id': self.id,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'annotation_type': self.annotation_type,
            'text': self.text,
            'user': self.user,
            'created_at': self.created_at.isoformat(),
            'color': self.color
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TranscriptAnnotation':
        """Create annotation from dictionary"""
        annotation = cls(
            start_pos=data['start_pos'],
            end_pos=data['end_pos'],
            annotation_type=data['annotation_type'],
            text=data['text'],
            user=data.get('user')
        )
        annotation.id = data['id']
        annotation.created_at = datetime.fromisoformat(data['created_at'])
        annotation.color = data.get('color', annotation.color)
        return annotation

class InteractiveTranscriptEditor:
    """Interactive transcript editor with annotation capabilities"""
    
    def __init__(self):
        self.annotations: List[TranscriptAnnotation] = []
        self.current_user = None
        self.annotation_types = [
            'highlight', 'comment', 'action_item', 'decision', 
            'question', 'important', 'note'
        ]
    
    def add_annotation(self, start_pos: int, end_pos: int, annotation_type: str, text: str, user: str = None) -> TranscriptAnnotation:
        """Add a new annotation"""
        annotation = TranscriptAnnotation(start_pos, end_pos, annotation_type, text, user)
        self.annotations.append(annotation)
        return annotation
    
    def remove_annotation(self, annotation_id: str) -> bool:
        """Remove an annotation by ID"""
        for i, annotation in enumerate(self.annotations):
            if annotation.id == annotation_id:
                del self.annotations[i]
                return True
        return False
    
    def get_annotations_for_range(self, start_pos: int, end_pos: int) -> List[TranscriptAnnotation]:
        """Get annotations that overlap with the given range"""
        overlapping = []
        for annotation in self.annotations:
            if (annotation.start_pos <= end_pos and annotation.end_pos >= start_pos):
                overlapping.append(annotation)
        return overlapping
    
    def get_annotations_by_type(self, annotation_type: str) -> List[TranscriptAnnotation]:
        """Get all annotations of a specific type"""
        return [a for a in self.annotations if a.annotation_type == annotation_type]
    
    def export_annotations(self) -> Dict:
        """Export all annotations"""
        return {
            'annotations': [a.to_dict() for a in self.annotations],
            'exported_at': datetime.now().isoformat(),
            'total_annotations': len(self.annotations)
        }
    
    def import_annotations(self, data: Dict):
        """Import annotations from dictionary"""
        if 'annotations' in data:
            self.annotations = [TranscriptAnnotation.from_dict(a) for a in data['annotations']]

def show_interactive_transcript_editor(transcript_data, current_user: str = None):
    """Display interactive transcript editor"""
    st.title("📝 Interactive Transcript Editor")
    
    # Initialize editor
    if 'transcript_editor' not in st.session_state:
        st.session_state.transcript_editor = InteractiveTranscriptEditor()
    
    editor = st.session_state.transcript_editor
    editor.current_user = current_user
    
    # Get transcript text
    if not transcript_data or not hasattr(transcript_data, 'utterances'):
        st.error("No transcript data available")
        return
    
    # Create transcript text with timestamps
    transcript_text = ""
    utterance_positions = []
    current_pos = 0
    
    for utterance in transcript_data.utterances:
        start_time = f"{utterance.start / 1000 / 60:.2f}:{(utterance.start / 1000) % 60:02.0f}"
        end_time = f"{utterance.end / 1000 / 60:.2f}:{(utterance.end / 1000) % 60:02.0f}"
        
        utterance_text = f"[{start_time}-{end_time}] Speaker {utterance.speaker}: {utterance.text}\n\n"
        
        # Store position information
        utterance_positions.append({
            'speaker': utterance.speaker,
            'start_time': start_time,
            'end_time': end_time,
            'text': utterance.text,
            'start_pos': current_pos,
            'end_pos': current_pos + len(utterance_text)
        })
        
        transcript_text += utterance_text
        current_pos += len(utterance_text)
    
    # Sidebar for annotation controls
    with st.sidebar:
        st.subheader("🎨 Annotation Tools")
        
        # Annotation type selector
        annotation_type = st.selectbox(
            "Annotation Type:",
            editor.annotation_types,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Add annotation manually
        st.subheader("➕ Add Annotation")
        start_pos = st.number_input("Start Position:", min_value=0, max_value=len(transcript_text), value=0)
        end_pos = st.number_input("End Position:", min_value=0, max_value=len(transcript_text), value=100)
        annotation_text = st.text_area("Annotation Text:", placeholder="Enter your annotation...")
        
        if st.button("Add Annotation") and annotation_text:
            editor.add_annotation(start_pos, end_pos, annotation_type, annotation_text, current_user)
            st.success("✅ Annotation added!")
            st.rerun()
        
        # Annotation filters
        st.subheader("🔍 Filter Annotations")
        filter_type = st.selectbox(
            "Filter by type:",
            ['All'] + editor.annotation_types
        )
        
        if filter_type != 'All':
            filtered_annotations = editor.get_annotations_by_type(filter_type)
        else:
            filtered_annotations = editor.annotations
        
        st.write(f"**Total Annotations:** {len(editor.annotations)}")
        st.write(f"**Filtered:** {len(filtered_annotations)}")
    
    # Main transcript display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📄 Transcript with Annotations")
        
        # Display transcript with annotations
        display_text = transcript_text
        annotations_in_range = editor.get_annotations_for_range(0, len(transcript_text))
        
        if annotations_in_range:
            # Sort annotations by start position (reverse to avoid position shifting)
            sorted_annotations = sorted(annotations_in_range, key=lambda x: x.start_pos, reverse=True)
            
            for annotation in sorted_annotations:
                # Create highlighted text
                before = display_text[:annotation.start_pos]
                highlighted = display_text[annotation.start_pos:annotation.end_pos]
                after = display_text[annotation.end_pos:]
                
                # Apply highlighting
                highlighted_text = f'<span style="background-color: {annotation.color}; padding: 2px; border-radius: 3px;" title="{annotation.text}">{highlighted}</span>'
                display_text = before + highlighted_text + after
        
        # Display transcript with HTML formatting
        st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd; max-height: 600px; overflow-y: auto;">
            <pre style="white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.5;">{display_text}</pre>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📋 Annotations List")
        
        if filtered_annotations:
            for annotation in filtered_annotations:
                with st.expander(f"{annotation.annotation_type.title()} - {annotation.text[:50]}..."):
                    st.write(f"**Type:** {annotation.annotation_type}")
                    st.write(f"**Text:** {annotation.text}")
                    st.write(f"**Position:** {annotation.start_pos} - {annotation.end_pos}")
                    st.write(f"**User:** {annotation.user or 'Unknown'}")
                    st.write(f"**Created:** {annotation.created_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Color preview
                    st.markdown(f"""
                    <div style="background-color: {annotation.color}; padding: 10px; border-radius: 5px; margin: 5px 0;">
                        <strong>Color Preview</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{annotation.id}"):
                        editor.remove_annotation(annotation.id)
                        st.success("✅ Annotation deleted!")
                        st.rerun()
        else:
            st.info("No annotations found.")
    
    # Export/Import section
    st.subheader("💾 Export/Import Annotations")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("📤 Export Annotations"):
            export_data = editor.export_annotations()
            st.download_button(
                label="Download Annotations JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"transcript_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col_export2:
        uploaded_file = st.file_uploader("Import Annotations", type=['json'])
        if uploaded_file:
            try:
                import_data = json.load(uploaded_file)
                editor.import_annotations(import_data)
                st.success("✅ Annotations imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error importing annotations: {e}")
    
    # Statistics
    st.subheader("📊 Annotation Statistics")
    
    if editor.annotations:
        # Create statistics
        type_counts = {}
        user_counts = {}
        
        for annotation in editor.annotations:
            type_counts[annotation.annotation_type] = type_counts.get(annotation.annotation_type, 0) + 1
            user_counts[annotation.user or 'Unknown'] = user_counts.get(annotation.user or 'Unknown', 0) + 1
        
        # Display statistics
        col_stats1, col_stats2 = st.columns(2)
        
        with col_stats1:
            st.write("**By Type:**")
            for annotation_type, count in type_counts.items():
                st.write(f"- {annotation_type.replace('_', ' ').title()}: {count}")
        
        with col_stats2:
            st.write("**By User:**")
            for user, count in user_counts.items():
                st.write(f"- {user}: {count}")
        
        # Create visualization
        if type_counts and PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(type_counts.keys()),
                    y=list(type_counts.values()),
                    marker_color=[TranscriptAnnotation(0, 0, t, "").color for t in type_counts.keys()]
                )
            ])
            fig.update_layout(
                title="Annotations by Type",
                xaxis_title="Annotation Type",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif type_counts:
            st.info("📊 Visualization requires plotly (currently unavailable)")
            # Show simple text-based chart
            for annotation_type, count in type_counts.items():
                st.write(f"📊 {annotation_type}: {'█' * count} ({count})")
    else:
        st.info("No annotations to display statistics for.")

def show_annotation_visualization(transcript_data, annotations: List[TranscriptAnnotation]):
    """Show annotation visualization over time"""
    st.subheader("📈 Annotation Timeline")
    
    if not annotations:
        st.info("No annotations to visualize.")
        return
    
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
        st.warning("📊 Timeline visualization requires plotly and pandas (currently unavailable)")
        
        # Show simple text-based timeline
        st.write("**Annotations by Type:**")
        type_counts = {}
        for annotation in annotations:
            type_counts[annotation.annotation_type] = type_counts.get(annotation.annotation_type, 0) + 1
        
        for annotation_type, count in type_counts.items():
            st.write(f"• {annotation_type.replace('_', ' ').title()}: {count}")
        return
    
    # Create timeline data
    timeline_data = []
    
    for annotation in annotations:
        # Find corresponding utterance
        for utterance in transcript_data.utterances:
            utterance_text = f"Speaker {utterance.speaker}: {utterance.text}"
            if annotation.text in utterance_text or utterance_text in annotation.text:
                start_time = utterance.start / 1000 / 60  # Convert to minutes
                timeline_data.append({
                    'time': start_time,
                    'annotation_type': annotation.annotation_type,
                    'text': annotation.text[:50] + "..." if len(annotation.text) > 50 else annotation.text,
                    'user': annotation.user or 'Unknown',
                    'color': annotation.color
                })
                break
    
    if timeline_data:
        # Create timeline visualization
        df = pd.DataFrame(timeline_data)
        
        fig = go.Figure()
        
        for annotation_type in df['annotation_type'].unique():
            type_data = df[df['annotation_type'] == annotation_type]
            color = type_data['color'].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=type_data['time'],
                y=[annotation_type] * len(type_data),
                mode='markers',
                name=annotation_type.replace('_', ' ').title(),
                marker=dict(color=color, size=10),
                text=type_data['text'],
                hovertemplate='<b>%{text}</b><br>Time: %{x:.2f} min<br>Type: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Annotation Timeline",
            xaxis_title="Time (minutes)",
            yaxis_title="Annotation Type",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timeline data available.")

# Utility functions
def highlight_text_with_annotations(text: str, annotations: List[TranscriptAnnotation]) -> str:
    """Highlight text with annotations using HTML"""
    if not annotations:
        return text
    
    # Sort annotations by start position (reverse to avoid position shifting)
    sorted_annotations = sorted(annotations, key=lambda x: x.start_pos, reverse=True)
    
    for annotation in sorted_annotations:
        before = text[:annotation.start_pos]
        highlighted = text[annotation.start_pos:annotation.end_pos]
        after = text[annotation.end_pos:]
        
        # Create highlighted span
        highlighted_html = f'<span style="background-color: {annotation.color}; padding: 2px; border-radius: 3px;" title="{annotation.text}">{highlighted}</span>'
        text = before + highlighted_html + after
    
    return text 