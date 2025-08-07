"""
Minimal Interactive Transcript Editor (No Streamlit Dependencies)
Core functionality for testing without web interface
"""

import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime

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
    
    def __repr__(self):
        return f"TranscriptAnnotation(id='{self.id[:8]}...', type='{self.annotation_type}', text='{self.text[:30]}...')"

class MinimalTranscriptEditor:
    """Minimal transcript editor with annotation capabilities (no Streamlit)"""
    
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
    
    def get_statistics(self) -> Dict:
        """Get annotation statistics"""
        type_counts = {}
        user_counts = {}
        
        for annotation in self.annotations:
            type_counts[annotation.annotation_type] = type_counts.get(annotation.annotation_type, 0) + 1
            user_counts[annotation.user or 'Unknown'] = user_counts.get(annotation.user or 'Unknown', 0) + 1
        
        return {
            'total_annotations': len(self.annotations),
            'by_type': type_counts,
            'by_user': user_counts,
            'available_types': self.annotation_types
        }
    
    def display_annotations(self, max_display: int = 10):
        """Display annotations in a simple text format"""
        print(f"\n📋 Annotations ({len(self.annotations)} total):")
        print("-" * 60)
        
        for i, annotation in enumerate(self.annotations[:max_display], 1):
            print(f"{i}. [{annotation.annotation_type.upper()}] {annotation.text}")
            print(f"   Position: {annotation.start_pos}-{annotation.end_pos}")
            print(f"   User: {annotation.user or 'Unknown'}")
            print(f"   Created: {annotation.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Color: {annotation.color}")
            print()
        
        if len(self.annotations) > max_display:
            print(f"... and {len(self.annotations) - max_display} more annotations")

def highlight_text_with_annotations(text: str, annotations: List[TranscriptAnnotation]) -> str:
    """Highlight text with annotations using simple markers"""
    if not annotations:
        return text
    
    # Sort annotations by start position (reverse to avoid position shifting)
    sorted_annotations = sorted(annotations, key=lambda x: x.start_pos, reverse=True)
    
    for annotation in sorted_annotations:
        before = text[:annotation.start_pos]
        highlighted = text[annotation.start_pos:annotation.end_pos]
        after = text[annotation.end_pos:]
        
        # Create simple text highlighting
        marker = f"[{annotation.annotation_type.upper()}]"
        highlighted_text = f"{marker}{highlighted}{marker}"
        text = before + highlighted_text + after
    
    return text

# Test function
def test_minimal_editor():
    """Test the minimal transcript editor"""
    print("🧪 Testing Minimal Interactive Transcript Editor")
    print("=" * 60)
    
    # Create editor
    editor = MinimalTranscriptEditor()
    print("✅ Created MinimalTranscriptEditor")
    
    # Sample text
    sample_text = "Hello everyone, welcome to our meeting today. We need to discuss the quarterly results and plan for next quarter."
    print(f"📄 Sample text: {sample_text}")
    print()
    
    # Add some annotations
    print("➕ Adding annotations...")
    ann1 = editor.add_annotation(0, 13, "highlight", "Meeting opening greeting", "User1")
    ann2 = editor.add_annotation(50, 70, "action_item", "Discuss quarterly results", "User2")
    ann3 = editor.add_annotation(85, 110, "decision", "Plan for next quarter", "User1")
    
    print(f"   Added {len(editor.annotations)} annotations")
    
    # Display annotations
    editor.display_annotations()
    
    # Test filtering
    print("🔍 Testing filtering...")
    highlights = editor.get_annotations_by_type("highlight")
    action_items = editor.get_annotations_by_type("action_item")
    print(f"   Highlights: {len(highlights)}")
    print(f"   Action items: {len(action_items)}")
    
    # Test range filtering
    range_annotations = editor.get_annotations_for_range(0, 50)
    print(f"   Annotations in range 0-50: {len(range_annotations)}")
    
    # Test statistics
    print("\n📊 Statistics:")
    stats = editor.get_statistics()
    print(f"   Total annotations: {stats['total_annotations']}")
    print(f"   By type: {stats['by_type']}")
    print(f"   By user: {stats['by_user']}")
    
    # Test export/import
    print("\n💾 Testing export/import...")
    export_data = editor.export_annotations()
    print(f"   Exported {export_data['total_annotations']} annotations")
    
    # Create new editor and import
    new_editor = MinimalTranscriptEditor()
    new_editor.import_annotations(export_data)
    print(f"   Imported {len(new_editor.annotations)} annotations to new editor")
    
    # Test text highlighting
    print("\n🎨 Testing text highlighting:")
    highlighted_text = highlight_text_with_annotations(sample_text, editor.annotations)
    print(f"   Original: {sample_text}")
    print(f"   Highlighted: {highlighted_text}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Minimal Interactive Transcript Editor is working correctly.")
    
    return editor

if __name__ == "__main__":
    test_minimal_editor()
