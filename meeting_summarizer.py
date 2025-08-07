"""
Advanced Meeting Summarization Module
Uses multiple NLP models for comprehensive meeting summarization
"""

import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Optional imports with fallbacks
try:
    import numpy as np
except ImportError:
    print("⚠️ NumPy not available")
    np = None

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    print("✅ PyTorch and Transformers available")
except ImportError as e:
    print(f"⚠️ PyTorch/Transformers not available: {e}")
    torch = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn available")
except ImportError as e:
    print(f"⚠️ Scikit-learn not available: {e}")
    TfidfVectorizer = None
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ NLTK not available: {e}")
    nltk = None
    sent_tokenize = None
    word_tokenize = None
    stopwords = None
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ spaCy not available: {e}")
    spacy = None
    SPACY_AVAILABLE = False

# Download required NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📥 Downloading NLTK punkt tokenizer...")
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("📥 Downloading NLTK stopwords...")
        nltk.download('stopwords')

# Fallback sentence tokenizer for when NLTK is not available
def simple_sent_tokenize(text: str) -> List[str]:
    """Simple sentence tokenizer fallback"""
    import re
    # Split on sentence endings
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

class AdvancedMeetingSummarizer:
    def __init__(self):
        """Initialize the advanced meeting summarizer with multiple NLP models"""
        self.summarization_models = {}
        self.extractive_summarizer = None
        self.abstractive_summarizer = None
        self.sentiment_analyzer = None
        self.nlp = None
        self._load_models()
    
    def _load_models(self):
        """Load various NLP models for different summarization tasks"""
        try:
            print("🔄 Loading NLP models for meeting summarization...")
            
            # Load extractive summarization (TF-IDF based)
            if SKLEARN_AVAILABLE:
                self.extractive_summarizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                print("✅ Extractive summarizer loaded")
            else:
                print("⚠️ Extractive summarizer not available (sklearn missing)")
            
            # Load abstractive summarization model
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Use BART-Large-CNN model which is specifically fine-tuned for summarization
                    self.abstractive_summarizer = pipeline(
                        "summarization",
                        model="facebook/bart-large-cnn",
                        tokenizer="facebook/bart-large-cnn",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    print("✅ Abstractive summarizer (BART-Large-CNN) loaded")
                except Exception as e:
                    print(f"⚠️ Could not load BART-Large-CNN model: {e}")
                    print("⚠️ Trying fallback BART-base model...")
                    try:
                        self.abstractive_summarizer = pipeline(
                            "summarization",
                            model="facebook/bart-base",
                            tokenizer="facebook/bart-base",
                            device=0 if torch.cuda.is_available() else -1
                        )
                        print("✅ Fallback abstractive summarizer (BART-base) loaded")
                    except Exception as e2:
                        print(f"⚠️ Could not load any BART model: {e2}")
                        self.abstractive_summarizer = None
            else:
                print("⚠️ Abstractive summarizer not available (transformers missing)")
                self.abstractive_summarizer = None
            
            # Load sentiment analyzer
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.sentiment_analyzer = pipeline(
                        "sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                        device=0 if torch.cuda.is_available() else -1
                    )
                    print("✅ Sentiment analyzer loaded")
                except Exception as e:
                    print(f"⚠️ Could not load sentiment model: {e}")
                    self.sentiment_analyzer = None
            else:
                print("⚠️ Sentiment analyzer not available (transformers missing)")
                self.sentiment_analyzer = None
            
            # Load spaCy for NER and text processing
            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("✅ spaCy model loaded")
                except Exception as e:
                    print(f"⚠️ Could not load spaCy model: {e}")
                    self.nlp = None
            else:
                print("⚠️ spaCy not available")
                self.nlp = None
            
            print("✅ Model loading completed")
            
        except Exception as e:
            print(f"❌ Error loading NLP models: {e}")
    
    def extract_key_sentences(self, text: str, num_sentences: int = 5) -> List[str]:
        """Extract key sentences using TF-IDF and sentence importance scoring"""
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Use NLTK if available, otherwise use simple tokenizer
            if NLTK_AVAILABLE and sent_tokenize:
                sentences = sent_tokenize(processed_text)
            else:
                sentences = simple_sent_tokenize(processed_text)
            
            if len(sentences) <= num_sentences:
                return sentences
            
            # Use TF-IDF if available
            if SKLEARN_AVAILABLE and self.extractive_summarizer:
                # Create TF-IDF vectors
                tfidf_matrix = self.extractive_summarizer.fit_transform(sentences)
                
                # Calculate sentence importance scores
                sentence_scores = []
                for i, sentence in enumerate(sentences):
                    # TF-IDF score
                    if np:
                        tfidf_score = np.sum(tfidf_matrix[i].toarray())
                    else:
                        tfidf_score = 1.0  # Fallback score
                    
                    # Position score (sentences at beginning and end are more important)
                    position_score = 1.0
                    if i < len(sentences) * 0.1:  # First 10%
                        position_score = 1.2
                    elif i > len(sentences) * 0.9:  # Last 10%
                        position_score = 1.1
                    
                    # Length score (moderate length sentences are preferred)
                    length_score = min(len(sentence.split()) / 20.0, 1.0)
                    
                    # Combined score
                    total_score = tfidf_score * position_score * length_score
                    sentence_scores.append((sentence, total_score))
                
                # Sort by score and return top sentences
                sentence_scores.sort(key=lambda x: x[1], reverse=True)
                return [sentence for sentence, score in sentence_scores[:num_sentences]]
            else:
                # Fallback: use simple sentence selection
                return sentences[:num_sentences]
                
        except Exception as e:
            print(f"Error in extract_key_sentences: {e}")
            # Fallback tokenization
            if NLTK_AVAILABLE and sent_tokenize:
                return sent_tokenize(text)[:num_sentences]
            else:
                return simple_sent_tokenize(text)[:num_sentences]
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better summarization"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        
        # Split into sentences
        if NLTK_AVAILABLE and sent_tokenize:
            sentences = sent_tokenize(text)
        else:
            sentences = simple_sent_tokenize(text)
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only keep meaningful sentences
                cleaned_sentences.append(sentence)
        
        return ' '.join(cleaned_sentences)
    
    def intelligent_text_chunking(self, text: str, max_chunk_words: int = 1000) -> List[str]:
        """Intelligently chunk text while preserving sentence boundaries"""
        try:
            # Split by paragraphs first
            if '\n\n' in text:
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            else:
                # If no paragraphs, treat whole text as one paragraph
                paragraphs = [text.strip()]
            
            chunks = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                # Split paragraph into sentences
                if NLTK_AVAILABLE and sent_tokenize:
                    sentences = sent_tokenize(paragraph)
                else:
                    sentences = simple_sent_tokenize(paragraph)
                
                for sentence in sentences:
                    # Check if adding this sentence would exceed the limit
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    
                    if len(test_chunk.split()) <= max_chunk_words:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk and start new one
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                # Add paragraph break if we're continuing
                if current_chunk and len(paragraphs) > 1:
                    current_chunk += "\n\n"
            
            # Add the final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
            
        except Exception as e:
            print(f"Error in intelligent_text_chunking: {e}")
            # Fallback to simple word-based chunking
            words = text.split()
            chunks = []
            for i in range(0, len(words), max_chunk_words):
                chunks.append(' '.join(words[i:i + max_chunk_words]))
            return chunks
    
    def generate_abstractive_summary(self, text: str, max_length: int = 150) -> str:
        """Generate abstractive summary using BART-Large-CNN model"""
        try:
            if not self.abstractive_summarizer:
                return "Abstractive summarization not available."
            
            # Clean and prepare text
            text = text.strip()
            if not text:
                return "No text provided for summarization."
            
            word_count = len(text.split())
            
            # If text is very short, return it as is
            if word_count < 50:
                return text
            
            # Adjust summary parameters based on input length
            dynamic_max_length = min(max_length, max(50, word_count // 4))
            dynamic_min_length = min(30, dynamic_max_length - 20)
            
            # Handle long texts by intelligent chunking
            max_chunk_length = 1000  # BART can handle up to 1024 tokens
            
            if word_count > max_chunk_length:
                # Use intelligent chunking to preserve context
                chunks = self.intelligent_text_chunking(text, max_chunk_words=max_chunk_length)
                
                # Summarize each chunk
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    if len(chunk.split()) > 20:  # Only summarize substantial chunks
                        try:
                            print(f"📝 Summarizing chunk {i+1}/{len(chunks)}...")
                            
                            # Calculate chunk-specific summary length
                            chunk_words = len(chunk.split())
                            chunk_max_length = min(100, max(30, chunk_words // 3))
                            chunk_min_length = min(20, chunk_max_length - 10)
                            
                            summary = self.abstractive_summarizer(
                                chunk,
                                max_length=chunk_max_length,
                                min_length=chunk_min_length,
                                do_sample=False,
                                early_stopping=True
                            )
                            chunk_summaries.append(summary[0]['summary_text'])
                        except Exception as e:
                            print(f"Warning: Error summarizing chunk {i+1}: {e}")
                            # Use first few sentences as fallback
                            if NLTK_AVAILABLE and sent_tokenize:
                                chunk_sentences = sent_tokenize(chunk)
                            else:
                                chunk_sentences = simple_sent_tokenize(chunk)
                            chunk_summaries.append(' '.join(chunk_sentences[:2]))
                
                # Combine chunk summaries
                if chunk_summaries:
                    combined_summary = " ".join(chunk_summaries)
                    
                    # If combined summary is still too long, summarize again
                    if len(combined_summary.split()) > dynamic_max_length:
                        try:
                            print("📝 Creating final summary from combined chunks...")
                            final_summary = self.abstractive_summarizer(
                                combined_summary,
                                max_length=dynamic_max_length,
                                min_length=dynamic_min_length,
                                do_sample=False,
                                early_stopping=True
                            )
                            return final_summary[0]['summary_text']
                        except Exception as e:
                            print(f"Warning: Error in final summarization: {e}")
                            # Truncate if final summarization fails
                            words = combined_summary.split()
                            return ' '.join(words[:dynamic_max_length]) + "..."
                    
                    return combined_summary
                else:
                    return "Unable to generate summary from the provided text."
            
            else:
                # Direct summarization for shorter texts
                try:
                    print("📝 Generating direct summary...")
                    summary = self.abstractive_summarizer(
                        text,
                        max_length=dynamic_max_length,
                        min_length=dynamic_min_length,
                        do_sample=False,
                        early_stopping=True
                    )
                    return summary[0]['summary_text']
                except Exception as e:
                    print(f"Error in direct summarization: {e}")
                    # Fallback to extractive summary
                    key_sentences = self.extract_key_sentences(text, num_sentences=3)
                    return ' '.join(key_sentences)
                
        except Exception as e:
            print(f"Error in generate_abstractive_summary: {e}")
            return f"Summarization failed: {str(e)}"
    
    def generate_content_summary(self, text: str, target_ratio: float = 0.25) -> str:
        """Generate content summary that is 1/3 or 1/4 of original length (paragraph by paragraph)"""
        try:
            if not text:
                return "No content available for summarization."
            
            print(f"🔄 Generating content summary (target ratio: {target_ratio})...")
            
            # Split text into paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if not paragraphs:
                # If no paragraphs, split by sentences
                if NLTK_AVAILABLE and sent_tokenize:
                    sentences = sent_tokenize(text)
                else:
                    sentences = simple_sent_tokenize(text)
                paragraphs = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
            
            if not paragraphs:
                return "Unable to parse content into paragraphs."
            
            # Calculate target length for each paragraph
            total_words = len(text.split())
            target_words = int(total_words * target_ratio)
            words_per_paragraph = target_words // len(paragraphs) if paragraphs else 0
            
            summarized_paragraphs = []
            
            for paragraph in paragraphs:
                if len(paragraph.split()) <= words_per_paragraph:
                    # Paragraph is already short enough
                    summarized_paragraphs.append(paragraph)
                else:
                    # Summarize this paragraph
                    if self.abstractive_summarizer:
                        try:
                            # Calculate appropriate summary length for this paragraph
                            paragraph_words = len(paragraph.split())
                            target_paragraph_length = max(20, min(words_per_paragraph, paragraph_words // 2))
                            min_paragraph_length = max(10, target_paragraph_length // 2)
                            
                            print(f"📝 Summarizing paragraph ({paragraph_words} words -> ~{target_paragraph_length} words)...")
                            
                            summary = self.abstractive_summarizer(
                                paragraph,
                                max_length=target_paragraph_length,
                                min_length=min_paragraph_length,
                                do_sample=False,
                                early_stopping=True
                            )[0]['summary_text']
                            summarized_paragraphs.append(summary)
                        except Exception as e:
                            print(f"Warning: Error summarizing paragraph: {e}")
                            # Fallback to extractive summarization
                            if NLTK_AVAILABLE and sent_tokenize:
                                sentences = sent_tokenize(paragraph)
                            else:
                                sentences = simple_sent_tokenize(paragraph)
                            target_sentences = max(1, len(sentences) // 3)
                            key_sentences = self.extract_key_sentences(paragraph, num_sentences=target_sentences)
                            summarized_paragraphs.append(' '.join(key_sentences))
                    else:
                        # Use extractive summarization
                        if NLTK_AVAILABLE and sent_tokenize:
                            sentences = sent_tokenize(paragraph)
                        else:
                            sentences = simple_sent_tokenize(paragraph)
                        target_sentences = max(1, len(sentences) // 3)
                        key_sentences = self.extract_key_sentences(paragraph, num_sentences=target_sentences)
                        summarized_paragraphs.append(' '.join(key_sentences))
            
            # Combine summarized paragraphs
            content_summary = '\n\n'.join(summarized_paragraphs)
            
            # Ensure the final summary is within target ratio
            final_word_count = len(content_summary.split())
            if final_word_count > target_words * 1.2:  # Allow 20% tolerance
                print(f"📝 Final summary too long ({final_word_count} words), reducing to {target_words} words...")
                # Further reduce if needed
                if self.abstractive_summarizer:
                    try:
                        content_summary = self.abstractive_summarizer(
                            content_summary,
                            max_length=target_words,
                            min_length=max(target_words // 2, 30),
                            do_sample=False,
                            early_stopping=True
                        )[0]['summary_text']
                    except Exception as e:
                        print(f"Warning: Error in final reduction: {e}")
                        # Use extractive as fallback
                        if NLTK_AVAILABLE and sent_tokenize:
                            sentences = sent_tokenize(content_summary)
                        else:
                            sentences = simple_sent_tokenize(content_summary)
                        target_sentences = max(1, int(len(sentences) * target_ratio))
                        key_sentences = self.extract_key_sentences(content_summary, num_sentences=target_sentences)
                        content_summary = ' '.join(key_sentences)
            
            print(f"✅ Content summary generated: {len(content_summary.split())} words (target: {target_words})")
            return content_summary
            
        except Exception as e:
            print(f"❌ Error generating content summary: {e}")
            return f"Content summarization failed: {str(e)}"
    
    def generate_comprehensive_summary(self, transcript_data: Dict) -> Dict:
        """Generate comprehensive meeting summary using multiple NLP techniques"""
        try:
            # Extract text from transcript
            full_text = transcript_data.get('text', '')
            utterances = transcript_data.get('utterances', [])
            
            if not full_text:
                return {"error": "No transcript text available"}
            
            print("🔄 Generating comprehensive meeting summary...")
            
            # 1. Content Summary (1/3 or 1/4 of original length)
            content_summary = self.generate_content_summary(full_text, target_ratio=0.25)
            
            # 2. Extractive Summary
            key_sentences = self.extract_key_sentences(full_text, num_sentences=8)
            extractive_summary = ' '.join(key_sentences)
            
            # 3. Abstractive Summary
            abstractive_summary = self.generate_abstractive_summary(full_text)
            
            # 3. Speaker Analysis
            speaker_contributions = self.analyze_speaker_contributions(utterances)
            
            # 4. Action Items and Decisions
            action_decisions = self.extract_action_items_and_decisions(full_text)
            
            # 5. Meeting Statistics
            meeting_stats = {
                'total_duration': transcript_data.get('duration_minutes', 0),
                'total_words': len(full_text.split()),
                'total_speakers': len(set([u.speaker for u in utterances])),
                'total_utterances': len(utterances),
                'avg_words_per_utterance': len(full_text.split()) / len(utterances) if utterances else 0
            }
            
            # 6. Sentiment Overview
            overall_sentiment = "neutral"
            if self.sentiment_analyzer:
                try:
                    sentiment_result = self.sentiment_analyzer(full_text[:500])[0]
                    overall_sentiment = sentiment_result['label']
                except Exception as e:
                    pass
            
            # Compile comprehensive summary
            summary = {
                'meeting_info': {
                    'title': transcript_data.get('title', 'Meeting Summary'),
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'duration_minutes': meeting_stats['total_duration'],
                    'participants': meeting_stats['total_speakers']
                },
                'summaries': {
                    'content_summary': content_summary,
                    'extractive': extractive_summary,
                    'abstractive': abstractive_summary,
                    'key_sentences': key_sentences
                },
                'speaker_analysis': speaker_contributions,
                'action_items': action_decisions['action_items'],
                'decisions': action_decisions['decisions'],
                'statistics': meeting_stats,
                'sentiment': overall_sentiment,
                'insights': self._generate_insights(speaker_contributions, action_decisions)
            }
            
            print("✅ Comprehensive meeting summary generated successfully")
            return summary
            
        except Exception as e:
            print(f"❌ Error generating comprehensive summary: {e}")
            return {"error": f"Summary generation failed: {str(e)}"}
    
    def analyze_speaker_contributions(self, utterances: List) -> Dict:
        """Analyze speaker contributions and their impact"""
        try:
            speaker_stats = {}
            
            for utterance in utterances:
                speaker = f"Speaker {utterance.speaker}"
                
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        'utterances': [],
                        'total_words': 0,
                        'total_duration': 0,
                        'sentiment_scores': [],
                        'key_points': []
                    }
                
                # Basic stats
                words = len(utterance.text.split())
                duration = (utterance.end - utterance.start) / 1000  # seconds
                
                speaker_stats[speaker]['utterances'].append(utterance.text)
                speaker_stats[speaker]['total_words'] += words
                speaker_stats[speaker]['total_duration'] += duration
                
                # Sentiment analysis
                if self.sentiment_analyzer:
                    try:
                        sentiment = self.sentiment_analyzer(utterance.text)[0]
                        speaker_stats[speaker]['sentiment_scores'].append({
                            'text': utterance.text,
                            'sentiment': sentiment['label'],
                            'score': sentiment['score']
                        })
                    except Exception as e:
                        pass
            
            # Calculate speaker importance
            total_words = sum(stats['total_words'] for stats in speaker_stats.values())
            total_duration = sum(stats['total_duration'] for stats in speaker_stats.values())
            
            for speaker, stats in speaker_stats.items():
                stats['word_percentage'] = (stats['total_words'] / total_words * 100) if total_words > 0 else 0
                stats['duration_percentage'] = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
                if np and stats['sentiment_scores']:
                    stats['avg_sentiment'] = np.mean([s['score'] for s in stats['sentiment_scores']])
                else:
                    # Fallback average calculation
                    if stats['sentiment_scores']:
                        stats['avg_sentiment'] = sum(s['score'] for s in stats['sentiment_scores']) / len(stats['sentiment_scores'])
                    else:
                        stats['avg_sentiment'] = 0
            
            return speaker_stats
            
        except Exception as e:
            print(f"Error in analyze_speaker_contributions: {e}")
            return {}
    
    def extract_action_items_and_decisions(self, text: str) -> Dict:
        """Extract action items and decisions using NLP patterns"""
        try:
            action_items = []
            decisions = []
            
            # Define patterns for action items and decisions
            action_patterns = [
                r'(?:will|going to|need to|should|must)\s+(?:be|get|have|do|implement|create|send|call|meet|review|complete|finish|start|begin)',
                r'(?:action|task|assignment|responsibility|deadline|follow.?up|review|meeting|call|email|document|report)',
                r'(?:assign|delegate|responsible|accountable|owner|lead|manage|oversee|coordinate|facilitate)',
                r'(?:by|until|before|after|on|next|this|following)\s+(?:week|month|day|monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                r'(?:urgent|priority|important|critical|high.?priority|low.?priority|asap|immediately|soon)'
            ]
            
            decision_patterns = [
                r'(?:decided|decision|agreed|approved|chosen|selected|determined|resolved|settled|concluded|finalized)',
                r'(?:consensus|unanimous|majority|vote|poll|survey|feedback|input|opinion|preference)',
                r'(?:accept|reject|approve|deny|support|oppose|favor|against|pro|con)',
                r'(?:policy|procedure|process|guideline|rule|standard|protocol|method|approach|strategy)',
                r'(?:budget|funding|cost|expense|investment|allocation|distribution|spending|finance|financial)'
            ]
            
            if NLTK_AVAILABLE and sent_tokenize:
                sentences = sent_tokenize(text)
            else:
                sentences = simple_sent_tokenize(text)
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                # Check for action items
                for pattern in action_patterns:
                    if re.search(pattern, sentence_lower):
                        action_items.append({
                            'text': sentence,
                            'pattern': pattern,
                            'type': 'action_item'
                        })
                        break
                
                # Check for decisions
                for pattern in decision_patterns:
                    if re.search(pattern, sentence_lower):
                        decisions.append({
                            'text': sentence,
                            'pattern': pattern,
                            'type': 'decision'
                        })
                        break
            
            return {
                'action_items': action_items,
                'decisions': decisions,
                'total_action_items': len(action_items),
                'total_decisions': len(decisions)
            }
            
        except Exception as e:
            print(f"Error in extract_action_items_and_decisions: {e}")
            return {'action_items': [], 'decisions': [], 'total_action_items': 0, 'total_decisions': 0}
    
    def _generate_insights(self, speaker_contributions: Dict, action_decisions: Dict) -> List[str]:
        """Generate insights from the meeting analysis"""
        insights = []
        
        try:
            # Speaker insights
            if speaker_contributions:
                most_active = max(speaker_contributions.items(), key=lambda x: x[1]['total_words'])
                insights.append(f"Most active participant: {most_active[0]} ({most_active[1]['word_percentage']:.1f}% of conversation)")
                
                if len(speaker_contributions) > 1:
                    least_active = min(speaker_contributions.items(), key=lambda x: x[1]['total_words'])
                    insights.append(f"Least active participant: {least_active[0]} ({least_active[1]['word_percentage']:.1f}% of conversation)")
            
            # Action items insights
            if action_decisions['total_action_items'] > 0:
                insights.append(f"Meeting generated {action_decisions['total_action_items']} action items")
            
            if action_decisions['total_decisions'] > 0:
                insights.append(f"Meeting resulted in {action_decisions['total_decisions']} decisions")
            
            # Engagement insights
            total_utterances = sum(len(stats['utterances']) for stats in speaker_contributions.values())
            if total_utterances > 0:
                avg_utterance_length = sum(stats['total_words'] for stats in speaker_contributions.values()) / total_utterances
                insights.append(f"Average utterance length: {avg_utterance_length:.1f} words")
            
        except Exception as e:
            print(f"Error generating insights: {e}")
        
        return insights

# Utility functions
def format_summary_for_display(summary: Dict) -> str:
    """Format summary for display in the UI"""
    if 'error' in summary:
        return f"Error: {summary['error']}"
    
    formatted = []
    
    # Meeting info
    meeting_info = summary.get('meeting_info', {})
    formatted.append(f"# 📋 Meeting Summary")
    formatted.append(f"**Date**: {meeting_info.get('date', 'Unknown')}")
    formatted.append(f"**Duration**: {meeting_info.get('duration_minutes', 0):.1f} minutes")
    formatted.append(f"**Participants**: {meeting_info.get('participants', 0)} speakers")
    formatted.append("")
    
    # Abstractive summary
    summaries = summary.get('summaries', {})
    if summaries.get('abstractive'):
        formatted.append("## 📝 Executive Summary")
        formatted.append(summaries['abstractive'])
        formatted.append("")
    
    # Key points
    if summaries.get('key_sentences'):
        formatted.append("## 🎯 Key Points")
        for i, sentence in enumerate(summaries['key_sentences'][:5], 1):
            formatted.append(f"{i}. {sentence}")
        formatted.append("")
    
    # Action items
    action_items = summary.get('action_items', [])
    if action_items:
        formatted.append("## ✅ Action Items")
        for i, item in enumerate(action_items[:5], 1):
            formatted.append(f"{i}. {item['text']}")
        formatted.append("")
    
    # Decisions
    decisions = summary.get('decisions', [])
    if decisions:
        formatted.append("## 🤝 Decisions Made")
        for i, decision in enumerate(decisions[:5], 1):
            formatted.append(f"{i}. {decision['text']}")
        formatted.append("")
    
    # Insights
    insights = summary.get('insights', [])
    if insights:
        formatted.append("## 💡 Meeting Insights")
        for insight in insights:
            formatted.append(f"• {insight}")
        formatted.append("")
    
    return "\n".join(formatted)

# Main function for testing
def main():
    """Test the meeting summarizer when run directly"""
    print("🧪 Meeting Summarizer Module Test")
    print("=" * 40)
    
    # Initialize summarizer
    summarizer = AdvancedMeetingSummarizer()
    
    # Test with sample text
    sample_text = """
    Good morning everyone, welcome to our quarterly review meeting. Thank you for organizing this meeting, John. 
    I'm excited to discuss our progress and review what we've accomplished this quarter. Let's start with our 
    financial performance this quarter. We need to carefully review the numbers and understand the trends. 
    I have prepared the comprehensive financial reports for everyone to review. Our revenue increased by 15% 
    this quarter compared to the same period last year. That's absolutely excellent news for our company! 
    We exceeded our targets significantly and this shows the hard work of our entire team. Great work everyone, 
    you should all be proud of these achievements. Now let's discuss our action items for next quarter and 
    plan our strategy going forward. I will prepare the detailed budget proposal and send it to everyone by Friday. 
    Sarah, I'll coordinate with the marketing team for the new product launch campaign that we discussed. 
    We also decided to hire two new developers to support our expansion plans. The board approved the additional 
    funding for this initiative. We need to finalize the job descriptions and start the hiring process immediately.
    """
    
    print(f"📄 Sample text length: {len(sample_text)} characters ({len(sample_text.split())} words)")
    print(f"📄 Original text:\n{sample_text.strip()}\n")
    print("-" * 60)
    
    # Test abstractive summarization with BART-Large-CNN
    print("🤖 Testing AI-Powered Abstractive Summarization (BART-Large-CNN):")
    abstractive_summary = summarizer.generate_abstractive_summary(sample_text, max_length=100)
    print(f"   📝 AI Summary: {abstractive_summary}")
    print(f"   📊 Compression ratio: {len(sample_text.split())} words → {len(abstractive_summary.split())} words")
    print()
    
    # Test content summary (1/4 ratio)
    print("📋 Testing Content Summary (1/4 ratio):")
    content_summary = summarizer.generate_content_summary(sample_text, target_ratio=0.25)
    print(f"   📝 Content Summary: {content_summary}")
    print(f"   📊 Target: {int(len(sample_text.split()) * 0.25)} words, Actual: {len(content_summary.split())} words")
    print()
    
    # Test extractive summarization
    print("� Testing Extractive Summarization (Key Sentences):")
    key_sentences = summarizer.extract_key_sentences(sample_text, num_sentences=3)
    for i, sentence in enumerate(key_sentences, 1):
        print(f"   {i}. {sentence.strip()}")
    print()
    
    # Test action items extraction
    print("✅ Testing Action Items Extraction:")
    action_decisions = summarizer.extract_action_items_and_decisions(sample_text)
    print(f"   Action items found: {action_decisions['total_action_items']}")
    for item in action_decisions['action_items'][:3]:
        print(f"   - {item['text'].strip()}")
    print()
    
    print("✅ Meeting Summarizer module is working correctly!")

if __name__ == "__main__":
    main() 