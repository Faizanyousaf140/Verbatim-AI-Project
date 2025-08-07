"""
Semantic Tasks for Celery
Handles semantic search and AI content mapping operations
"""

from celery import current_app as celery_app
from celery.utils.log import get_task_logger
from datetime import datetime
import json
import numpy as np

logger = get_task_logger(__name__)

@celery_app.task(bind=True, name='semantic_search_query')
def semantic_search_query(self, query, transcript_segments, top_k=10):
    """
    Perform semantic search on transcript segments
    
    Args:
        query: Search query string
        transcript_segments: List of transcript segments
        top_k: Number of top results to return
        
    Returns:
        dict: Search results with similarity scores
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Loading semantic search engine...'
            }
        )
        
        from semantic_search import SemanticSearch
        
        semantic_engine = SemanticSearch()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 40,
                'total': 100,
                'status': 'Generating query embedding...'
            }
        )
        
        # Generate embeddings for segments if not already done
        if isinstance(transcript_segments[0], str):
            segment_embeddings = semantic_engine.generate_embeddings(transcript_segments)
        else:
            # Assume embeddings are already provided
            segment_embeddings = np.array([seg['embedding'] for seg in transcript_segments])
            transcript_segments = [seg['text'] for seg in transcript_segments]
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 60,
                'total': 100,
                'status': 'Performing semantic search...'
            }
        )
        
        # Perform search
        search_results = semantic_engine.semantic_search(
            query, 
            transcript_segments, 
            segment_embeddings, 
            top_k=top_k
        )
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 80,
                'total': 100,
                'status': 'Calculating relevance scores...'
            }
        )
        
        # Enhance results with additional metadata
        enhanced_results = []
        for result in search_results:
            enhanced_result = {
                'text': result['text'],
                'similarity_score': float(result['score']),
                'segment_index': result['index'],
                'relevance_category': 'high' if result['score'] > 0.8 else 'medium' if result['score'] > 0.6 else 'low',
                'word_count': len(result['text'].split()),
                'char_count': len(result['text'])
            }
            enhanced_results.append(enhanced_result)
        
        result = {
            'query': query,
            'results': enhanced_results,
            'total_results': len(enhanced_results),
            'search_timestamp': datetime.utcnow().isoformat(),
            'average_similarity': sum(r['similarity_score'] for r in enhanced_results) / len(enhanced_results) if enhanced_results else 0
        }
        
        logger.info(f"Semantic search completed for query: '{query}'. Found {len(enhanced_results)} results")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'message': 'Semantic search completed successfully'
        }
        
    except Exception as exc:
        logger.error(f"Semantic search failed: {str(exc)}")
        
        if self.request.retries < 2:
            raise self.retry(countdown=30, exc=exc)
        
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Semantic search failed'
        }

@celery_app.task(bind=True, name='map_ai_content_blocks')
def map_ai_content_blocks(self, transcript_data):
    """
    Identify and map AI-related content blocks in transcript
    
    Args:
        transcript_data: Transcript text or structured data
        
    Returns:
        dict: AI content mapping results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 15,
                'total': 100,
                'status': 'Analyzing content for AI patterns...'
            }
        )
        
        # AI-related keywords and patterns
        ai_keywords = [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
            'algorithm', 'model', 'prediction', 'classification', 'regression', 'clustering',
            'natural language processing', 'nlp', 'computer vision', 'automation',
            'chatbot', 'ai assistant', 'gpt', 'transformer', 'embedding', 'vector',
            'training data', 'dataset', 'feature', 'parameter', 'hyperparameter',
            'tensorflow', 'pytorch', 'scikit-learn', 'openai', 'api', 'prompt'
        ]
        
        # Extract text segments
        if isinstance(transcript_data, dict):
            text = transcript_data.get('text', '')
            utterances = transcript_data.get('utterances', [])
        else:
            text = transcript_data
            utterances = []
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 30,
                'total': 100,
                'status': 'Identifying AI content blocks...'
            }
        )
        
        ai_blocks = []
        
        # Analyze utterances for AI content
        if utterances:
            for i, utterance in enumerate(utterances):
                utterance_text = utterance.get('text', '').lower()
                ai_score = 0
                matched_keywords = []
                
                for keyword in ai_keywords:
                    if keyword in utterance_text:
                        ai_score += 1
                        matched_keywords.append(keyword)
                
                if ai_score > 0:
                    ai_blocks.append({
                        'text': utterance.get('text', ''),
                        'speaker': utterance.get('speaker', 'Unknown'),
                        'start_time': utterance.get('start', 0),
                        'end_time': utterance.get('end', 0),
                        'ai_score': ai_score,
                        'matched_keywords': matched_keywords,
                        'relevance': 'high' if ai_score >= 3 else 'medium' if ai_score >= 2 else 'low',
                        'block_index': i
                    })
        else:
            # Analyze sentences if no utterances
            import re
            sentences = re.split(r'[.!?]+', text)
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_lower = sentence.lower()
                ai_score = 0
                matched_keywords = []
                
                for keyword in ai_keywords:
                    if keyword in sentence_lower:
                        ai_score += 1
                        matched_keywords.append(keyword)
                
                if ai_score > 0:
                    ai_blocks.append({
                        'text': sentence,
                        'speaker': 'Unknown',
                        'start_time': 0,
                        'end_time': 0,
                        'ai_score': ai_score,
                        'matched_keywords': matched_keywords,
                        'relevance': 'high' if ai_score >= 3 else 'medium' if ai_score >= 2 else 'low',
                        'block_index': i
                    })
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 60,
                'total': 100,
                'status': 'Categorizing AI content...'
            }
        )
        
        # Categorize AI content by type
        ai_categories = {
            'machine_learning': ['machine learning', 'ml', 'model', 'training', 'algorithm'],
            'natural_language': ['nlp', 'natural language', 'text', 'language model'],
            'computer_vision': ['computer vision', 'image', 'vision', 'opencv'],
            'automation': ['automation', 'bot', 'automated', 'script'],
            'tools_frameworks': ['tensorflow', 'pytorch', 'scikit-learn', 'openai'],
            'concepts': ['artificial intelligence', 'neural network', 'deep learning']
        }
        
        categorized_blocks = {}
        for category, keywords in ai_categories.items():
            categorized_blocks[category] = []
            for block in ai_blocks:
                for keyword in keywords:
                    if keyword in [k.lower() for k in block['matched_keywords']]:
                        categorized_blocks[category].append(block)
                        break
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 80,
                'total': 100,
                'status': 'Generating AI insights...'
            }
        )
        
        # Generate insights
        total_ai_mentions = len(ai_blocks)
        high_relevance_blocks = [b for b in ai_blocks if b['relevance'] == 'high']
        
        # Speaker AI engagement analysis
        speaker_ai_engagement = {}
        for block in ai_blocks:
            speaker = block['speaker']
            if speaker not in speaker_ai_engagement:
                speaker_ai_engagement[speaker] = {
                    'mentions': 0,
                    'total_score': 0,
                    'keywords': set()
                }
            
            speaker_ai_engagement[speaker]['mentions'] += 1
            speaker_ai_engagement[speaker]['total_score'] += block['ai_score']
            speaker_ai_engagement[speaker]['keywords'].update(block['matched_keywords'])
        
        # Convert sets to lists for JSON serialization
        for speaker_data in speaker_ai_engagement.values():
            speaker_data['keywords'] = list(speaker_data['keywords'])
        
        result = {
            'ai_blocks': ai_blocks,
            'categorized_blocks': categorized_blocks,
            'speaker_engagement': speaker_ai_engagement,
            'summary': {
                'total_ai_mentions': total_ai_mentions,
                'high_relevance_mentions': len(high_relevance_blocks),
                'unique_speakers_discussing_ai': len(speaker_ai_engagement),
                'top_keywords': sorted(
                    list(set(kw for block in ai_blocks for kw in block['matched_keywords'])),
                    key=lambda x: sum(1 for block in ai_blocks if x in block['matched_keywords']),
                    reverse=True
                )[:10]
            },
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"AI content mapping completed. Found {total_ai_mentions} AI-related blocks")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'message': 'AI content mapping completed successfully'
        }
        
    except Exception as exc:
        logger.error(f"AI content mapping failed: {str(exc)}")
        
        if self.request.retries < 2:
            raise self.retry(countdown=30, exc=exc)
        
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'AI content mapping failed'
        }

@celery_app.task(bind=True, name='similarity_analysis')
def similarity_analysis(self, transcript_segments):
    """
    Perform similarity analysis between transcript segments
    
    Args:
        transcript_segments: List of transcript segments
        
    Returns:
        dict: Similarity analysis results
    """
    try:
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Generating embeddings for similarity analysis...'
            }
        )
        
        from semantic_search import SemanticSearch
        
        semantic_engine = SemanticSearch()
        
        # Generate embeddings
        embeddings = semantic_engine.generate_embeddings(transcript_segments)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 50,
                'total': 100,
                'status': 'Calculating similarity matrix...'
            }
        )
        
        # Get similarity heatmap data
        heatmap_data = semantic_engine.get_similarity_heatmap_data(transcript_segments, embeddings)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 70,
                'total': 100,
                'status': 'Finding similar segment pairs...'
            }
        )
        
        # Find most similar pairs
        similarity_matrix = heatmap_data['similarity_matrix']
        similar_pairs = []
        
        for i in range(len(transcript_segments)):
            for j in range(i + 1, len(transcript_segments)):
                similarity = similarity_matrix[i][j]
                if similarity > 0.7:  # High similarity threshold
                    similar_pairs.append({
                        'segment_1': {
                            'index': i,
                            'text': transcript_segments[i]
                        },
                        'segment_2': {
                            'index': j,
                            'text': transcript_segments[j]
                        },
                        'similarity_score': float(similarity)
                    })
        
        # Sort by similarity score
        similar_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 90,
                'total': 100,
                'status': 'Generating similarity insights...'
            }
        )
        
        # Calculate statistics
        all_similarities = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix[i])):
                all_similarities.append(similarity_matrix[i][j])
        
        avg_similarity = sum(all_similarities) / len(all_similarities) if all_similarities else 0
        max_similarity = max(all_similarities) if all_similarities else 0
        min_similarity = min(all_similarities) if all_similarities else 0
        
        result = {
            'heatmap_data': heatmap_data,
            'similar_pairs': similar_pairs[:20],  # Top 20 most similar pairs
            'statistics': {
                'average_similarity': float(avg_similarity),
                'max_similarity': float(max_similarity),
                'min_similarity': float(min_similarity),
                'total_segments': len(transcript_segments),
                'high_similarity_pairs': len([p for p in similar_pairs if p['similarity_score'] > 0.8])
            },
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Similarity analysis completed for {len(transcript_segments)} segments")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'message': 'Similarity analysis completed successfully'
        }
        
    except Exception as exc:
        logger.error(f"Similarity analysis failed: {str(exc)}")
        
        if self.request.retries < 2:
            raise self.retry(countdown=30, exc=exc)
        
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Similarity analysis failed'
        }

@celery_app.task(name='topic_clustering')
def topic_clustering(transcript_segments, num_clusters=5):
    """
    Perform topic clustering on transcript segments
    
    Args:
        transcript_segments: List of transcript segments
        num_clusters: Number of clusters to create
        
    Returns:
        dict: Topic clustering results
    """
    try:
        from semantic_search import SemanticSearch
        from sklearn.cluster import KMeans
        import numpy as np
        
        semantic_engine = SemanticSearch()
        
        # Generate embeddings
        embeddings = semantic_engine.generate_embeddings(transcript_segments)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(num_clusters, len(transcript_segments)), random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group segments by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'index': i,
                'text': transcript_segments[i]
            })
        
        # Generate cluster summaries
        cluster_summaries = {}
        for cluster_id, segments in clusters.items():
            # Extract key words from cluster
            all_text = ' '.join([seg['text'] for seg in segments])
            words = all_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            
            cluster_summaries[cluster_id] = {
                'segment_count': len(segments),
                'top_keywords': [kw[0] for kw in top_keywords],
                'segments': segments
            }
        
        result = {
            'clusters': cluster_summaries,
            'total_clusters': len(clusters),
            'cluster_labels': cluster_labels.tolist(),
            'clustering_timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Topic clustering completed. Created {len(clusters)} clusters")
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'message': 'Topic clustering completed successfully'
        }
        
    except Exception as exc:
        logger.error(f"Topic clustering failed: {str(exc)}")
        return {
            'status': 'FAILURE',
            'error': str(exc),
            'message': 'Topic clustering failed'
        }
