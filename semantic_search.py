"""
Enhanced Semantic Search Module with Advanced Analytics
Uses sentence transformers and FAISS for semantic search capabilities
Includes similarity analysis, embeddings, and detailed analytics
"""

import numpy as np
import pickle
import os
import json
import re
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from config import SENTENCE_TRANSFORMER_MODEL, FAISS_INDEX_TYPE

# Check dependencies with better error handling
DEPENDENCIES_AVAILABLE = False
IMPORT_ERROR_MESSAGE = ""

# Try to import without triggering numpy/pandas issues
try:
    # Import sentence-transformers without sklearn first
    import sentence_transformers
    print("✅ sentence-transformers package found")
    
    # Try importing SentenceTransformer directly
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformer class imported")
    
    # Try faiss
    import faiss
    print("✅ faiss imported successfully")
    
    # Optional dependencies for advanced analytics
    try:
        import sklearn
        from sklearn.cluster import KMeans
        from sklearn.manifold import TSNE
        from sklearn.metrics.pairwise import cosine_similarity
        SKLEARN_AVAILABLE = True
        print("✅ PyTorch and Transformers available")
    except ImportError:
        SKLEARN_AVAILABLE = False
        print("⚠️ scikit-learn not available - some features disabled")
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        PLOTLY_AVAILABLE = True
        print("✅ Scikit-learn available")
    except ImportError:
        PLOTLY_AVAILABLE = False
        print("⚠️ Plotly not available - visualization features disabled")
    
    DEPENDENCIES_AVAILABLE = True
    print("✅ All semantic search dependencies loaded successfully")
    
except ImportError as e:
    IMPORT_ERROR_MESSAGE = str(e)
    print(f"⚠️ Import error: {e}")
    DEPENDENCIES_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    PLOTLY_AVAILABLE = False
    
except ValueError as e:
    # Handle numpy compatibility issues
    IMPORT_ERROR_MESSAGE = f"NumPy compatibility issue: {e}"
    print(f"⚠️ NumPy compatibility issue: {e}")
    print("🔄 Falling back to keyword-based search mode")
    DEPENDENCIES_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    PLOTLY_AVAILABLE = False
    
except Exception as e:
    IMPORT_ERROR_MESSAGE = f"Unexpected error: {e}"
    print(f"⚠️ Unexpected error: {e}")
    print("🔄 Falling back to keyword-based search mode")
    DEPENDENCIES_AVAILABLE = False
    SKLEARN_AVAILABLE = False
    PLOTLY_AVAILABLE = False

class SemanticSearchEngine:
    def __init__(self):
        """Initialize the semantic search engine"""
        self.model = None
        self.index = None
        self.utterances = []
        self.embeddings = None
        self.error_message = None
        self.fallback_mode = False
        
        if DEPENDENCIES_AVAILABLE:
            self._load_model()
        else:
            self.error_message = IMPORT_ERROR_MESSAGE
            self.fallback_mode = True
            print("🔄 Falling back to keyword-based search")
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            print("🔄 Loading sentence transformer model...")
            self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
            print("✅ Sentence transformer model loaded successfully")
            self.error_message = None
        except Exception as e:
            print(f"❌ Failed to load sentence transformer model: {e}")
            self.model = None
            self.error_message = str(e)
            self.fallback_mode = True
    
    def is_available(self):
        """Check if semantic search is available"""
        return self.model is not None or self.fallback_mode
    
    def build_index(self, utterances: List[str]) -> bool:
        """Build semantic search index from utterances"""
        if not self.is_available():
            print("❌ Semantic search not available")
            return False
        
        try:
            print(f"🔄 Building semantic index for {len(utterances)} utterances...")
            
            # Store utterances
            self.utterances = utterances
            
            # Generate embeddings
            self.embeddings = self.model.encode(utterances, convert_to_numpy=True)
            
            # Build FAISS index
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            
            print("✅ Semantic search index built successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error building semantic index: {e}")
            self.error_message = str(e)
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar utterances with enhanced analytics"""
        if self.fallback_mode:
            return self._keyword_search(query, top_k)
        
        if not self.model or not self.index:
            print("❌ Semantic search index not built")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.utterances):
                    results.append({
                        'text': self.utterances[idx],
                        'score': float(score),
                        'rank': i + 1,
                        'index': int(idx),
                        'similarity_type': 'semantic',
                        'embedding_distance': float(1 - score),  # Convert to distance
                        'search_type': 'semantic_transformer'
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Error in semantic search: {e}")
            return []
    
    def get_embedding_analytics(self) -> Dict[str, Any]:
        """Get detailed analytics about embeddings and semantic structure"""
        if not self.embeddings or not SKLEARN_AVAILABLE:
            return {'error': 'Embeddings or scikit-learn not available'}
        
        try:
            analytics = {
                'embedding_stats': {},
                'similarity_matrix': None,
                'clusters': {},
                'outliers': [],
                'semantic_density': 0.0,
                'diversity_score': 0.0
            }
            
            # Basic embedding statistics
            analytics['embedding_stats'] = {
                'total_utterances': len(self.utterances),
                'embedding_dimension': self.embeddings.shape[1],
                'mean_norm': float(np.mean(np.linalg.norm(self.embeddings, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(self.embeddings, axis=1))),
                'sparsity': float(np.mean(self.embeddings == 0))
            }
            
            # Calculate similarity matrix (for smaller datasets)
            if len(self.utterances) <= 100:
                similarity_matrix = cosine_similarity(self.embeddings)
                analytics['similarity_matrix'] = similarity_matrix.tolist()
                
                # Calculate semantic density (average similarity)
                mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
                analytics['semantic_density'] = float(np.mean(similarity_matrix[mask]))
            
            # Diversity score (how spread out the embeddings are)
            if len(self.utterances) > 1:
                distances = []
                for i in range(min(50, len(self.embeddings))):
                    for j in range(i+1, min(50, len(self.embeddings))):
                        dist = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                        distances.append(dist)
                analytics['diversity_score'] = float(np.mean(distances)) if distances else 0.0
            
            # Clustering analysis
            n_clusters = min(5, len(self.utterances) // 2)
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(self.embeddings)
                
                analytics['clusters'] = {
                    'n_clusters': n_clusters,
                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                    'cluster_assignments': cluster_labels.tolist(),
                    'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(n_clusters)],
                    'inertia': float(kmeans.inertia_)
                }
            
            # Outlier detection (utterances far from cluster centers)
            if 'cluster_centers' in analytics['clusters']:
                outlier_threshold = 1.5  # Standard deviations from center
                for i, embedding in enumerate(self.embeddings):
                    min_distance = min([
                        np.linalg.norm(embedding - center) 
                        for center in analytics['clusters']['cluster_centers']
                    ])
                    if min_distance > outlier_threshold:
                        analytics['outliers'].append({
                            'index': i,
                            'text': self.utterances[i][:100] + '...' if len(self.utterances[i]) > 100 else self.utterances[i],
                            'distance_from_center': float(min_distance)
                        })
            
            return analytics
            
        except Exception as e:
            return {'error': f'Error calculating analytics: {str(e)}'}
    
    def get_similarity_heatmap_data(self, max_utterances: int = 50) -> Dict[str, Any]:
        """Generate data for similarity heatmap visualization"""
        if not self.embeddings or not SKLEARN_AVAILABLE:
            return {'error': 'Embeddings or scikit-learn not available'}
        
        try:
            # Limit to manageable size
            n_utterances = min(max_utterances, len(self.utterances))
            subset_embeddings = self.embeddings[:n_utterances]
            subset_utterances = self.utterances[:n_utterances]
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(subset_embeddings)
            
            # Prepare labels (truncated utterances)
            labels = [
                f"{i}: {utterance[:30]}..." if len(utterance) > 30 else f"{i}: {utterance}"
                for i, utterance in enumerate(subset_utterances)
            ]
            
            return {
                'similarity_matrix': similarity_matrix.tolist(),
                'labels': labels,
                'n_utterances': n_utterances
            }
            
        except Exception as e:
            return {'error': f'Error generating heatmap data: {str(e)}'}
    
    def get_embedding_visualization_data(self, method: str = 'tsne') -> Dict[str, Any]:
        """Generate 2D visualization data using t-SNE or PCA"""
        if not self.embeddings or not SKLEARN_AVAILABLE:
            return {'error': 'Embeddings or scikit-learn not available'}
        
        try:
            # Reduce dimensions for visualization
            if method.lower() == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.embeddings)-1))
            else:
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
            
            # Apply dimensionality reduction
            embeddings_2d = reducer.fit_transform(self.embeddings)
            
            # Prepare data
            return {
                'x': embeddings_2d[:, 0].tolist(),
                'y': embeddings_2d[:, 1].tolist(),
                'texts': [text[:50] + '...' if len(text) > 50 else text for text in self.utterances],
                'method': method.upper(),
                'explained_variance': getattr(reducer, 'explained_variance_ratio_', []).tolist() if hasattr(reducer, 'explained_variance_ratio_') else []
            }
            
        except Exception as e:
            return {'error': f'Error generating visualization data: {str(e)}'}
    
    def find_semantic_themes(self, n_themes: int = 5) -> List[Dict[str, Any]]:
        """Identify semantic themes in the transcript using clustering"""
        if self.embeddings is None or len(self.embeddings) == 0 or not SKLEARN_AVAILABLE:
            return []
        
        try:
            # Perform clustering
            n_clusters = min(n_themes, len(self.utterances) // 2)
            if n_clusters < 2:
                return []
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            themes = []
            for cluster_id in range(n_clusters):
                # Get utterances in this cluster
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_utterances = [self.utterances[i] for i in cluster_indices]
                
                # Calculate cluster statistics
                cluster_embeddings = self.embeddings[cluster_indices]
                center = kmeans.cluster_centers_[cluster_id]
                
                # Find most representative utterance (closest to center)
                distances_to_center = [
                    np.linalg.norm(embedding - center) 
                    for embedding in cluster_embeddings
                ]
                representative_idx = np.argmin(distances_to_center)
                representative_utterance = cluster_utterances[representative_idx]
                
                # Extract key terms from cluster
                all_text = ' '.join(cluster_utterances).lower()
                words = re.findall(r'\b\w+\b', all_text)
                word_freq = {}
                for word in words:
                    if len(word) > 3:  # Filter short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                
                theme = {
                    'theme_id': cluster_id,
                    'size': len(cluster_utterances),
                    'percentage': (len(cluster_utterances) / len(self.utterances)) * 100,
                    'representative_utterance': representative_utterance,
                    'key_terms': [word for word, freq in top_words],
                    'utterances': cluster_utterances[:5],  # Top 5 utterances
                    'coherence_score': float(1.0 / (1.0 + np.mean(distances_to_center)))  # Higher is more coherent
                }
                themes.append(theme)
            
            # Sort by size
            themes.sort(key=lambda x: x['size'], reverse=True)
            return themes
            
        except Exception as e:
            print(f"Error finding semantic themes: {e}")
            return []
    
    def analyze_topic_progression(self) -> Dict[str, Any]:
        """Analyze how topics change over time in the conversation"""
        if not self.embeddings or not SKLEARN_AVAILABLE:
            return {'error': 'Embeddings or scikit-learn not available'}
        
        try:
            # Divide transcript into time segments
            n_segments = min(10, len(self.utterances) // 3)
            if n_segments < 2:
                return {'error': 'Not enough utterances for progression analysis'}
            
            segment_size = len(self.utterances) // n_segments
            segments = []
            
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size if i < n_segments - 1 else len(self.utterances)
                
                # Calculate segment centroid
                segment_embeddings = self.embeddings[start_idx:end_idx]
                centroid = np.mean(segment_embeddings, axis=0)
                
                segments.append({
                    'segment_id': i,
                    'start_utterance': start_idx,
                    'end_utterance': end_idx - 1,
                    'centroid': centroid,
                    'size': end_idx - start_idx
                })
            
            # Calculate similarity between consecutive segments
            progression = []
            for i in range(len(segments) - 1):
                current_centroid = segments[i]['centroid']
                next_centroid = segments[i + 1]['centroid']
                
                # Cosine similarity between centroids
                similarity = np.dot(current_centroid, next_centroid) / (
                    np.linalg.norm(current_centroid) * np.linalg.norm(next_centroid)
                )
                
                progression.append({
                    'from_segment': i,
                    'to_segment': i + 1,
                    'similarity': float(similarity),
                    'topic_shift': float(1 - similarity)  # Higher means more topic shift
                })
            
            return {
                'segments': [{
                    'segment_id': seg['segment_id'],
                    'start_utterance': seg['start_utterance'],
                    'end_utterance': seg['end_utterance'],
                    'size': seg['size']
                } for seg in segments],
                'progression': progression,
                'average_similarity': float(np.mean([p['similarity'] for p in progression])),
                'topic_volatility': float(np.std([p['similarity'] for p in progression]))
            }
            
        except Exception as e:
            return {'error': f'Error analyzing topic progression: {str(e)}'}
    
    def export_analytics_to_csv(self) -> str:
        """Export semantic analytics to CSV format"""
        try:
            analytics = self.get_embedding_analytics()
            themes = self.find_semantic_themes()
            progression = self.analyze_topic_progression()
            
            import csv
            import io
            
            output = io.StringIO()
            
            # Write summary statistics
            output.write("SEMANTIC ANALYTICS REPORT\n")
            output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Embedding statistics
            output.write("EMBEDDING STATISTICS\n")
            for key, value in analytics.get('embedding_stats', {}).items():
                output.write(f"{key},{value}\n")
            output.write("\n")
            
            # Themes
            output.write("SEMANTIC THEMES\n")
            output.write("theme_id,size,percentage,key_terms,representative_utterance\n")
            for theme in themes:
                key_terms = '; '.join(theme['key_terms'])
                rep_utterance = theme['representative_utterance'].replace(',', ';').replace('\n', ' ')
                output.write(f"{theme['theme_id']},{theme['size']},{theme['percentage']:.2f},\"{key_terms}\",\"{rep_utterance}\"\n")
            output.write("\n")
            
            # Topic progression
            if 'progression' in progression:
                output.write("TOPIC PROGRESSION\n")
                output.write("from_segment,to_segment,similarity,topic_shift\n")
                for prog in progression['progression']:
                    output.write(f"{prog['from_segment']},{prog['to_segment']},{prog['similarity']:.4f},{prog['topic_shift']:.4f}\n")
            
            return output.getvalue()
            
        except Exception as e:
            return f"Error exporting analytics: {str(e)}"
        
        if not self.is_available() or self.index is None:
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.utterances):
                    results.append({
                        'text': self.utterances[idx],
                        'score': float(score),
                        'rank': i + 1,
                        'index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Error during semantic search: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Fallback keyword-based search when semantic search is not available"""
        if not self.utterances:
            return []
        
        query_words = set(query.lower().split())
        results = []
        
        for idx, utterance in enumerate(self.utterances):
            utterance_words = set(utterance.lower().split())
            
            # Calculate word overlap score
            common_words = query_words.intersection(utterance_words)
            if common_words:
                score = len(common_words) / len(query_words.union(utterance_words))
                
                # Boost score for exact phrase matches
                if query.lower() in utterance.lower():
                    score += 0.5
                
                results.append({
                    'text': utterance,
                    'score': float(score),
                    'index': idx,
                    'search_type': 'keyword'
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Add rank
        for i, result in enumerate(results[:top_k]):
            result['rank'] = i + 1
        
        return results[:top_k]
    
    def get_similar_utterances(self, target_utterance: str, top_k: int = 3) -> List[Dict]:
        """Find utterances similar to a target utterance"""
        return self.search(target_utterance, top_k)
    
    def cluster_utterances(self, n_clusters: int = 5) -> Dict:
        """Cluster utterances into topics"""
        if self.fallback_mode:
            return self._simple_topic_clustering(n_clusters)
            
        if not self.is_available() or self.embeddings is None:
            return {}
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.embeddings)
            
            # Organize results
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'text': self.utterances[i],
                    'index': i
                })
            
            # Add cluster summaries
            cluster_summaries = {}
            for cluster_id, utterances in clusters.items():
                # Simple summary: most common words
                all_text = ' '.join([u['text'] for u in utterances])
                words = all_text.lower().split()
                from collections import Counter
                common_words = Counter(words).most_common(5)
                cluster_summaries[cluster_id] = {
                    'utterances': utterances,
                    'count': len(utterances),
                    'keywords': [word for word, count in common_words if len(word) > 3]
                }
            
            return cluster_summaries
            
        except Exception as e:
            print(f"❌ Error clustering utterances: {e}")
            return {}
    
    def _simple_topic_clustering(self, n_clusters: int = 5) -> Dict:
        """Simple keyword-based topic clustering when ML libraries are not available"""
        if not self.utterances:
            return {}
        
        # Extract common words from all utterances
        from collections import Counter
        
        # Combine all text and extract keywords
        all_text = ' '.join(self.utterances).lower()
        words = [word for word in all_text.split() if len(word) > 3 and word.isalpha()]
        word_counts = Counter(words)
        
        # Get top keywords
        top_keywords = [word for word, count in word_counts.most_common(20)]
        
        # Simple clustering based on keyword presence
        clusters = {}
        cluster_keywords = {}
        
        # Distribute keywords across clusters
        keywords_per_cluster = max(1, len(top_keywords) // n_clusters)
        
        for i in range(n_clusters):
            start_idx = i * keywords_per_cluster
            end_idx = start_idx + keywords_per_cluster
            cluster_keywords[i] = top_keywords[start_idx:end_idx]
            clusters[i] = []
        
        # Assign utterances to clusters based on keyword matches
        for idx, utterance in enumerate(self.utterances):
            utterance_lower = utterance.lower()
            best_cluster = 0
            best_score = 0
            
            for cluster_id, keywords in cluster_keywords.items():
                score = sum(1 for keyword in keywords if keyword in utterance_lower)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_id
            
            clusters[best_cluster].append({
                'text': utterance,
                'index': idx
            })
        
        # Format results
        cluster_summaries = {}
        for cluster_id, utterances in clusters.items():
            if utterances:  # Only include non-empty clusters
                cluster_summaries[cluster_id] = {
                    'utterances': utterances,
                    'count': len(utterances),
                    'keywords': cluster_keywords[cluster_id][:5]  # Top 5 keywords
                }
        
        return cluster_summaries
    
    def save_index(self, filepath: str) -> bool:
        """Save the search index to file"""
        if not self.is_available() or self.index is None:
            return False
        
        try:
            data = {
                'utterances': self.utterances,
                'embeddings': self.embeddings.tolist() if self.embeddings is not None else None
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Save FAISS index separately
            faiss_path = filepath.replace('.pkl', '.faiss')
            faiss.write_index(self.index, faiss_path)
            
            print(f"✅ Semantic search index saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving index: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """Load a search index from file"""
        if not self.is_available():
            return False
        
        try:
            # Load data
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.utterances = data['utterances']
            self.embeddings = np.array(data['embeddings']) if data['embeddings'] else None
            
            # Load FAISS index
            faiss_path = filepath.replace('.pkl', '.faiss')
            if os.path.exists(faiss_path):
                self.index = faiss.read_index(faiss_path)
            
            print(f"✅ Semantic search index loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False

def show_semantic_search():
    """Streamlit UI for semantic search"""
    import streamlit as st
    
    st.header("🔍 Semantic Search")
    
    # Check if transcript is available
    if not st.session_state.verbatim_ai.transcript_data:
        st.info("No transcript available. Please transcribe a meeting first.")
        return
    
    transcript = st.session_state.verbatim_ai.transcript_data
    
    # Initialize semantic search engine
    if not hasattr(st.session_state, 'semantic_engine'):
        st.session_state.semantic_engine = SemanticSearchEngine()
    
    engine = st.session_state.semantic_engine
    
    # Check if semantic search is available
    if not engine.is_available():
        st.warning("🔍 Semantic search is not properly configured.")
        
        if engine.error_message:
            st.error(f"Error: {engine.error_message}")
        
        st.info("""
        **To enable semantic search, ensure the following packages are installed:**
        ```bash
        pip install sentence-transformers faiss-cpu
        ```
        """)
        
        if st.button("🔄 Try to Reinitialize"):
            st.session_state.semantic_engine = SemanticSearchEngine()
            st.rerun()
        return
    
    # Build index if needed
    if not hasattr(st.session_state, 'semantic_index_built') or not st.session_state.semantic_index_built:
        with st.spinner("Building semantic search index..."):
            utterances = [utterance.text for utterance in transcript.utterances]
            success = engine.build_index(utterances)
            if success:
                st.session_state.semantic_index_built = True
                st.success("✅ Semantic search index built successfully!")
            else:
                st.error("❌ Failed to build semantic search index.")
                return
    
    # Search interface
    st.subheader("🔍 Search Transcript")
    
    search_query = st.text_input("Enter your search query:", placeholder="e.g., project timeline, budget discussion")
    
    if search_query:
        with st.spinner("Searching..."):
            results = engine.search(search_query, top_k=10)
        
        if results:
            st.subheader(f"📋 Search Results ({len(results)} found)")
            
            for i, result in enumerate(results):
                with st.expander(f"Result {i+1} (Score: {result['score']:.3f})"):
                    st.write(result['text'])
                    st.caption(f"Position in transcript: {result['index'] + 1}")
        else:
            st.info("No results found for your query.")
    
    # Topic clustering
    st.subheader("🎯 Topic Clusters")
    
    if st.button("🔄 Generate Topic Clusters"):
        with st.spinner("Clustering topics..."):
            clusters = engine.cluster_utterances(n_clusters=5)
        
        if clusters:
            for cluster_id, cluster_data in clusters.items():
                with st.expander(f"📂 Topic {cluster_id + 1} ({cluster_data['count']} utterances)"):
                    if cluster_data['keywords']:
                        st.write(f"**Keywords:** {', '.join(cluster_data['keywords'])}")
                    
                    for utterance in cluster_data['utterances'][:3]:  # Show top 3
                        st.write(f"• {utterance['text']}")
                    
                    if len(cluster_data['utterances']) > 3:
                        st.caption(f"... and {len(cluster_data['utterances']) - 3} more")
        else:
            st.error("Failed to generate topic clusters.")

if __name__ == "__main__":
    # Test the semantic search engine
    engine = SemanticSearchEngine()
    print(f"Semantic search available: {engine.is_available()}")
    
    if engine.is_available():
        # Test with sample data
        test_utterances = [
            "Let's discuss the project timeline and milestones",
            "The budget for this quarter needs to be reviewed",
            "We need to schedule the next meeting for Friday",
            "The client feedback was very positive",
            "Can we increase the budget allocation for marketing?",
            "The project deadline is approaching fast",
            "Meeting with stakeholders scheduled for next week"
        ]
        
        print("\n🧪 Testing semantic search with sample data...")
        
        # Build index with test data
        engine.build_index(test_utterances)
        
        # Test search
        test_query = "budget discussion"
        print(f"\n🔍 Searching for: '{test_query}'")
        results = engine.search(test_query, top_k=3)
        
        if results:
            print("✅ Search results:")
            for result in results:
                search_type = result.get('search_type', 'semantic')
                print(f"  {result['rank']}. {result['text']} (score: {result['score']:.3f}, type: {search_type})")
        else:
            print("❌ No results found")
        
        # Test clustering
        print(f"\n📊 Testing topic clustering...")
        clusters = engine.cluster_utterances(n_clusters=3)
        
        if clusters:
            print("✅ Topic clusters:")
            for cluster_id, data in clusters.items():
                print(f"  Topic {cluster_id + 1}: {data['count']} items, keywords: {data['keywords']}")
        else:
            print("❌ No clusters generated")
    
    print("\n✅ Semantic search test completed!")
