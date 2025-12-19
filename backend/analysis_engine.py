import re
import io
import os
import sys
import base64
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import AnalysisRequest, AnalysisResults, InfluenceMetrics, SentimentMetrics, ParticipationMetrics, TopicCredibilityMetrics
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

for pkg in ['opinion_lexicon']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)


# Download NLTK dependencies
for pkg in ['punkt', 'stopwords', 'vader_lexicon']:
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg ==
                       'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)


class HierarchicalAttention(nn.Module):
    """Hierarchical attention mechanism for dialogue comprehension"""

    def __init__(self, hidden_dim: int = 384, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Turn-level attention
        self.turn_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Word-level attention
        self.word_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Context fusion
        self.context_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings: torch.Tensor, turn_masks: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = embeddings.shape

        # Turn-level attention
        turn_output, turn_weights = self.turn_attention(
            embeddings, embeddings, embeddings, key_padding_mask=turn_masks
        )

        # Word-level attention with turn context
        word_output, word_weights = self.word_attention(
            embeddings, turn_output, turn_output, key_padding_mask=turn_masks
        )

        # Fuse contexts
        fused = torch.cat([turn_output, word_output], dim=-1)
        output = self.context_fusion(fused)
        output = self.layer_norm(output + embeddings)  # Residual connection
        output = self.dropout(output)

        return output, (turn_weights, word_weights)


class GraphAttentionInfluence(nn.Module):
    """Graph Attention Network for modeling dialogue influence"""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # GAT layers
        self.gat1 = GATConv(input_dim, hidden_dim,
                            heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * num_heads,
                            hidden_dim, heads=1, dropout=0.2)

        # Influence prediction heads
        self.influence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Global pooling for graph-level features
        if batch is not None:
            graph_features = global_mean_pool(x, batch)
        else:
            graph_features = torch.mean(x, dim=0, keepdim=True)

        # Predict influence scores
        influence_scores = self.influence_predictor(x)

        return {
            'node_embeddings': x,
            'graph_embeddings': graph_features,
            'influence_scores': influence_scores
        }


class DynamicTopicModel:
    """Dynamic topic modeling with contextual awareness"""

    def __init__(self, n_topics: int = 5, alpha: float = 0.1):
        self.n_topics = n_topics
        self.alpha = alpha
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            doc_topic_prior=alpha,
            random_state=42,
            max_iter=100
        )
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.topic_evolution = []

    def fit_transform(self, texts: List[str], timestamps: List[int] = None) -> np.ndarray:
        """Fit dynamic topic model and return topic distributions"""
        # Vectorize texts
        text_vectors = self.vectorizer.fit_transform(texts)

        # Fit LDA
        topic_distributions = self.lda.fit_transform(text_vectors)

        # Track topic evolution if timestamps provided
        if timestamps:
            self._track_topic_evolution(topic_distributions, timestamps)

        return topic_distributions

    def _track_topic_evolution(self, distributions: np.ndarray, timestamps: List[int]):
        """Track how topics evolve over time"""
        sorted_indices = np.argsort(timestamps)
        sorted_distributions = distributions[sorted_indices]

        # Calculate topic shift magnitudes
        for i in range(1, len(sorted_distributions)):
            shift = np.linalg.norm(
                sorted_distributions[i] - sorted_distributions[i-1])
            self.topic_evolution.append({
                'timestamp': timestamps[sorted_indices[i]],
                'shift_magnitude': shift,
                'dominant_topics': np.argsort(sorted_distributions[i])[-3:][::-1]
            })

    def get_topic_keywords(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Get top keywords for each topic"""
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}

        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            topics[topic_idx] = [feature_names[i] for i in top_words_idx]

        return topics


class EnhancedDialogueAnalysisEngine:
    """Enhanced dialogue analysis with state-of-the-art models"""

    def __init__(self):
        self._initialize_models()
        self.stop_words = set(stopwords.words('english'))
        from nltk.corpus import opinion_lexicon
        self.positive_words = set(opinion_lexicon.positive())
        self.negative_words = set(opinion_lexicon.negative())

    def _initialize_models(self):
        """Initialize all models with proper error handling"""
        try:
            # RoBERTa for sentiment analysis
            self.sentiment_tokenizer = RobertaTokenizer.from_pretrained(
                'roberta-base')
            self.sentiment_model = RobertaForSequenceClassification.from_pretrained(
                'cardiffnlp/twitter-roberta-base-sentiment-latest'
            )
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
        except Exception as e:
            print(f"Warning: Could not load RoBERTa sentiment model: {e}")
            self.sentiment_model = None
            self.sentiment_pipeline = None

        # Sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention()

        # Graph attention for influence
        self.graph_attention = GraphAttentionInfluence()

        # Dynamic topic model
        self.topic_model = DynamicTopicModel()

        # Device setup
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        if self.sentiment_model:
            self.sentiment_model.to(self.device)
        self.hierarchical_attention.to(self.device)
        self.graph_attention.to(self.device)

    def analyze_dialogue(self, request: AnalysisRequest) -> AnalysisResults:
        """Enhanced dialogue analysis with all improvements"""
        dialogue = [{"agent": turn.agent, "text": turn.text}
                    for turn in request.dialogues]
        topic = request.topic

        # Extract basic information
        utterances = [turn["text"] for turn in dialogue]
        agents = [turn["agent"] for turn in dialogue]

        # Generate embeddings
        embeddings = self.sentence_model.encode(
            utterances, normalize_embeddings=True)
        topic_embedding = self.sentence_model.encode(
            [topic], normalize_embeddings=True)[0]

        # Enhanced analysis components
        sentiment_metrics = self._analyze_enhanced_sentiment(
            dialogue, topic, topic_embedding)
        influence_metrics = self._analyze_enhanced_influence(
            dialogue, embeddings, sentiment_metrics.sentiment_labels)
        participation_metrics = self._analyze_enhanced_participation(
            dialogue, sentiment_metrics.topic_relevance_scores
        )
        topic_credibility_metrics = self._analyze_topic_credibility(
            dialogue, topic, topic_embedding, sentiment_metrics.topic_relevance_scores)
        coherence_scores = self._analyze_enhanced_coherence(
            embeddings, dialogue, topic_embedding)
        topic_analysis = self._analyze_dynamic_topics(
            dialogue, utterances, topic, topic_embedding)

        # Generate visualizations
        visualizations = self._generate_enhanced_visualizations(
            dialogue, influence_metrics, sentiment_metrics,
            participation_metrics, coherence_scores, embeddings, topic_analysis, topic_credibility_metrics
        )

        # Generate enhanced summary
        summary_text = self._generate_enhanced_summary(
            influence_metrics, sentiment_metrics, participation_metrics, topic_analysis, topic_credibility_metrics
        )

        return AnalysisResults(
            influence_metrics=influence_metrics,
            sentiment_metrics=sentiment_metrics,
            participation_metrics=participation_metrics,
            topic_credibility_metrics=topic_credibility_metrics,
            coherence_scores=coherence_scores,
            topic_shifts=topic_analysis.get('shifts', []),
            visualizations=visualizations,
            summary_text=summary_text
        )

    def _plot_agent_sentiment_table(self, dialogue, sentiment_labels, sentiment_word_stats):
        import matplotlib.pyplot as plt
        table_data = []
        for idx, turn in enumerate(dialogue):
            agent = turn['agent']
            turn_num = idx + 1
            stance = sentiment_labels[idx]
            pos_cnt = sentiment_word_stats[idx].get('pos', 0)
            neg_cnt = sentiment_word_stats[idx].get('neg', 0)
            neu_cnt = sentiment_word_stats[idx].get('neu', 0)
            table_data.append([
                agent, turn_num, stance, pos_cnt, neg_cnt, neu_cnt
            ])
        col_labels = ['Agent', 'Turn', 'Stance',
                      'Positive Words', 'Negative Words', 'Neutral Words']
        fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.4 + 2))
        ax.axis('off')

        mpl_table = ax.table(cellText=table_data,
                             colLabels=col_labels,
                             loc='center',
                             cellLoc='center')
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(12)
        mpl_table.auto_set_column_width(col=list(range(len(col_labels))))
        plt.title("Agent-wise Sentiment Per Turn", fontsize=14, pad=16)
        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_agent_sentiment_per_turn(self, dialogue, sentiment_labels):
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import numpy as np

        # Organize sentiments and turn indices by agent
        agent_data = defaultdict(lambda: {'turns': [], 'sentiments': []})
        for idx, turn in enumerate(dialogue):
            agent = turn['agent']
            agent_data[agent]['turns'].append(idx + 1)
            agent_data[agent]['sentiments'].append(sentiment_labels[idx])

        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}

        fig, ax = plt.subplots(figsize=(9, 4.5))
        for agent, data in agent_data.items():
            x = data['turns']
            y = [sentiment_map[s] for s in data['sentiments']]
            # Plot points and connect them for each agent
            ax.plot(x, y, marker='o', label=agent,
                    alpha=0.8, linestyle='-', linewidth=2)
            for xi, yi, s in zip(x, y, data['sentiments']):
                ax.scatter(xi, yi, color=colors[s],
                           s=60, edgecolor='k', zorder=10)

        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(['Negative', 'Neutral', 'Positive'])
        ax.set_xlabel('Turn Number')
        ax.set_ylabel('Sentiment')
        ax.set_title('Sentiment per Turn by Agent')
        ax.legend(title='Agent')
        ax.grid(alpha=0.5, linestyle='--')
        plt.tight_layout()

        return self._plot_to_base64(fig)

    def _analyze_enhanced_influence(self, dialogue: List[Dict], embeddings: np.ndarray, sentiment_labels: List[str]) -> InfluenceMetrics:
        """Enhanced influence analysis using Graph Attention Networks"""
        # Traditional influence tracking
        introduced_by = {}
        agent_influence_count = {turn["agent"]: 0 for turn in dialogue}
        influence_pairs = {}

        # Build graph structure
        edge_indices = []
        edge_weights = []
        node_features = torch.FloatTensor(embeddings).to(self.device)
        # Create edges based on semantic similarity and temporal order
        for i in range(len(dialogue)):
            for j in range(i + 1, min(i + 5, len(dialogue))):  # Window of 5 turns
                similarity = cosine_similarity(
                    [embeddings[i]], [embeddings[j]])[0][0]
                if similarity > 0.3:  # Threshold for connection
                    # Bidirectional edges
                    edge_indices.extend([[i, j], [j, i]])
                    edge_weights.extend([similarity, similarity])

        if edge_indices:
            edge_index = torch.LongTensor(edge_indices).t().to(self.device)

            # Apply Graph Attention Network
            with torch.no_grad():
                gat_output = self.graph_attention(node_features, edge_index)
                influence_scores = gat_output['influence_scores'].cpu(
                ).numpy().flatten()

            # Update influence counts based on GAT scores
            for i, score in enumerate(influence_scores):
                agent = dialogue[i]["agent"]
                agent_influence_count[agent] += float(score)

        # Traditional word-based influence (enhanced)
        for idx, turn in enumerate(dialogue):
            agent = turn["agent"]
            text = turn["text"]
            words = word_tokenize(text.lower())
            words = [w for w in words if w.isalpha() and w not in self.stop_words]
            unique_words = set(words)

            for w in unique_words:
                if w not in introduced_by:
                    introduced_by[w] = (agent, idx)
                else:
                    origin_agent, origin_idx = introduced_by[w]
                    if origin_agent != agent:

                        # only count if sentiment is positive or neutral
                        if sentiment_labels[idx] == 'negative':
                            continue  # Not a positive influence
                        influence_pairs[(origin_agent, agent)] = influence_pairs.get(
                            (origin_agent, agent), 0
                        ) + 1

        # Convert influence pairs to string format
        influence_pairs_str = {
            f"{src}->{dest}": weight for (src, dest), weight in influence_pairs.items()
        }

        agent_influence_count = {k: int(round(v))
                                 for k, v in agent_influence_count.items()}

        return InfluenceMetrics(
            agent_influence_count=agent_influence_count,
            influence_pairs=influence_pairs_str
        )

    def _analyze_enhanced_sentiment(self, dialogue: List[Dict], topic: str, topic_embedding: np.ndarray) -> SentimentMetrics:
        """Enhanced sentiment analysis using RoBERTa and ensemble methods"""
        sentiment_scores = []
        sentiment_labels = []
        topic_relevance_scores = []

        # [ADDED BEGIN] --- Prepare sentiment word stat
        sentiment_word_stats = []
        try:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import opinion_lexicon
            self.positive_words = set(opinion_lexicon.positive())
            self.negative_words = set(opinion_lexicon.negative())
        except Exception as e:
            # fallback: use empty sets
            self.positive_words, self.negative_words = set(), set()

        for turn in dialogue:
            text = turn["text"]
            # [ADDED BEGIN] --- Sentiment word stats per turn
            words = [w for w in word_tokenize(text.lower()) if w.isalpha()]
            pos_cnt = sum(1 for w in words if w in self.positive_words)
            neg_cnt = sum(1 for w in words if w in self.negative_words)
            neu_cnt = len(words) - pos_cnt - neg_cnt
            sentiment_word_stats.append(
                {'pos': pos_cnt, 'neg': neg_cnt, 'neu': neu_cnt})

            # RoBERTa-based sentiment analysis
            if self.sentiment_pipeline:
                try:
                    roberta_result = self.sentiment_pipeline(text)[0]
                    # Map RoBERTa labels to our format
                    label_mapping = {
                        'LABEL_0': 'negative',
                        'LABEL_1': 'neutral',
                        'LABEL_2': 'positive',
                        'negative': 'negative',
                        'neutral': 'neutral',
                        'positive': 'positive'
                    }
                    sentiment_label = label_mapping.get(
                        roberta_result['label'].lower(), 'neutral')
                    sentiment_score = roberta_result['score']

                    # Convert to compound score format
                    if sentiment_label == 'positive':
                        compound_score = sentiment_score
                    elif sentiment_label == 'negative':
                        compound_score = -sentiment_score
                    else:
                        compound_score = 0.0

                    sentiment_scores.append({
                        'compound': compound_score,
                        'pos': sentiment_score if sentiment_label == 'positive' else 0,
                        'neu': sentiment_score if sentiment_label == 'neutral' else 0,
                        'neg': sentiment_score if sentiment_label == 'negative' else 0
                    })
                    sentiment_labels.append(sentiment_label)

                except Exception as e:
                    print(f"RoBERTa sentiment analysis failed: {e}")
                    # Fallback to neutral
                    sentiment_scores.append(
                        {'compound': 0.0, 'pos': 0, 'neu': 1, 'neg': 0})
                    sentiment_labels.append('neutral')
            else:
                # Fallback to neutral if model not available
                sentiment_scores.append(
                    {'compound': 0.0, 'pos': 0, 'neu': 1, 'neg': 0})
                sentiment_labels.append('neutral')

            # Enhanced topic relevance using contextual embeddings
            embedding = self.sentence_model.encode([text])[0]
            similarity = cosine_similarity(
                [embedding], [topic_embedding])[0][0]
            topic_relevance_scores.append(float(similarity))

        # Calculate agent-level metrics
        agent_sentiment = defaultdict(float)
        agent_sent_count = defaultdict(int)
        agent_topic_relevance_sum = defaultdict(float)

        for turn, scores, relevance in zip(dialogue, sentiment_scores, topic_relevance_scores):
            agent = turn["agent"]
            agent_sentiment[agent] += scores['compound']
            agent_sent_count[agent] += 1
            agent_topic_relevance_sum[agent] += relevance

        avg_sentiment_per_agent = {
            a: agent_sentiment[a] / agent_sent_count[a] for a in agent_sentiment
        }
        avg_topic_relevance_per_agent = {
            a: agent_topic_relevance_sum[a] / agent_sent_count[a] for a in agent_sentiment
        }

        # [ADDED] --- Return sentiment_word_stats in SentimentMetrics
        return SentimentMetrics(
            sentiment_labels=sentiment_labels,
            topic_relevance_scores=topic_relevance_scores,
            avg_sentiment_per_agent=avg_sentiment_per_agent,
            avg_topic_relevance_per_agent=avg_topic_relevance_per_agent,
            sentiment_word_stats=sentiment_word_stats
        )

    def _analyze_enhanced_participation(self, dialogue: List[Dict], topic_relevance_scores: List[float]) -> ParticipationMetrics:
        """Enhanced participation analysis with advanced metrics"""
        turn_counts = defaultdict(int)
        word_counts = defaultdict(int)
        agent_topic_sum = defaultdict(float)
        introduced_by = {}
        topic_introduced_count = defaultdict(int)

        # Advanced disagreement detection patterns
        disagree_patterns = [
            r'\b(i disagree|disagree with|don\'t agree|not sure|wrong|incorrect)\b',
            r'\b(however|but|although|nevertheless|on the contrary)\b',
            r'\b(no,|nope|nah|that\'s not)\b',
            r'\b(i think you\'re|you\'re completely|totally wrong)\b'
        ]

        disagree_count = defaultdict(int)

        for i, turn in enumerate(dialogue):
            agent = turn["agent"]
            text = turn["text"].lower()

            turn_counts[agent] += 1
            word_counts[agent] += len(turn["text"].split())
            agent_topic_sum[agent] += topic_relevance_scores[i]

            # Enhanced disagreement detection
            for pattern in disagree_patterns:
                if re.search(pattern, text):
                    disagree_count[agent] += 1
                    break  # Count only once per turn

            # Topic introduction tracking
            words = [w for w in word_tokenize(
                text) if w.isalpha() and w not in self.stop_words]
            unique_words = set(words)

            for w in unique_words:
                if w not in introduced_by:
                    introduced_by[w] = (agent, i)
                    topic_introduced_count[agent] += 1

        # Calculate advanced metrics
        avg_words_per_turn = {
            agent: word_counts[agent] / turn_counts[agent] for agent in turn_counts
        }

        # Engagement score (combines multiple factors)
        engagement_scores = {}
        for agent in turn_counts:
            engagement = (
                turn_counts[agent] * 0.3 +  # Participation
                avg_words_per_turn[agent] * 0.2 +  # Verbosity
                topic_introduced_count.get(agent, 0) * 0.3 +  # Innovation
                (agent_topic_sum[agent] /
                 turn_counts[agent]) * 0.2  # Relevance
            )
            engagement_scores[agent] = engagement

        return ParticipationMetrics(
            turn_counts=dict(turn_counts),
            word_counts=dict(word_counts),
            avg_words_per_turn=avg_words_per_turn,
            topic_introduced_count=dict(topic_introduced_count),
            disagree_count=dict(disagree_count)
        )

    def _analyze_enhanced_coherence(self, embeddings, dialogue, topic_embedding):
        if len(embeddings) == 0:
            return []

        coherence_scores = []
        embeddings_tensor = torch.FloatTensor(
            embeddings).unsqueeze(0).to(self.device)
        turn_masks = torch.zeros(
            1, len(embeddings), dtype=torch.bool).to(self.device)

        with torch.no_grad():
            enhanced_embeddings, _ = self.hierarchical_attention(
                embeddings_tensor, turn_masks)
            enhanced_embeddings = enhanced_embeddings.squeeze(0).cpu().numpy()

        for i in range(len(enhanced_embeddings)):
            combined_coherence = 0.0

            # Safe similarity calculation helper
            def safe_similarity(a, b):
                try:
                    result = cosine_similarity(
                        np.atleast_2d(a), np.atleast_2d(b))
                    return result[0][0] if result.size > 0 else 0.0
                except:
                    return 0.0

            topic_coherence = safe_similarity(
                enhanced_embeddings[i], topic_embedding)

            if i == 0:
                combined_coherence = topic_coherence
            else:
                local_coherence = safe_similarity(
                    enhanced_embeddings[i-1], enhanced_embeddings[i])

                if i > 2:
                    history_avg = np.mean(enhanced_embeddings[:i-1], axis=0)
                    global_coherence = safe_similarity(
                        enhanced_embeddings[i], history_avg)
                else:
                    global_coherence = local_coherence

                combined_coherence = (
                    topic_coherence * 0.5 +
                    local_coherence * 0.3 +
                    global_coherence * 0.2
                )

            coherence_scores.append(float(combined_coherence))

        return coherence_scores

    def _analyze_dynamic_topics(self, dialogue: List[Dict], utterances: List[str], topic: str, topic_embedding: np.ndarray) -> Dict[str, Any]:
        """Analyze topics with dynamic modeling"""
        if len(utterances) < 3:
            return {'topics': {}, 'evolution': [], 'shifts': []}

        # Create timestamps (turn indices)
        timestamps = list(range(len(utterances)))

        # Fit dynamic topic model
        try:
            topic_distributions = self.topic_model.fit_transform(
                utterances, timestamps)
            topics = self.topic_model.get_topic_keywords()

            # Compute embeddings for each turn, for topic relevance calculation
            turn_embeddings = self.sentence_model.encode(
                utterances, normalize_embeddings=True)
            discussion_topic_embedding = topic_embedding  # Already precomputed and passed in
            turn_topic_relevance = [
                float(cosine_similarity(
                    [emb], [discussion_topic_embedding])[0][0])
                for emb in turn_embeddings
            ]

            # Detect topic shifts
            shifts = []

            for i in range(1, len(topic_distributions)):
                prev_agent = dialogue[i-1]['agent']
                curr_agent = dialogue[i]['agent']
                shift_magnitude = np.linalg.norm(
                    topic_distributions[i] - topic_distributions[i-1])
                # Semantic similarity as inverse of shift magnitude (optional: can use cosine similarity)
                semantic_similarity = float(cosine_similarity(
                    [topic_distributions[i-1]], [topic_distributions[i]])[0][0])
                if shift_magnitude > 0.3:
                    prev_dominant = np.argmax(topic_distributions[i-1])
                    curr_dominant = np.argmax(topic_distributions[i])
                    # If you have topic relevance scores, add them:
                    # # or use your own relevance metric
                    prev_relevance = float(max(topic_distributions[i-1]))
                    curr_relevance = float(max(topic_distributions[i]))
                    shifts.append({
                        'shift_display': f"({prev_agent}) Turn {i} → ({curr_agent}) Turn {i+1}",
                        'from_agent': prev_agent,
                        'from_turn': i,
                        'to_agent': curr_agent,
                        'to_turn': i+1,

                        'shift_magnitude': float(shift_magnitude),
                        'semantic_similarity': semantic_similarity,
                        'from_topic': int(prev_dominant),
                        'to_topic': int(curr_dominant),
                        'from_keywords': topics.get(prev_dominant, [])[:3],
                        'to_keywords': topics.get(curr_dominant, [])[:3],
                        'from_topic_relevance': prev_relevance,
                        'to_topic_relevance': curr_relevance,
                        'from_discussion_relevance': turn_topic_relevance[i-1],
                        'to_discussion_relevance': turn_topic_relevance[i],
                    })
            return {
                'topics': topics,
                'distributions': topic_distributions,
                'evolution': self.topic_model.topic_evolution,
                'shifts': shifts
            }

        except Exception as e:
            print(f"Topic modeling failed: {e}")
            return {'topics': {}, 'evolution': [], 'shifts': []}

    def _generate_enhanced_summary(self, influence_metrics: InfluenceMetrics,
                                   sentiment_metrics: SentimentMetrics,
                                   participation_metrics: ParticipationMetrics,
                                   topic_analysis: Dict[str, Any],
                                   topic_credibility_metrics: TopicCredibilityMetrics) -> str:
        """Generate enhanced summary with advanced insights"""
        summary_parts = []

        # Influence analysis
        most_influential = max(
            influence_metrics.agent_influence_count.items(),
            key=lambda x: x[1], default=("None", 0)
        )
        summary_parts.append(
            f"Most influential: {most_influential[0]} (score: {most_influential[1]:.3f})")

        # Sentiment analysis
        most_positive = max(
            sentiment_metrics.avg_sentiment_per_agent.items(),
            key=lambda x: x[1], default=("None", 0)
        )
        most_negative = min(
            sentiment_metrics.avg_sentiment_per_agent.items(),
            key=lambda x: x[1], default=("None", 0)
        )
        summary_parts.append(
            f"Most positive: {most_positive[0]} ({most_positive[1]:.3f})")
        summary_parts.append(
            f"Most negative: {most_negative[0]} ({most_negative[1]:.3f})")

        # Topic relevance
        most_relevant = max(
            sentiment_metrics.avg_topic_relevance_per_agent.items(),
            key=lambda x: x[1], default=("None", 0)
        )
        summary_parts.append(
            f"Most topic-relevant: {most_relevant[0]} ({most_relevant[1]:.3f})")

        # Participation
        most_active = max(
            participation_metrics.word_counts.items(),
            key=lambda x: x[1], default=("None", 0)
        )
        summary_parts.append(
            f"Most active: {most_active[0]} ({most_active[1]} words)")

        # Topic insights
        if topic_analysis.get('shifts'):
            summary_parts.append(
                f"Topic shifts detected: {len(topic_analysis['shifts'])}")

        if topic_credibility_metrics.credibility_ranking:
            most_credible = topic_credibility_metrics.credibility_ranking[0]
            summary_parts.append(
                f"Most Credible: {most_credible['agent']} ({most_credible['credibility_level']}, {most_credible['credibility_score']: .3f})")

            least_credible = topic_credibility_metrics.credibility_ranking[-1]
            summary_parts.append(
                f"Least credible: {least_credible['agent']} ({least_credible['credibility_level']}, {least_credible['credibility_score']: .3f})")

        return " | ".join(summary_parts)

    def _generate_enhanced_visualizations(self, dialogue: List[Dict], influence_metrics: InfluenceMetrics,
                                          sentiment_metrics: SentimentMetrics, participation_metrics: ParticipationMetrics,
                                          coherence_scores: List[float], embeddings: np.ndarray,
                                          topic_analysis: Dict[str, Any], topic_credibility_metrics: TopicCredibilityMetrics) -> Dict[str, str]:
        """Generate enhanced visualizations"""
        visualizations = {}

        # Enhanced influence network with GAT insights
        visualizations['influence_network'] = self._plot_enhanced_influence_network(
            dialogue, influence_metrics
        )

        visualizations['agent_sentiment_per_turn'] = self._plot_agent_sentiment_per_turn(
            dialogue, sentiment_metrics.sentiment_labels)

        # Enhanced sentiment trajectory with RoBERTa results
        visualizations['sentiment_trajectory'] = self._plot_enhanced_sentiment_trajectory(
            dialogue, sentiment_metrics
        )

        # Enhanced participation metrics
        visualizations['participation_metrics'] = self._plot_enhanced_participation_metrics(
            participation_metrics
        )

        # Enhanced coherence analysis
        visualizations['coherence_analysis'] = self._plot_enhanced_coherence_analysis(
            dialogue, coherence_scores
        )

        # Topic evolution visualization
        if topic_analysis.get('topics'):
            visualizations['topic_analysis'] = self._plot_topic_analysis(
                dialogue, topic_analysis
            )

            # [ADDED BEGIN] Agent-wise sentiment-per-turn table visualization
        if hasattr(sentiment_metrics, 'sentiment_word_stats'):
            visualizations['sentiment_per_turn_table'] = self._plot_agent_sentiment_table(
                dialogue,
                sentiment_metrics.sentiment_labels,
                sentiment_metrics.sentiment_word_stats
            )

        # Topic Credibility Visualization
        visualizations['topic_credibility_analysis'] = self._plot_topic_credibility_analysis(
            topic_credibility_metrics)

        return visualizations

    def _plot_to_base64(self, fig) -> str:
        """Convert plot to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        return f"data:image/png;base64,{image_base64}"

    def _plot_enhanced_influence_network(self, dialogue: List[Dict], influence_metrics: InfluenceMetrics) -> str:
        """Enhanced influence network visualization"""
        fig, ax1 = plt.subplots(figsize=(16, 8))

        # Original influence network
        G = nx.DiGraph()
        agents = {turn["agent"] for turn in dialogue}

        for agent in agents:
            G.add_node(agent)

        for pair_str, weight in influence_metrics.influence_pairs.items():
            src, dest = pair_str.split('->')
            G.add_edge(src, dest, weight=weight)

        pos = nx.spring_layout(G, k=0.5)
        node_sizes = [
            300 + influence_metrics.agent_influence_count.get(node, 0) * 50 for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                               node_color='lightblue', ax=ax1)
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax1)

        if G.number_of_edges() > 0:
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w * 0.5 for w in weights],
                                   arrowstyle='->', arrowsize=10, alpha=0.7, ax=ax1)

        ax1.set_title(
            "Traditional Influence Network: How Ideas Spread Between Agents", fontsize=15)
        ax1.axis('off')

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_enhanced_sentiment_trajectory(self, dialogue: List[Dict], sentiment_metrics: SentimentMetrics) -> str:
        """Enhanced sentiment trajectory with RoBERTa insights"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

        turns = list(range(1, len(dialogue) + 1))

        # Extract RoBERTa compound scores
        compound_scores = []
        for i, turn in enumerate(dialogue):
            # Use the sentiment data from our enhanced analysis
            label = sentiment_metrics.sentiment_labels[i]
            if label == 'positive':
                compound_scores.append(0.5)  # Positive sentiment
            elif label == 'negative':
                compound_scores.append(-0.5)  # Negative sentiment
            else:
                compound_scores.append(0.0)  # Neutral sentiment

        # Plot sentiment and topic relevance
        ax1.plot(turns, compound_scores, marker='o', label='Sentiment (RoBERTa)',
                 linewidth=2, color='blue')
        ax1.plot(turns, sentiment_metrics.topic_relevance_scores, marker='s',
                 label='Topic Relevance', linewidth=2, color='red', linestyle='--')

        for i, turn in enumerate(dialogue):
            agent = turn["agent"]
            ax1.annotate(agent, (turns[i], compound_scores[i]), textcoords="offset points", xytext=(
                0, 10), ha='center', fontsize=9, color='navy')
            ax1.annotate(agent, (turns[i], sentiment_metrics.topic_relevance_scores[i]), textcoords="offset points", xytext=(
                0, -12), ha='center', fontsize=9, color='maroon')
        ax1.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax1.set_title(
            "Enhanced Sentiment & Topic Relevance Trajectory", fontsize=14)
        ax1.set_xlabel("Turn Number")
        ax1.set_ylabel("Score")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Agent-wise sentiment comparison
        agents = list(sentiment_metrics.avg_sentiment_per_agent.keys())
        avg_sentiments = list(
            sentiment_metrics.avg_sentiment_per_agent.values())
        avg_relevance = list(
            sentiment_metrics.avg_topic_relevance_per_agent.values())

        x_pos = np.arange(len(agents))
        width = 0.35

        bars1 = ax2.bar(x_pos - width/2, avg_sentiments, width, label='Avg Sentiment',
                        color='lightblue', alpha=0.7)
        bars2 = ax2.bar(x_pos + width/2, avg_relevance, width, label='Avg Topic Relevance',
                        color='lightcoral', alpha=0.7)

        ax2.set_title(
            "Agent-wise Sentiment & Topic Relevance Comparison", fontsize=14)
        ax2.set_xlabel("Agent")
        ax2.set_ylabel("Score")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(agents)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_enhanced_participation_metrics(self, participation_metrics: ParticipationMetrics) -> str:
        """Enhanced participation metrics visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

        agents = list(participation_metrics.turn_counts.keys())

        # Turn counts and word counts
        turn_counts = [participation_metrics.turn_counts[a] for a in agents]
        word_counts = [participation_metrics.word_counts[a] for a in agents]

        ax1.bar(agents, turn_counts, color='skyblue', alpha=0.7)
        ax1.set_title("Turn Counts by Agent", fontsize=12)
        ax1.set_ylabel("Number of Turns")

        ax2.bar(agents, word_counts, color='lightcoral', alpha=0.7)
        ax2.set_title("Word Counts by Agent", fontsize=12)
        ax2.set_ylabel("Number of Words")

        # Average words per turn
        avg_words = [participation_metrics.avg_words_per_turn[a]
                     for a in agents]
        ax3.bar(agents, avg_words, color='lightgreen', alpha=0.7)
        ax3.set_title("Average Words per Turn", fontsize=12)
        ax3.set_ylabel("Words per Turn")

        # Topic introduction vs disagreement
        topic_introduced = [
            participation_metrics.topic_introduced_count.get(a, 0) for a in agents]
        disagree_counts = [
            participation_metrics.disagree_count.get(a, 0) for a in agents]

        x_pos = np.arange(len(agents))
        width = 0.35

        ax4.bar(x_pos - width/2, topic_introduced, width, label='Topics Introduced',
                color='gold', alpha=0.7)
        ax4.bar(x_pos + width/2, disagree_counts, width, label='Disagreements',
                color='salmon', alpha=0.7)
        ax4.set_title("Topic Introduction vs Disagreement", fontsize=12)
        ax4.set_xlabel("Agent")
        ax4.set_ylabel("Count")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(agents)
        ax4.legend()

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_enhanced_coherence_analysis(self, dialogue: List[Dict], coherence_scores: List[float]) -> str:
        """Enhanced coherence analysis visualization with topic awareness"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

        if len(coherence_scores) > 0:
            # Coherence trajectory - now shows topic coherence for each turn
            turns = list(range(1, len(coherence_scores) + 1))

            ax1.plot(turns, coherence_scores, marker='o',
                     color='purple', linewidth=2)
            ax1.axhline(0.3, color='red', linestyle='--',
                        label='Low Topic Coherence Threshold', alpha=0.7)
            ax1.axhline(0.7, color='green', linestyle='--',
                        label='High Topic Coherence Threshold', alpha=0.7)

            # Annotate each point with agent name and turn number
            for i, score in enumerate(coherence_scores):
                agent = dialogue[i]['agent']
                ax1.annotate(f"{agent}", (turns[i], score),
                             textcoords="offset points", xytext=(0, 10),
                             ha='center', fontsize=8, color='purple')

            ax1.set_title("Topic-Aware Coherence Analysis", fontsize=14)
            ax1.set_xlabel("Turn Number")
            ax1.set_ylabel("Coherence Score with Topic")
            ax1.set_xticks(turns[::max(1, len(turns)//10)])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Coherence distribution
            ax2.hist(coherence_scores, bins=20, alpha=0.7,
                     color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(coherence_scores), color='red', linestyle='--',
                        label=f'Mean Topic Coherence: {np.mean(coherence_scores):.3f}')
            ax2.axvline(np.median(coherence_scores), color='green', linestyle='--',
                        label=f'Median Topic Coherence: {np.median(coherence_scores):.3f}')

            ax2.set_title("Topic Coherence Score Distribution", fontsize=14)
            ax2.set_xlabel("Topic Coherence Score")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        else:
            ax1.text(0.5, 0.5, 'No coherence data available',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax2.text(0.5, 0.5, 'No coherence data available',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=14)

        plt.tight_layout()
        return self._plot_to_base64(fig)

    def _plot_topic_analysis(self, dialogue: List[Dict], topic_analysis: Dict[str, Any]) -> str:
        """Topic analysis visualization"""
        fig, ax1 = plt.subplots(figsize=(12, 7))

        topics = topic_analysis.get('topics', {})
        distributions = topic_analysis.get('distributions', [])

        if topics and len(distributions) > 0:
            # Topic evolution over turns
            n_topics = len(topics)
            colors = plt.cm.Set3(np.linspace(0, 1, n_topics))

            for topic_id in range(n_topics):
                topic_probs = [dist[topic_id] for dist in distributions]
                ax1.plot(range(1, len(topic_probs) + 1), topic_probs,
                         marker='o', label=f'Topic {topic_id}',
                         color=colors[topic_id], linewidth=2)

            ax1.set_title("Topic Evolution Over Turns", fontsize=14)
            ax1.set_xlabel("Turn Number")
            ax1.set_ylabel("Topic Probability")
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)

        return self._plot_to_base64(fig)

    # Topic Credibility

    def _analyze_topic_credibility(self, dialogue: List[Dict], topic: str, topic_embedding: np.ndarray,
                                   topic_relevance_scores: List[float]) -> 'TopicCredibilityMetrics':
        """
        Analyze agent credibility based on topic adherence and semantic quality
        """
        from collections import defaultdict
        import numpy as np

        # Initialize agent metrics
        agent_relevance_sum = defaultdict(float)
        agent_relevance_count = defaultdict(int)
        agent_contributions = defaultdict(list)
        agent_embeddings = defaultdict(list)

        # Collect agent-specific data
        for i, (turn, relevance_score) in enumerate(zip(dialogue, topic_relevance_scores)):
            agent = turn["agent"]
            text = turn["text"]

            agent_relevance_sum[agent] += relevance_score
            agent_relevance_count[agent] += 1
            agent_contributions[agent].append(text)

            # Generate embedding for this turn
            embedding = self.sentence_model.encode([text])[0]
            agent_embeddings[agent].append(embedding)

        # Calculate metrics for each agent
        agent_topic_relevance = {}
        agent_consistency_scores = {}
        agent_depth_scores = {}
        agent_off_topic_penalty = {}
        agent_credibility_scores = {}

        # Define thresholds
        TOPIC_ADHERENCE_THRESHOLD = 0.3  # Minimum relevance score to be considered on-topic
        OFF_TOPIC_PENALTY_MULTIPLIER = 0.5

        for agent in agent_relevance_sum.keys():
            # 1. Average Topic Relevance
            avg_relevance = agent_relevance_sum[agent] / \
                agent_relevance_count[agent]
            agent_topic_relevance[agent] = float(avg_relevance)

            # 2. Consistency Score (how consistently agent stays on topic)
            agent_scores = [topic_relevance_scores[i] for i, turn in enumerate(dialogue)
                            if turn["agent"] == agent]
            # Lower std = higher consistency
            consistency = 1.0 - np.std(agent_scores)
            agent_consistency_scores[agent] = float(max(0.0, consistency))

            # 3. Semantic Depth Score (average semantic richness)
            if len(agent_embeddings[agent]) > 1:
                # Calculate internal coherence of agent's contributions
                embeddings_matrix = np.array(agent_embeddings[agent])
                pairwise_similarities = []
                for i in range(len(embeddings_matrix)):
                    for j in range(i+1, len(embeddings_matrix)):
                        sim = cosine_similarity([embeddings_matrix[i]], [
                                                embeddings_matrix[j]])[0][0]
                        pairwise_similarities.append(sim)
                depth_score = np.mean(
                    pairwise_similarities) if pairwise_similarities else 0.5
            else:
                depth_score = 0.5  # Neutral score for single contribution
            agent_depth_scores[agent] = float(depth_score)

            # 4. Off-topic Penalty
            off_topic_count = sum(
                1 for score in agent_scores if score < TOPIC_ADHERENCE_THRESHOLD)
            off_topic_ratio = off_topic_count / len(agent_scores)
            penalty = off_topic_ratio * OFF_TOPIC_PENALTY_MULTIPLIER
            agent_off_topic_penalty[agent] = float(penalty)

            # 5. Overall Credibility Score (weighted combination)
            credibility = (
                avg_relevance * 0.4 +           # Topic relevance (40%)
                consistency * 0.25 +            # Consistency (25%)
                depth_score * 0.20 +            # Semantic depth (20%)
                (1.0 - penalty) * 0.15          # Off-topic penalty (15%)
            )
            agent_credibility_scores[agent] = float(credibility)

        # Create credibility ranking
        credibility_ranking = []
        sorted_agents = sorted(agent_topic_relevance.items(),
                               key=lambda x: x[1], reverse=True)

        for rank, (agent, relevance_score) in enumerate(sorted_agents, 1):
            # Determine credibility level
            score = agent_credibility_scores[agent]
            if relevance_score >= 0.7:
                level = "Excellent"
            elif relevance_score >= 0.5:
                level = "Good"
            elif relevance_score >= 0.3:
                level = "Average"
            else:
                level = "Poor"

            credibility_ranking.append({
                "agent": agent,
                "rank": rank,
                "credibility_score": score,
                "credibility_level": level,
                "topic_relevance": relevance_score,
                "consistency": agent_consistency_scores[agent],
                "depth": agent_depth_scores[agent],
                "off_topic_penalty": agent_off_topic_penalty[agent]
            })

        return TopicCredibilityMetrics(
            agent_topic_relevance=agent_topic_relevance,
            agent_consistency_scores=agent_consistency_scores,
            agent_depth_scores=agent_depth_scores,
            agent_off_topic_penalty=agent_off_topic_penalty,
            agent_credibility_scores=agent_credibility_scores,
            credibility_ranking=credibility_ranking,
            topic_adherence_threshold=TOPIC_ADHERENCE_THRESHOLD
        )

    def _plot_topic_credibility_analysis(self, topic_credibility_metrics: 'TopicCredibilityMetrics') -> str:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        agents = list(
            topic_credibility_metrics.agent_credibility_scores.keys())
        credibility_scores = [
            topic_credibility_metrics.agent_credibility_scores[agent] for agent in agents]
        relevance_scores = [
            topic_credibility_metrics.agent_topic_relevance[agent] for agent in agents]
        consistency_scores = [
            topic_credibility_metrics.agent_consistency_scores[agent] for agent in agents]
        depth_scores = [topic_credibility_metrics.agent_depth_scores[agent]
                        for agent in agents]

        colors = ['green' if score >= 0.7 else 'orange' if score >=
                  0.5 else 'red' for score in credibility_scores]
        bars1 = ax1.bar(agents, credibility_scores, color=colors, alpha=0.7)
        ax1.set_title('Agent Topic Credibility Scores',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Credibility Score')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.7, color='green', linestyle='--',
                    alpha=0.5, label='Excellent (≥0.7)')
        ax1.axhline(y=0.5, color='orange', linestyle='--',
                    alpha=0.5, label='Good (≥0.5)')
        ax1.axhline(y=0.3, color='red', linestyle='--',
                    alpha=0.5, label='Average (≥0.3)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for bar, score in zip(bars1, credibility_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

        ax2.scatter(relevance_scores, consistency_scores, c=credibility_scores,
                    cmap='RdYlGn', s=100, alpha=0.7, edgecolors='black')
        for i, agent in enumerate(agents):
            ax2.annotate(agent, (relevance_scores[i], consistency_scores[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax2.set_xlabel('Topic Relevance Score')
        ax2.set_ylabel('Consistency Score')
        ax2.set_title('Topic Relevance vs Consistency', fontsize=14)
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Credibility Score')

        # 3. Component Breakdown
        x_pos = np.arange(len(agents))
        width = 0.2

        ax3.bar(x_pos - width*1.5, relevance_scores, width,
                label='Topic Relevance', alpha=0.8, color='blue')
        ax3.bar(x_pos - width*0.5, consistency_scores, width,
                label='Consistency', alpha=0.8, color='green')
        ax3.bar(x_pos + width*0.5, depth_scores, width,
                label='Semantic Depth', alpha=0.8, color='purple')
        ax3.bar(x_pos + width*1.5, credibility_scores, width,
                label='Overall Credibility', alpha=0.8, color='red')

        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Score')
        ax3.set_title('Credibility Component Breakdown', fontsize=14)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(agents)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Credibility Ranking Table
        ax4.axis('off')
        ranking_data = []
        for entry in topic_credibility_metrics.credibility_ranking:
            ranking_data.append([
                f"#{entry['rank']}",
                entry['agent'],
                entry['credibility_level'],
                f"{entry['credibility_score']:.3f}",
                f"{entry['topic_relevance']:.3f}"
            ])

        table = ax4.table(cellText=ranking_data,
                          colLabels=['Rank', 'Agent',
                                     'Level', 'Consistency', 'Topic Relevance'],
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(5)))
        ax4.set_title('Agent Credibility Ranking', fontsize=14, pad=20)

        # Color code the table rows
        for i, entry in enumerate(topic_credibility_metrics.credibility_ranking):
            if entry['credibility_level'] == 'Excellent':
                color = 'lightgreen'
            elif entry['credibility_level'] == 'Good':
                color = 'lightyellow'
            elif entry['credibility_level'] == 'Average':
                color = 'lightcoral'
            else:
                color = 'mistyrose'

            for j in range(5):
                table[(i+1), j].set_facecolor(color)

        plt.tight_layout()
        return self._plot_to_base64(fig)


# Optional: Advanced features that can be added separately. This part is only for using high end LLM to get more accurate result.

class LLMDialogueAnalyzer:
    """Large Language Model integration for advanced dialogue analysis"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def analyze_dialogue_with_llm(self, dialogue: List[Dict], topic: str) -> Dict[str, Any]:
        """Analyze dialogue using LLM for advanced insights"""
        # This would be implemented with actual LLM API calls
        # For now, return simulated results
        return {
            "dialogue_quality": "high",
            "key_insights": [
                "Strong topic adherence throughout conversation",
                "Balanced participation between agents",
                "Positive overall sentiment trajectory"
            ],
            "suggested_improvements": [
                "Encourage more disagreement for richer discussion",
                "Introduce more specific examples"
            ]
        }


class EnsembleDialogueAnalyzer:
    """Ensemble methods for robust dialogue analysis"""

    def __init__(self):
        self.analyzers = []

    def add_analyzer(self, analyzer):
        """Add an analyzer to the ensemble"""
        self.analyzers.append(analyzer)

    def analyze_with_ensemble(self, dialogue: List[Dict], topic: str) -> Dict[str, Any]:
        """Combine results from multiple analyzers"""
        results = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze_dialogue_basic(dialogue, topic)
                results.append(result)
            except Exception as e:
                print(f"Analyzer failed: {e}")
                continue

        # Combine results (simplified ensemble logic)
        if results:
            # Average sentiment scores, combine influence metrics, etc.
            return self._combine_results(results)
        else:
            return {"error": "All analyzers failed"}

    def _combine_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Combine multiple analysis results"""

        return results[0] if results else {}


# Export the main enhanced engine
__all__ = ['EnhancedDialogueAnalysisEngine',
           'LLMDialogueAnalyzer', 'EnsembleDialogueAnalyzer']
