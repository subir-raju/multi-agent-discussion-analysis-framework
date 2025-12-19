from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class DialogueTurn(BaseModel):
    agent: str
    text: str


class AnalysisRequest(BaseModel):
    topic: str
    num_agents: int
    turns_per_agent: int
    dialogues: List[DialogueTurn]


class InfluenceMetrics(BaseModel):
    agent_influence_count: Dict[str, int]
    influence_pairs: Dict[str, int]


class SentimentMetrics(BaseModel):
    sentiment_labels: List[str]
    topic_relevance_scores: List[float]
    avg_sentiment_per_agent: Dict[str, float]
    avg_topic_relevance_per_agent: Dict[str, float]


class ParticipationMetrics(BaseModel):
    turn_counts: Dict[str, int]
    word_counts: Dict[str, int]
    avg_words_per_turn: Dict[str, float]
    topic_introduced_count: Dict[str, int]
    disagree_count: Dict[str, int]


class TopicCredibilityMetrics(BaseModel):
    # Average topic relevance per agent
    agent_topic_relevance: Dict[str, float]
    agent_consistency_scores: Dict[str, float]
    agent_depth_scores: Dict[str, float]  # Semantic depth of contributions
    agent_off_topic_penalty: Dict[str, float]
    agent_credibility_scores: Dict[str, float]  # Overall credibility score
    credibility_ranking: List[Dict[str, Any]]
    topic_adherence_threshold: float  # Threshold used for evaluation


class AnalysisResults(BaseModel):
    influence_metrics: InfluenceMetrics
    sentiment_metrics: SentimentMetrics
    participation_metrics: ParticipationMetrics
    topic_credibility_metrics: TopicCredibilityMetrics
    coherence_scores: List[float]
    topic_shifts: List[Dict[str, Any]]
    visualizations: Dict[str, str]  # base64 encoded images
    summary_text: str
