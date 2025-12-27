# M-DIAL: Multi-Agent Discussion Analysis Framework
A comprehensive NLP/ML framework for analyzing AI-simulated multi-agent conversations. M-DIAL extracts insights on sentiment dynamics, semantic coherence, topic evolution, and agent influence patterns through integrated natural language processing and machine learning techniques.

## Overview
M-DIAL (Multi-Agent Discussion Analysis Framework) is developed as part of a Master's thesis research project at Tampere University's GPT Lab. The framework provides a unified platform for analyzing dialogue data from multi-agent systems using state-of-the-art NLP and ML methods.

## Key Capabilities
- **Sentiment Analysis**: Turn-level and dialogue-level sentiment tracking using VADER, BERT, and RoBERTa

- **Semantic Coherence**: Measure conversation flow and topic continuity with Sentence-BERT embeddings

- **Topic Modeling**: Extract and track topic shifts across conversation turns

- **Agent Influence Analysis**: Quantify individual agent impact on discussion dynamics

- **Interactive Visualization**: Real-time charts and graphs for exploring analytical results

## Research Context
This framework implements a Design Science Research (DSR) methodology to address three core research questions:

1. How can NLP/ML methods be integrated into a unified multi-agent dialogue analysis framework?

2. How should analytical outputs be represented for interpretability and actionability?

3. What is the relationship between framework performance and dialogue characteristics?

Field: Natural Language Processing, Machine Learning, Multi-Agent Systems

## Tech Stack
### Backend
- Python 3.8+

- FastAPI: REST API server

- PyTorch: Deep learning models (BERT, RoBERTa, Sentence-BERT)

- NLTK: Text preprocessing and VADER sentiment analysis

- Pandas/NumPy: Data manipulation and numerical computation

### Frontend
- React 18: User interface framework

- Material-UI (MUI): Component library

- Axios: HTTP client for API communication

- Recharts: Data visualization

Project Structure
```bash
multi-agent-discussion-analysis-framework/
├── backend/
│   ├── __pycache__/    
│   ├── venv/
│   ├── .env
│   ├── analysis_engine.py
│   ├── main.py
│   ├── models.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   │   └── index.html          
│   ├── src/
│   │   ├── App.js         
│   │   ├── components/    
│   │   │   ├── InpuForm.js
│   │   │   └── ResultsDispplay.js
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.css
│   │   └── index.js            
│   ├── package.json            
│   └── package-lock.json
├── MultiAgent_Dialogue_Example.csv
├── README.md                  
└── LICENSE                  
```
## Installation
### Prerequisites
- Python 3.8+

- Node.js 14+ and npm

- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/subir-raju/multi-agent-discussion-analysis-framework.git
cd multi-agent-discussion-analysis-framework
```
### Step 2: Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download NLTK Data (required for VADER):

```python
import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')
```
### Step 3: Frontend Setup
```bash
cd ../frontend
npm install
```
### Usage
Starting the Backend
```bash
cd backend
source venv/bin/activate      # On Windows: venv\Scripts\activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend API will be available at: `http://localhost:8000`

### Starting the Frontend
Open a new terminal:

```bash
cd frontend
npm start
```
Frontend will open automatically at: `http://localhost:3000`


## Features
### 1. Sentiment Analysis
- VADER: Rule-based sentiment for social media-style text

- BERT/RoBERTa: Transformer-based contextual sentiment

- Turn-level tracking: Sentiment evolution across conversation

- Agent-specific sentiment: Individual sentiment profiles

### 2. Semantic Coherence
- Sentence-BERT embeddings: Dense vector representations

- Cosine similarity: Measure topic continuity between turns

- Coherence score: Overall conversation flow metric

### 3. Topic Modeling
- Extractive keywords: NLTK-based topic extraction

- Topic shift detection: Identify conversation pivots

- Topic clustering: Group related discussion themes

### 4. Agent Influence
- Turn frequency: Speaking time distribution

- Sentiment impact: Influence on overall tone

- Topic initiation: Identify discussion leaders

### 5. Interactive Visualization
- Line charts for sentiment trends

- Bar charts for agent comparisons

- Heatmaps for topic-agent relationships

- Export capabilities for reports



## Contributing
This research project was developed for academic purposes. Feedback, suggestions, and contributions are welcome.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
