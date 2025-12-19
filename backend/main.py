from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import AnalysisRequest, AnalysisResults
from analysis_engine import EnhancedDialogueAnalysisEngine
import uvicorn
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')


app = FastAPI(title="Dialogue Analysis API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "http://localhost:8000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the analysis engine
analysis_engine = EnhancedDialogueAnalysisEngine()


@app.get("/")
async def root():
    return {"message": "Dialogue Analysis API is running"}


@app.post("/analyze", response_model=AnalysisResults)
async def analyze_dialogue(request: AnalysisRequest):
    """
    Analyze dialogue and return comprehensive results including visualizations.
    """
    try:
        # Validate input
        if not request.dialogues:
            raise HTTPException(
                status_code=400, detail="No dialogue data provided")

        if len(request.dialogues) != request.num_agents * request.turns_per_agent:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {request.num_agents * request.turns_per_agent} dialogue turns, got {len(request.dialogues)}"
            )

        # Run analysis
        results = analysis_engine.analyze_dialogue(request)
        return results

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Analysis engine is ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
