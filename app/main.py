from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from app.model import SentimentAnalyzer
from app.schemas import SentimentRequest, SentimentResponse

PREDICTION_COUNTER = Counter(
    "sentiment_predictions_total",
    "Conteggio predizioni per classe di sentiment",
    ["label"],
)
CONFIDENCE_HISTOGRAM = Histogram(
    "sentiment_prediction_confidence",
    "Distribuzione dello score di confidenza",
    buckets=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

ml_models: dict[str, SentimentAnalyzer] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["sentiment"] = SentimentAnalyzer()
    yield
    ml_models.clear()


app = FastAPI(
    title="MachineInnovators Sentiment API",
    version="1.0.0",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health_check():
    if "sentiment" not in ml_models:
        raise HTTPException(status_code=503, detail="Modello non ancora caricato")
    return {"status": "ok", "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"}


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(payload: SentimentRequest):
    analyzer = ml_models["sentiment"]
    result = analyzer.predict(payload.text)
    PREDICTION_COUNTER.labels(label=result["label"]).inc()
    CONFIDENCE_HISTOGRAM.observe(result["score"])
    return result
