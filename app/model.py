import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class SentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            top_k=None,
        )

    def predict(self, text: str) -> dict:
        results = self.pipe(text)[0]
        probs = {item["label"].lower(): round(float(item["score"]), 4) for item in results}
        best = max(results, key=lambda x: x["score"])
        return {
            "label": best["label"].lower(),
            "score": round(float(best["score"]), 4),
            "probabilities": probs,
        }
