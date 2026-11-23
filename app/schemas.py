from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Testo del commento o post social da analizzare",
    )


class SentimentResponse(BaseModel):
    label: str
    score: float
    probabilities: dict[str, float]
