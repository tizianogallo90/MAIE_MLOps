import pytest

from app.model import SentimentAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()


def test_directional_positive(analyzer):
    res = analyzer.predict("I love this new release, everything works smoothly!")
    assert res["label"] == "positive"
    assert res["score"] > 0.7


def test_directional_negative(analyzer):
    res = analyzer.predict("Worst experience ever, completely broken and useless.")
    assert res["label"] == "negative"
    assert res["score"] > 0.7


def test_invariance_punctuation(analyzer):
    base = analyzer.predict("Great service")
    punctuated = analyzer.predict("Great service!!!")
    assert base["label"] == punctuated["label"]
