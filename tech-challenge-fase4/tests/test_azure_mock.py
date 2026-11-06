"""Exercita o parsing das respostas do Azure AI Language com um cliente simulado."""

from types import SimpleNamespace as NS

import azure.ai.textanalytics as ta

from medmonitor.audio.azure_text import analyze_text


class FakeClient:
    def __init__(self, endpoint, credential):
        self.endpoint = endpoint

    def analyze_sentiment(self, docs, language, show_opinion_mining):
        scores = NS(positive=0.05, neutral=0.1, negative=0.85)
        sentence = NS(text=docs[0], sentiment="negative", confidence_scores=scores,
                      mined_opinions=[NS(target=NS(text="ar", sentiment="negative"), assessments=[NS(text="falta")])])
        return [NS(is_error=False, sentiment="negative", confidence_scores=scores, sentences=[sentence])]

    def extract_key_phrases(self, docs, language):
        return [NS(is_error=False, key_phrases=["falta de ar", "remédio para dor"])]

    def begin_analyze_healthcare_entities(self, docs, language):
        ent = NS(text="falta de ar", category="SymptomOrSign", confidence_score=0.97,
                 assertion=None, normalized_text="Dyspnea")
        return NS(result=lambda: [NS(is_error=False, entities=[ent])])


def test_azure_language_parsing(cfg, monkeypatch):
    monkeypatch.setattr(ta, "TextAnalyticsClient", FakeClient)
    c = {**cfg, "azure": {**cfg["azure"], "language_key": "k", "language_endpoint": "https://x"}}
    r = analyze_text("Doutor, eu estou com falta de ar.", c)
    assert r["source"] == "azure-language"
    assert r["sentiment"]["label"] == "negative" and r["sentiment"]["sentences"][0]["opinions"][0]["target"] == "ar"
    assert r["key_phrases"] == ["falta de ar", "remédio para dor"]
    assert r["health_entities"][0]["normalized"] == "Dyspnea" and not r["health_entities"][0]["negated"]
    assert r["critical_terms"][0]["term"] == "falta de ar"


def test_azure_failure_falls_back_to_local(cfg, monkeypatch):
    class Broken(FakeClient):
        def analyze_sentiment(self, *a, **k):
            raise ConnectionError("sem rede")

    monkeypatch.setattr(ta, "TextAnalyticsClient", Broken)
    c = {**cfg, "azure": {**cfg["azure"], "language_key": "k", "language_endpoint": "https://x"}}
    assert analyze_text("Estou bem melhor.", c)["source"] == "local"
