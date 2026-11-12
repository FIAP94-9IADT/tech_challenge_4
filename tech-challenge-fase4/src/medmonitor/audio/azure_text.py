"""Análise da transcrição com Azure AI Language (Text Analytics).

Recursos usados:
  * Análise de sentimento com mineração de opinião (por frase)
  * Extração de frases-chave
  * Text Analytics for Health — entidades clínicas (sintomas, medicamentos,
    partes do corpo) com detecção de negação ("não tenho dor" não é sintoma)

Os termos críticos configurados em `config.yaml` são sempre procurados (com
detecção simples de negação), garantindo rastreabilidade mesmo sem Azure.
Sem credenciais, o sentimento é estimado por um léxico local em português e o
resultado é marcado com `source="local"`.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from ..config import azure_language_enabled

log = logging.getLogger(__name__)

MAX_CHARS = 5000  # limite por documento do serviço (5.120 caracteres)
NEGATIONS = ("não", "nao", "nunca", "nem", "sem")

_POS = {"bem", "melhor", "ótimo", "otimo", "tranquilo", "tranquila", "consegui", "bom", "boa", "aliviado", "aliviada",
        "disposto", "disposta", "feliz", "animado", "animada", "obrigado", "obrigada", "melhorando"}
_NEG = {"dor", "dores", "cansado", "cansada", "cansaço", "mal", "pior", "medo", "falta", "difícil", "dificil",
        "fraco", "fraca", "fraqueza", "preocupado", "preocupada", "tontura", "ruim", "sufoco", "aperto", "exausto",
        "exausta", "piorou", "angustiado", "angustiada", "enrolada", "formigamento", "desmaio", "chiado"}


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


def _chunks(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > MAX_CHARS and cur:
            out.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    return out + ([cur] if cur else [])


def find_critical_terms(text: str, lexicon: dict[str, list[str]]) -> list[dict]:
    """Busca termos críticos ignorando acentos, com negação em janela de 3 palavras."""
    norm = _norm(text)
    found = []
    for category, terms in lexicon.items():
        for term in terms:
            for m in re.finditer(r"\b" + re.escape(_norm(term)) + r"\b", norm):
                before = norm[: m.start()].split()[-3:]
                negated = any(w in NEGATIONS for w in before)
                found.append({"term": term, "category": category, "negated": negated, "position": m.start()})
    # remove termos contidos em outro termo maior no mesmo ponto (ex.: "cansaço" dentro de "cansaço para respirar")
    found.sort(key=lambda f: (f["position"], -len(f["term"])))
    dedup, last_end = [], -1
    for f in found:
        if f["position"] >= last_end:
            dedup.append(f)
            last_end = f["position"] + len(_norm(f["term"]))
    return dedup


def _local_sentiment(text: str) -> dict:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    results, total = [], {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for s in sentences:
        toks = re.findall(r"\w+", s.lower())
        pos = sum(t in _POS for t in toks)
        neg = sum(t in _NEG for t in toks)
        # "não estou bem" inverte o positivo
        for i, t in enumerate(toks):
            if t in _POS and any(w in NEGATIONS for w in toks[max(0, i - 2):i]):
                pos -= 1
                neg += 1
        label = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        results.append({"text": s, "sentiment": label})
        total[label] += 1
    n = max(len(sentences), 1)
    scores = {k: round(v / n, 2) for k, v in total.items()}
    overall = max(scores, key=scores.get) if sentences else "neutral"
    if scores["positive"] and scores["negative"] and abs(scores["positive"] - scores["negative"]) < 0.2:
        overall = "mixed"
    return {"label": overall, "scores": scores, "sentences": results}


def _azure_analyze(text: str, cfg: dict) -> dict:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    az = cfg["azure"]
    client = TextAnalyticsClient(endpoint=az["language_endpoint"], credential=AzureKeyCredential(az["language_key"]))
    docs = _chunks(text)
    lang = cfg["audio"]["language"]

    sent_docs = [d for d in client.analyze_sentiment(docs, language=lang, show_opinion_mining=True) if not d.is_error]
    sentences, agg = [], {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for d in sent_docs:
        for k in agg:
            agg[k] += getattr(d.confidence_scores, k) / len(sent_docs)
        for s in d.sentences:
            sentences.append({"text": s.text, "sentiment": s.sentiment,
                              "scores": {"positive": s.confidence_scores.positive,
                                         "negative": s.confidence_scores.negative},
                              "opinions": [{"target": o.target.text, "sentiment": o.target.sentiment,
                                            "assessments": [a.text for a in o.assessments]} for o in s.mined_opinions]})
    labels = {d.sentiment for d in sent_docs}
    sentiment = {"label": labels.pop() if len(labels) == 1 else "mixed",
                 "scores": {k: round(v, 3) for k, v in agg.items()}, "sentences": sentences}

    key_phrases = []
    for d in client.extract_key_phrases(docs, language=lang):
        if not d.is_error:
            key_phrases.extend(d.key_phrases)

    health = []
    try:  # Text Analytics for Health (pode não estar disponível em todas as regiões/idiomas)
        poller = client.begin_analyze_healthcare_entities(docs, language=lang.split("-")[0])
        for d in poller.result():
            if d.is_error:
                continue
            for e in d.entities:
                health.append({
                    "text": e.text, "category": e.category, "confidence": round(e.confidence_score, 3),
                    "negated": bool(e.assertion and e.assertion.certainty and "negative" in str(e.assertion.certainty).lower()),
                    "normalized": e.normalized_text,
                })
    except Exception as exc:  # noqa: BLE001
        log.info("Text Analytics for Health indisponível: %s", exc)

    return {"sentiment": sentiment, "key_phrases": key_phrases, "health_entities": health, "source": "azure-language"}


def analyze_text(text: str, cfg: dict) -> dict:
    result = None
    if text and azure_language_enabled(cfg):
        try:
            result = _azure_analyze(text, cfg)
        except Exception as exc:  # noqa: BLE001
            log.warning("Azure Language indisponível (%s); usando análise local.", exc)
    if result is None:
        result = {"sentiment": _local_sentiment(text), "key_phrases": [], "health_entities": [], "source": "local"}
    result["critical_terms"] = find_critical_terms(text, cfg["audio"]["critical_terms"])
    return result
