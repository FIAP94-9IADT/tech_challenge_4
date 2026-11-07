from medmonitor.alerts import Alert, AlertManager, Severity
from medmonitor.audio.azure_text import _local_sentiment, find_critical_terms
from medmonitor.fusion import correlate, fuse_patient, noisy_or


def _a(mod, title, ts, sev=Severity.WARNING, **ev):
    return Alert("P1", mod, sev, title, title, ts, evidence=ev)


def test_noisy_or():
    assert noisy_or({}, {}) == 0
    assert abs(noisy_or({"a": 0.5, "b": 0.5}, {"a": 1, "b": 1}) - 0.75) < 1e-9


def test_opioid_respiratory_rule(cfg):
    alerts = [_a("prescription", "Prescrição atípica: morfina (alta vigilância)", "2026-01-01T21:00:00", medicamento="morfina"),
              _a("vitals", "Dessaturação + Bradipneia", "2026-01-01T22:15:00", Severity.CRITICAL)]
    out = correlate("P1", alerts, cfg)
    assert any("opioide" in a.title and a.severity is Severity.CRITICAL for a in out)


def test_rule_respects_order_and_window(cfg):
    late = [_a("prescription", "x", "2026-01-01T21:00:00", medicamento="morfina"),
            _a("vitals", "Bradipneia", "2026-01-02T08:00:00")]
    before = [_a("vitals", "Bradipneia", "2026-01-01T20:00:00"),
              _a("prescription", "x", "2026-01-01T21:00:00", medicamento="morfina")]
    assert not correlate("P1", late, cfg) and not any("opioide" in a.title for a in correlate("P1", before, cfg))


def test_stroke_rule_raises_risk(cfg):
    alerts = [_a("vitals", "Taquicardia", "2026-01-01T02:00:00"),
              _a("audio", "Padrão de fala sugestivo de disartria", "2026-01-01T03:00:00")]
    f = fuse_patient("P1", {"vitals": 0.3, "audio": 0.6}, alerts, cfg)
    assert any("AVC" in c.title for c in f.correlations) and f.risk >= 0.9 and f.level == "alto"


def test_alert_cooldown(cfg, tmp_path):
    mgr = AlertManager({**cfg, "alerts": {**cfg["alerts"], "console": False}}, tmp_path / "a.jsonl")
    a1 = _a("vitals", "Taquicardia", "2026-01-01T10:00:00")
    a2 = _a("vitals", "Taquicardia", "2026-01-01T10:10:00")
    a3 = _a("vitals", "Taquicardia", "2026-01-01T11:00:00")
    assert [mgr.dispatch(a) for a in (a1, a2, a3)] == [True, False, True]
    assert len((tmp_path / "a.jsonl").read_text().splitlines()) == 2


def test_teams_payload_is_adaptive_card(cfg):
    mgr = AlertManager({**cfg, "alerts": {**cfg["alerts"], "webhook_format": "teams"}})
    payload = mgr._webhook_payload(_a("fusion", "Teste", "2026-01-01T10:00:00", Severity.CRITICAL, x=1))
    assert payload["attachments"][0]["content"]["type"] == "AdaptiveCard"


def test_critical_terms_with_negation(cfg):
    terms = find_critical_terms("Não tenho falta de ar, mas sinto dor no peito e cansaço para respirar.",
                                cfg["audio"]["critical_terms"])
    got = {(t["term"], t["negated"]) for t in terms}
    assert ("falta de ar", True) in got and ("dor no peito", False) in got
    assert ("cansaço para respirar", False) in got and not any(t == "cansaço" for t, _ in got)


def test_local_sentiment():
    assert _local_sentiment("Estou me sentindo bem melhor hoje.")["label"] == "positive"
    assert _local_sentiment("Estou com muita dor e falta de ar.")["label"] == "negative"
    assert _local_sentiment("Não estou bem.")["label"] == "negative"


def test_telegram_message_and_routing(cfg, monkeypatch):
    import medmonitor.alerts as al

    sent = []

    class FakeResp:
        def json(self):
            return {"ok": True, "result": {}}

    monkeypatch.setattr(al.requests, "post", lambda url, json, timeout: sent.append((url, json)) or FakeResp())
    monkeypatch.setattr(al.time, "sleep", lambda s: None)
    c = {**cfg, "alerts": {**cfg["alerts"], "console": False, "webhook_url": "",
                           "telegram": {"token": "123:ABC", "chat_id": "42", "min_severity": "CRITICAL"}}}
    mgr = AlertManager(c)
    mgr.dispatch(_a("vitals", "Taquicardia <FC>", "2026-01-01T10:00:00", Severity.WARNING))   # abaixo do mínimo
    mgr.dispatch(_a("fusion", "AVC & arritmia", "2026-01-01T11:00:00", Severity.CRITICAL, x="<b>"))
    mgr.telegram.flush(5)
    assert len(sent) == 1
    url, body = sent[0]
    assert url.endswith("/bot123:ABC/sendMessage") and body["chat_id"] == "42" and body["parse_mode"] == "HTML"
    assert "<b>CRÍTICO — AVC &amp; arritmia</b>" in body["text"] and "&lt;b&gt;" in body["text"]  # HTML escapado
