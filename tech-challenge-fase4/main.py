"""MedMonitor — CLI.

Exemplos:
  python main.py demo                         # fluxo multimodal completo + relatório
  python main.py stream --patient P003        # monitor de sinais vitais em tempo real (replay)
  python main.py audio data/audio/P003_consulta_dispneia.wav --patient P003
  python main.py video data/video/P001_agachamento.mp4 --profile agachamento --patient P001
  python main.py physionet --record 5         # dados reais (PhysioNet BIDMC)
  python main.py azure-check                  # valida as credenciais do Azure
  python main.py prepare                      # gera/baixa todos os dados de demonstração
  python main.py gui                          # aplicativo desktop da equipe médica
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from medmonitor.config import azure_language_enabled, azure_speech_enabled, load_config, resolve  # noqa: E402


def cmd_prepare(cfg, args):
    from medmonitor.pipeline import ensure_audio, ensure_data, ensure_video
    print("Dados sintéticos:", ensure_data(cfg, regenerate=True)["vitals"].parent)
    print("Áudios:", ensure_audio(cfg))
    print("Vídeos:", ensure_video(cfg))


def cmd_demo(cfg, args):
    from medmonitor.pipeline import run_pipeline
    print(f"Azure Speech: {'ATIVO' if azure_speech_enabled(cfg) else 'fallback local'} | "
          f"Azure Language: {'ATIVO' if azure_language_enabled(cfg) else 'fallback local'}\n")
    t = time.time()
    res = run_pipeline(cfg, with_audio=not args.no_audio, with_video=not args.no_video,
                       reuse_video=not args.reprocess_video, regenerate=args.regenerate, quiet=args.quiet)
    print(f"\n{'=' * 78}\nRESUMO POR PACIENTE (risco multimodal)\n{'=' * 78}")
    for p in res["patients"]:
        s = " ".join(f"{k}={v:.2f}" for k, v in p["modality_scores"].items())
        print(f"{p['patient_id']:8} risco {p['risk']:.2f} ({p['level']:8}) | {s}")
        for c in p["correlations"]:
            print(f"         ↳ [{c['severity']}] {c['title']}")
    a = res["alerts"]
    print(f"\nAlertas: {a['dispatched']} despachados, {a['suppressed']} suprimidos (cooldown) — {a['by_severity']}")
    for row in res["metrics"]["vitals_methods"]:
        print(f"  vitais · {row['metodo']:36} recall={row['recall']} precisão={row['precisao']} F1={row['f1']}")
    print(f"\nRelatório: {resolve(cfg['paths']['output_dir']) / 'relatorio_resultados.md'}  ({time.time() - t:.0f}s)")


def cmd_stream(cfg, args):
    import pandas as pd
    from medmonitor.alerts import AlertManager
    from medmonitor.anomaly.vitals import StreamingVitalsMonitor, news2_score
    from medmonitor.data.synthetic import load_csv
    from medmonitor.pipeline import ensure_data

    df = load_csv(ensure_data(cfg)["vitals"])
    df = df[df["patient_id"] == args.patient].reset_index(drop=True)
    if args.from_hour:
        warm = int(cfg["vitals"]["baseline_hours"] * 60)
        start = max(0, int(args.from_hour * 60) - warm)
        df = df.iloc[start:]
    mon = StreamingVitalsMonitor(args.patient, cfg)
    mgr = AlertManager(cfg, resolve(cfg["paths"]["output_dir"]) / "stream_alerts.jsonl")
    print(f"Monitorando {args.patient} ({len(df)} amostras, {args.speed or 'máx.'} amostras/s). Ctrl+C para sair.\n")
    try:
        for i, row in enumerate(df.to_dict(orient="records")):
            alerts = mon.update(row)
            if i % args.every == 0:
                n, _ = news2_score(row)
                print(f"\033[2m{pd.Timestamp(row['timestamp']):%d/%m %H:%M}  FC {row['hr']:>3}  PA {row['sbp']}/{row['dbp']}"
                      f"  SpO2 {row['spo2']}%  FR {row['rr']:>2}  T {row['temp']:.1f}  NEWS2 {n}\033[0m")
            mgr.dispatch_many(alerts)
            if args.speed:
                time.sleep(1 / args.speed)
    except KeyboardInterrupt:
        pass
    print(f"\n{len(mgr.history)} alertas emitidos, {mgr.suppressed} suprimidos.")


def cmd_audio(cfg, args):
    from medmonitor.alerts import AlertManager
    from medmonitor.audio.analyzer import analyze_audio
    r = analyze_audio(args.file, cfg, args.patient, args.timestamp)
    print(json.dumps({k: v for k, v in r.to_dict().items() if k != "alerts"}, ensure_ascii=False, indent=2, default=str))
    AlertManager(cfg, resolve(cfg["paths"]["alerts_log"])).dispatch_many(r.alerts)


def cmd_video(cfg, args):
    from medmonitor.alerts import AlertManager
    from medmonitor.video.analyzer import VideoAnalyzer
    if args.openpose:
        cfg["video"]["pose_backend"] = "openpose"
    r = VideoAnalyzer(cfg, with_objects=not args.no_objects).analyze(
        args.file, args.patient, args.profile, resolve(cfg["paths"]["output_dir"]) / "video", args.timestamp,
        args.target, args.start, args.max_seconds, use_zones=args.zones)
    print(json.dumps(r.summary, ensure_ascii=False, indent=2, default=str))
    AlertManager(cfg, resolve(cfg["paths"]["alerts_log"])).dispatch_many(r.alerts)
    print(f"\nVídeo anotado: {r.annotated_path}\nRelatório: {r.report_path}")


def cmd_physionet(cfg, args):
    from medmonitor.anomaly.vitals import detect_vitals_anomalies
    from medmonitor.data.physionet import load_bidmc, physionet_config
    from medmonitor.report import plot_vitals
    df = load_bidmc(args.record, resolve(cfg["paths"]["data_dir"]) / "physionet")
    pcfg = physionet_config(cfg)
    r = detect_vitals_anomalies(df, pcfg)
    print(f"{r.patient_id}: {len(df)} amostras (1 Hz) — sinais: {[c for c in ('hr', 'rr', 'spo2') if c in df]}")
    print(df[["hr", "rr", "spo2"]].describe().round(1).to_string())
    for a in r.alerts:
        print(f"[{a.severity.name}] {a.timestamp[11:19]} {a.title}: {a.description}")
    if not r.alerts:
        print("Nenhuma anomalia persistente detectada.")
    out = resolve(cfg["paths"]["output_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print("Figura:", plot_vitals(r, out / f"physionet_{r.patient_id}.png"))


def cmd_gui(cfg, args):
    from medmonitor.gui.app import main as gui_main
    sys.exit(gui_main())


def cmd_azure_check(cfg, args):
    """Valida as credenciais: TTS → Speech to Text → Text Analytics, com uma frase de teste."""
    import numpy as np
    from medmonitor.audio.azure_speech import transcribe_azure
    from medmonitor.audio.azure_text import analyze_text
    from medmonitor.data.demo_audio import _tts

    phrase = "Doutor, estou com falta de ar e uma dor forte no peito desde ontem."
    if not azure_speech_enabled(cfg):
        print("✗ AZURE_SPEECH_KEY/AZURE_SPEECH_REGION não configurados (.env)")
    else:
        y = _tts(phrase, cfg)
        t = time.time()
        r = transcribe_azure(np.asarray(y, dtype=np.float32), 16000, cfg)
        print(f"✓ Azure Speech to Text ({time.time() - t:.1f}s, confiança {r['confidence']}): {r['text']}")
        print(f"  {len(r['words'])} palavras com timestamp; primeira: {r['words'][:1]}")
    if not azure_language_enabled(cfg):
        print("✗ AZURE_LANGUAGE_KEY/AZURE_LANGUAGE_ENDPOINT não configurados (.env)")
    else:
        r = analyze_text(phrase, cfg)
        print(f"✓ Azure Text Analytics ({r['source']}): sentimento={r['sentiment']['label']} {r['sentiment']['scores']}")
        print(f"  frases-chave: {r['key_phrases']}")
        print(f"  entidades de saúde: {[(e['text'], e['category']) for e in r['health_entities']]}")


def main():
    ap = argparse.ArgumentParser(description="MedMonitor — monitoramento multimodal de pacientes")
    ap.add_argument("--config", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare", help="gera dados sintéticos, áudios e vídeos de demonstração")
    d = sub.add_parser("demo", help="executa o fluxo multimodal completo")
    d.add_argument("--no-audio", action="store_true")
    d.add_argument("--no-video", action="store_true")
    d.add_argument("--reprocess-video", action="store_true", help="ignora resultados de vídeo em cache")
    d.add_argument("--regenerate", action="store_true", help="regera os dados sintéticos")
    d.add_argument("--quiet", action="store_true", help="não imprime cada alerta")

    s = sub.add_parser("stream", help="replay em tempo real dos sinais vitais de um paciente")
    s.add_argument("--patient", default="P003")
    s.add_argument("--speed", type=float, default=20, help="amostras por segundo (0 = sem pausa)")
    s.add_argument("--from-hour", type=float, default=0, help="começa N horas após a admissão (com aquecimento)")
    s.add_argument("--every", type=int, default=10, help="imprime 1 a cada N amostras")

    a = sub.add_parser("audio", help="analisa um áudio de consulta")
    a.add_argument("file")
    a.add_argument("--patient", default="desconhecido")
    a.add_argument("--timestamp", default=None)

    v = sub.add_parser("video", help="analisa um vídeo clínico")
    v.add_argument("file")
    v.add_argument("--patient", default="desconhecido")
    v.add_argument("--profile", default=None, help="agachamento | sentar_levantar | elevacao_braco | flexao_cotovelo")
    v.add_argument("--target", default="largest", help="pessoa monitorada: largest | left | right | center")
    v.add_argument("--zones", action="store_true", help="ativa monitoramento das áreas críticas")
    v.add_argument("--openpose", action="store_true", help="usa o backend OpenPose (requer pesos Caffe)")
    v.add_argument("--no-objects", action="store_true")
    v.add_argument("--timestamp", default=None)
    v.add_argument("--start", type=float, default=0.0)
    v.add_argument("--max-seconds", type=float, default=None)

    sub.add_parser("azure-check", help="testa as credenciais Azure (Speech + Language)")
    sub.add_parser("gui", help="abre o aplicativo desktop da equipe médica")

    p = sub.add_parser("physionet", help="detecção em sinais vitais reais do PhysioNet (BIDMC)")
    p.add_argument("--record", type=int, default=1)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    {"prepare": cmd_prepare, "demo": cmd_demo, "stream": cmd_stream, "audio": cmd_audio, "video": cmd_video,
     "physionet": cmd_physionet, "azure-check": cmd_azure_check, "gui": cmd_gui}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
