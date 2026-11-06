"""Geração do relatório de resultados (Markdown + figuras PNG)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .anomaly.vitals import VITAL_NAMES, VITAL_UNITS  # noqa: E402

SERIES = "#2a78d6"
DETECTED = "#e34948"
TRUTH = "#9a9893"
MUTED = "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c9c7c1",
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
                     "axes.grid": True, "grid.color": "#ecebe8", "grid.linewidth": 0.6})


def _spans(ax, t, mask, color, alpha, y0=0, y1=1):
    m = pd.Series(mask).to_numpy()
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            ax.axvspan(t.iloc[i], t.iloc[j], ymin=y0, ymax=y1, color=color, alpha=alpha, lw=0)
            i = j + 1
        else:
            i += 1


def plot_vitals(r, path: Path) -> Path:
    df = r.frame
    cols = [c for c in VITAL_NAMES if c in df]
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 1.55 * len(cols) + 0.6), sharex=True)
    truth = (df.get("anomaly_type", pd.Series([""] * len(df))).replace("artifact", "") != "")
    for ax, c in zip(axes, cols):
        ax.plot(df["timestamp"], df[c], color=SERIES, lw=1.0)
        _spans(ax, df["timestamp"], truth, TRUTH, 0.35, 0.0, 0.12)
        _spans(ax, df["timestamp"], df["anomaly"], DETECTED, 0.18)
        ax.set_ylabel(f"{VITAL_NAMES[c]}\n({VITAL_UNITS[c]})", rotation=0, ha="right", va="center")
    axes[0].set_title(f"{r.patient_id} — sinais vitais  |  faixa vermelha: anomalia detectada  ·  "
                      f"barra cinza na base: anomalia real (ground truth)", loc="left", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_movement(r, path: Path) -> Path:
    df = r.frame
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.plot(df["timestamp"], df["activity"], color=SERIES, lw=0.6)
    _spans(ax, df["timestamp"], df.get("anomaly_type", "") != "", TRUTH, 0.35, 0.0, 0.08)
    _spans(ax, df["timestamp"], df["anomaly"], DETECTED, 0.18)
    _spans(ax, df["timestamp"], df["night"], "#6250d6", 0.05)
    for ev in r.events:
        ax.annotate(ev["type"], (df["timestamp"].iloc[ev["start"]], ax.get_ylim()[1] * 0.9), fontsize=8, color=MUTED)
    ax.set_ylabel("atividade\n(contagens/min)")
    ax.set_title(f"{r.patient_id} — movimentação no leito (faixa lilás = noite)", loc="left", fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_risk(patients: list[dict], path: Path) -> Path:
    df = pd.DataFrame(patients).sort_values("risk")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 1))
    bars = ax.barh(df["patient_id"], df["risk"], color=SERIES, height=0.55)
    for b, (_, row) in zip(bars, df.iterrows()):
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f"{row['risk']:.2f} ({row['level']})",
                va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("risco fundido (0–1)")
    ax.set_title("Risco multimodal por paciente", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(sem dados)_"
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def write_results_report(cfg, results: dict, vit_results: dict, mov_results: dict, out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for pid, r in vit_results.items():
        plot_vitals(r, fig_dir / f"vitals_{pid}.png")
    for pid, r in mov_results.items():
        if r.events:
            plot_movement(r, fig_dir / f"movement_{pid}.png")
    plot_risk(results["patients"], fig_dir / "risk.png")

    m = results["metrics"]
    az = results["azure"]
    L = [f"# Relatório de resultados — MedMonitor", "",
         f"Execução em {datetime.fromisoformat(results['generated_at']):%d/%m/%Y %H:%M}. "
         f"Azure Speech: **{'ativo' if az['speech'] else 'fallback local'}** · "
         f"Azure Language: **{'ativo' if az['language'] else 'fallback local'}**.", "",
         "## 1. Risco multimodal por paciente", "", "![risco](figures/risk.png)", "",
         _table([{"Paciente": p["patient_id"], "Cenário": p.get("cenario"), "Risco": p["risk"], "Nível": p["level"],
                  "Vitais": p["modality_scores"].get("vitals"), "Prescr.": p["modality_scores"].get("prescription"),
                  "Movim.": p["modality_scores"].get("movement"), "Áudio": p["modality_scores"].get("audio"),
                  "Vídeo": p["modality_scores"].get("video"), "Alertas críticos": p["n_critical"]}
                 for p in results["patients"]]), "",
         "## 2. Alertas de fusão (correlação entre modalidades)", ""]
    fused = [c for p in results["patients"] for c in p["correlations"]]
    L += [_table([{"Horário": c["timestamp"][5:16].replace("T", " "), "Paciente": c["patient_id"],
                   "Severidade": c["severity"], "Alerta": c["title"],
                   "Evidências": "; ".join(f"{k}: {v}" for k, v in c["evidence"].items())} for c in fused]), ""]
    a = results["alerts"]
    L += [f"Total: {a['generated']} alertas gerados, {a['dispatched']} despachados, {a['suppressed']} suprimidos "
          f"pelo cooldown. Por severidade: {a['by_severity']}. Por modalidade: {a['by_modality']}.", "",
          "## 3. Sinais vitais — desempenho dos detectores", "", _table(m["vitals_methods"]), "",
          "Por paciente (ensemble):", "", _table(m["vitals_by_patient"]), "",
          "### Monitor em tempo real (streaming) — atraso até o primeiro alerta", "", _table(m["streaming"]), ""]
    for pid in vit_results:
        L += [f"![{pid}](figures/vitals_{pid}.png)", ""]
    L += ["## 4. Movimentação no leito", "", _table([{**r, "tipos_detectados": ", ".join(r["tipos_detectados"])}
                                                     for r in m["movement"]]), ""]
    for pid, r in mov_results.items():
        if r.events:
            L += [f"![{pid}](figures/movement_{pid}.png)", ""]
    L += ["## 5. Prescrições", "", _table([m["prescriptions"]]), "",
          _table([{"Paciente": p["patient_id"], "Horário": p["timestamp"][11:16], "Medicamento": p["drug"],
                   "Dose/dia": f"{p['daily_dose']:g} {p.get('unit', 'mg')}", "Achados": "; ".join(p["findings"]),
                   "Esperado": p["expected_anomaly"], "Detectado": p["anomaly"]}
                  for p in results["prescriptions"] if p["anomaly"] or p["expected_anomaly"]]), "",
          "## 6. Áudio de consultas", "", _table(m["audio"]), ""]
    for au in results["audio"]:
        L += [f"**{au['file']}** — transcrição ({au['transcript']['source']}): _{au['transcript']['text']}_  ",
              f"Sentimento ({au['text_analysis']['source']}): {au['text_analysis']['sentiment']['label']} · "
              f"termos críticos: {', '.join(t['term'] + (' (negado)' if t['negated'] else '') for t in au['text_analysis']['critical_terms']) or '—'}", ""]
    L += ["## 7. Vídeo", "", _table([{"Vídeo": v["file"], "Descrição": v.get("description"), "Perfil": v["profile"],
                                      "Eventos": ", ".join(sorted({e["type"] for e in v["events"]})) or "—",
                                      "Repetições": v["summary"].get("repeticoes"), "Fonte": v.get("source")}
                                     for v in results["video"]]), ""]
    for v in results["video"]:
        if v.get("plot_path"):
            rel = Path(v["plot_path"]).relative_to(out_dir) if Path(v["plot_path"]).is_absolute() else v["plot_path"]
            L += [f"![{v['file']}]({rel})", ""]
    L += ["> Ferramenta de apoio à decisão clínica. Dados de pacientes são sintéticos; os achados não substituem avaliação profissional."]
    p = out_dir / "relatorio_resultados.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p
