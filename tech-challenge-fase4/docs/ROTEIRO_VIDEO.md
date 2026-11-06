# Roteiro do vídeo de demonstração (≤ 15 min)

Requisitos do enunciado cobertos: ✅ análise de áudio e vídeo · ✅ detecção e resposta a anomalias · ✅ integração dos serviços Azure · ✅ fluxo final do alerta à equipe médica.

## Preparação (antes de gravar)

```bash
source .venv/bin/activate
bash infra/azure_setup.sh          # ou preencha o .env manualmente
python main.py azure-check         # confirme que os dois serviços respondem
python main.py demo --quiet        # deixa todos os resultados prontos
```

- Configure `ALERT_WEBHOOK_URL` com um fluxo *Workflows* do Teams num canal de teste, para mostrar o alerta chegando.
- Abra: o terminal, o aplicativo (`python main.py gui`), o portal do Azure (grupo `rg-medmonitor`), o canal do Teams e o `docs/RELATORIO_TECNICO.md` no GitHub.
- Grave em 1080p com o zoom do terminal ampliado.

## Roteiro

| Tempo | Cena | O que mostrar / falar |
|---|---|---|
| **0:00–1:00** | Abertura | Problema: monitorar continuamente pacientes com dados de naturezas diferentes. Mostrar o diagrama do README. "Cada modalidade tem seu modelo especializado e a fusão cruza os eventos no tempo." |
| **1:00–2:30** | Integração Azure | Portal do Azure: recursos Speech e Language. Terminal: `python main.py azure-check` — a frase de teste é sintetizada, transcrita (mostrar timestamps por palavra) e analisada (sentimento, frases-chave, entidades de saúde). |
| **2:30–4:30** | **Áudio** | Aplicativo → aba Áudio → ▶ Ouvir → `P003_consulta_dispneia.wav`. Tocar alguns segundos (inspirações entre as frases). Mostrar: transcrição do Azure Speech, sentimento negativo, termos críticos, indicador de dispneia 0,82 e as componentes (inspirações audíveis, fonação curta). Comparar rapidamente com `P001` (normal, "falta de ar" **negado**) e `P004` (disartria, tremor vocal). |
| **4:30–7:00** | **Vídeo** | Aba Vídeo → `P001_agachamento.mp4` → ▶ Reproduzir: vídeo anotado com esqueleto, ângulo do joelho e contador; gráfico com 6 repetições e a compensação de tronco aos 9,4 s; relatório automático. Depois `P005_queda_simulada.mp4`: a faixa "CRITICAL: Queda detectada" (explicar que é uma simulação). Citar o OpenPose: mostrar `docs/img/comparacao_yolo_openpose.jpg` e o comando `--openpose` (mesmas repetições, ~17× mais lento — por isso o YOLOv8-pose é o padrão). Mostrar também o botão **Abrir vídeo…** (perfil, pessoa, áreas críticas, OpenPose). Opcional: `zona_critica.mp4` (lotação na área restrita). |
| **7:00–9:30** | **Anomalias em tempo real** | Terminal: `python main.py stream --patient P003 --from-hour 12 --speed 40`. Os vitais rolam; às ~22:15 os alertas de bradipneia/dessaturação surgem ao vivo. No aplicativo, aba Sinais vitais → P003 → sub-aba **Tempo real** → início 14 h → ▶ Iniciar: os valores mudam, os alertas aparecem na lista e o alerta crítico acende a faixa vermelha com notificação do sistema. Mostrar abas Prescrições (morfina 5×, interação com midazolam) e Movimentação (P005: imobilidade e queda). |
| **9:30–12:00** | **Fusão e alerta à equipe** | Aplicativo → Central de alertas: KPIs, tabela de risco (P001 baixo × demais alto), feed filtrado em CRÍTICO. Abrir os alertas de fusão: "depressão respiratória induzida por opioide" (prescrição → vitais) e "AVC cardioembólico" (arritmia + disartria) — explicar que **nenhuma fonte isolada gera esse alerta**. Mostrar a evidência e a conduta sugerida. Trocar para o Teams: o Adaptive Card chegou no canal. Mostrar `outputs/alerts.jsonl` como trilha de auditoria. |
| **12:00–14:00** | Resultados | Aba Métricas ou relatório: ensemble de vitais com recall 1,0 e precisão 0,8 (melhor F1 que cada método isolado); streaming detecta 10/10 eventos em 2–6 min; prescrições 7/7 sem falso positivo; áudio 4/4. Dados reais do PhysioNet: `python main.py physionet --record 14` (taquipneia real). |
| **14:00–15:00** | Encerramento | Limitações (dados sintéticos, apoio à decisão, LGPD) e próximos passos (Event Hubs, pesos cirúrgicos, recalibração com áudio real). Link do repositório. |

## Dicas

- Se o tempo apertar, corte o vídeo de área crítica e o PhysioNet; os quatro itens obrigatórios estão nas cenas de 1:00 a 12:00.
- Sem credenciais Azure, tudo funciona em modo local, mas **a cena de integração Azure é obrigatória** — grave-a com o `.env` preenchido.
- Publique como **não listado** no YouTube e coloque o link no README.
