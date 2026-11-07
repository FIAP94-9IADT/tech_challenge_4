# Roteiro do vídeo — MedMonitor (meta: 12 min · limite: 15 min)

Cobre os quatro itens exigidos: ✅ análise de áudio e vídeo · ✅ detecção e resposta a anomalias · ✅ integração dos serviços Azure · ✅ fluxo final do alerta à equipe médica (aplicativo + **Telegram**).

As falas somam cerca de 1.600 palavras. Num ritmo calmo dá **~12 minutos**, e sobram 3 minutos de folga para cliques e carregamentos.

## Antes de gravar

```bash
cd ~/Desktop/fiap/tech-challenge-fase4
source .venv/bin/activate
python main.py azure-check      # confirme que Speech e Language respondem (chaves no .env)
python main.py telegram-test    # confirme que o alerta chega no celular
python main.py demo --quiet     # gera todos os resultados (envia ~15 alertas críticos ao Telegram)
python main.py gui              # deixe o aplicativo aberto
```

- **Telegram na tela:** abra o **Telegram Desktop** ou o **web.telegram.org** ao lado do aplicativo, na conversa com o @Sftwa_bot. É mais legível que filmar o celular. Se preferir o celular, espelhe a tela ou filme por cima.
- **Limpe a conversa do bot** depois de rodar o `demo`, para que, na Cena 5, só o alerta ao vivo apareça ali.
- Deixe abertos também: um terminal com fonte grande, o portal do Azure e o README no GitHub.
- Grave em 1080p e ensaie uma vez cronometrando.

---

## Cena 1 — Abertura · 0:00 a 1:00

**Tela:** README no GitHub, com o diagrama do fluxo.

**Fala:**
> Olá, eu sou o Lucas e este é o MedMonitor, meu projeto da Fase 4 da Pós-Tech em IA da FIAP.
>
> O desafio é monitorar pacientes internados continuamente usando dados de naturezas diferentes: vídeo, áudio, texto e sinais vitais, para identificar sinais precoces de risco e avisar a equipe médica a tempo.
>
> A ideia central é esta: cada tipo de dado é analisado pelo modelo mais adequado a ele. Vídeo com YOLOv8 e OpenPose, áudio com o Azure Speech e o Azure Text Analytics, e sinais vitais, prescrições e movimentação com técnicas de detecção de anomalias. Uma camada de fusão cruza os eventos de todas as fontes no tempo, e os alertas chegam à equipe pelo aplicativo e pelo Telegram, direto no celular do plantão.

## Cena 2 — Integrações: Azure e Telegram · 1:00 a 2:45

**Tela:** portal do Azure com os dois recursos; depois o terminal rodando `python main.py azure-check`.

**Fala:**
> Começando pela nuvem. Aqui no portal estão os dois serviços que eu uso: o Azure AI Speech e o Azure AI Language. O projeto tem um script que cria esses recursos e grava as chaves automaticamente.
>
> Vou rodar o teste de integração. Ele sintetiza uma frase de paciente, manda para o Speech to Text e recebe a transcrição em português, com o horário de cada palavra. Esses horários permitem medir velocidade de fala e pausas.
>
> Depois o texto vai para o Text Analytics, que devolve o sentimento, as frases-chave e, com o Text Analytics for Health, as entidades clínicas, como "falta de ar" classificada como sintoma. Ele também detecta negação: se o paciente diz "não tenho falta de ar", isso não vira alerta.

**Tela:** terminal rodando `python main.py telegram-test`; a mensagem aparece no Telegram ao lado.

> A outra ponta é a entrega do alerta. Criei um bot no Telegram, o MedMonitor bot. Com este comando ele envia um alerta de teste… e aqui está: severidade, paciente, evidências e a conduta sugerida, no celular da equipe. Daqui a pouco vamos ver isso acontecendo ao vivo.

## Cena 3 — Análise de áudio · 2:45 a 4:30

**Tela:** aplicativo → aba **Áudio** → `P003_consulta_dispneia.wav` → ▶ Ouvir (uns 8 segundos).

**Fala:**
> Agora a análise de áudio de consultas. Esta é a paciente P003. Repare nas respirações entre as frases.
>
> O sistema faz duas análises. A primeira é o que o paciente diz: a transcrição do Azure, o sentimento negativo e os termos críticos, "falta de ar" e "cansaço para respirar".
>
> A segunda é como o paciente fala. Eu extraio biomarcadores vocais do sinal: ritmo, pausas, frequência da voz, tremor e inspirações audíveis. Eles viram três indicadores: fadiga, dificuldade respiratória e disartria. Aqui a dificuldade respiratória deu 0,82, bem acima do limiar. Como a voz e o relato concordam, o alerta sobe para crítico.

**Tela:** clique rápido em `P001` e depois em `P004`.

> Para comparar: o P001 é estável, com todos os indicadores baixos, e o "falta de ar" aparece riscado porque foi negado. Já o P004 tem padrão de disartria, com tremor na voz e fala lenta, e relata fala enrolada e formigamento. Guardem esse paciente, ele volta daqui a pouco.

## Cena 4 — Análise de vídeo · 4:30 a 6:45

**Tela:** aba **Vídeo** → `P001_agachamento.mp4` → ▶ Reproduzir.

**Fala:**
> Na análise de vídeo, o exemplo é uma sessão de fisioterapia. O YOLOv8-pose detecta 17 pontos do corpo em cada quadro, e a partir deles eu calculo os ângulos das articulações e a inclinação do tronco.
>
> No topo aparece o ângulo do joelho e o contador de repetições. No gráfico, cada vale é uma repetição: foram 6, com amplitude média de 107 graus. Aos 9 segundos o sistema marcou uma compensação de tronco, uma inclinação de 56 graus quando o limite é 45. Tudo isso sai num relatório automático.

**Tela:** `docs/img/comparacao_yolo_openpose.jpg`.

> O enunciado pede OpenPose, e ele está implementado como segunda opção. Aqui estão os dois no mesmo quadro: encontram as mesmas repetições, mas o OpenPose é cerca de 17 vezes mais lento em CPU. Por isso o padrão é o YOLOv8.

**Tela:** `P005_queda_simulada.mp4` → ▶ Reproduzir até a faixa vermelha.

> O caso mais importante é a queda. Este vídeo é uma simulação, feita por rotação da imagem, para validar o detector. O sistema percebe que o quadril desce rápido demais e que o corpo fica na horizontal, e gera um alerta crítico.

**Tela:** `zona_critica.mp4` por 5 segundos.

> O YOLOv8 também monitora áreas críticas, como um campo estéril: aqui a área permite duas pessoas e o sistema alerta quando entram seis.

## Cena 5 — Anomalias em tempo real + alerta no Telegram · 6:45 a 9:15

**Tela:** aba **Sinais vitais** → paciente P003 → sub-aba **Tempo real** → início 14 h → 60 amostras/s. O Telegram fica visível ao lado. ▶ Iniciar.

**Fala:**
> Agora a detecção de anomalias. Os sinais vitais usam três detectores juntos: o escore clínico NEWS2, um z-score robusto que compara o paciente com ele mesmo, e um Isolation Forest que procura combinações incomuns entre os sinais. A anomalia só é confirmada quando dois dos três concordam por pelo menos três minutos, o que elimina, por exemplo, um oxímetro que escorregou do dedo.
>
> Aqui estou reproduzindo os dados do P003 uma amostra por vez, como chegariam do monitor do leito. Olhem a saturação e a frequência respiratória caindo… os alertas de atenção aparecendo: dessaturação, bradipneia… e quando o quadro fica crítico, a faixa vermelha acende, com som e notificação no computador.

**Tela:** aponte para o Telegram quando a mensagem 🔴 chegar.

> E no mesmo instante a mensagem chega no Telegram. É assim que a equipe é avisada mesmo longe do computador. Só os alertas críticos vão para o celular, para não virar ruído, e repetições do mesmo alerta em menos de 30 minutos são suprimidas.

**Tela:** aba **Prescrições**, filtro em P003; depois aba **Movimentação**, paciente P005.

> E o que causou isso? Nas prescrições, a morfina dessa paciente foi multiplicada por cinco às 21 horas, e meia hora depois entrou o midazolam, que interage com ela.
>
> Na movimentação no leito, a P005 ficou cinco horas e meia sem ser reposicionada, risco de lesão por pressão, e às 3 da manhã teve um pico de movimento seguido de silêncio total: uma possível queda.

## Cena 6 — Fusão multimodal e o fluxo do alerta · 9:15 a 11:30

**Tela:** aba **Central de alertas**.

**Fala:**
> Tudo isso chega aqui, na central de alertas. Em cima estão os indicadores; na tabela, o risco de cada paciente. O P001, nosso controle, ficou baixo, e os outros quatro, com cenários graves, ficaram altos.

**Tela:** selecione o alerta de fusão "Suspeita de depressão respiratória induzida por opioide".

> O diferencial são os alertas de fusão, em azul. Este juntou a prescrição de morfina das 21h com a queda respiratória das 22h17 e concluiu: suspeita de depressão respiratória causada pelo opioide. Ele mostra as evidências e sugere a conduta: reduzir o opioide e considerar naloxona.

**Tela:** selecione "Suspeita de AVC cardioembólico".

> E este é o meu exemplo favorito. O P004 teve taquicardia nos sinais vitais e, na consulta gravada, fala disártrica. Separados, nenhum dos dois é crítico. Juntos, o sistema levanta a suspeita de AVC e recomenda acionar o protocolo. É exatamente o problema que a fusão multimodal resolve.

**Tela:** Telegram, rolando as mensagens do `demo` (role a conversa ou mostre uma captura feita antes de limpar).

> Fechando o fluxo: cada alerta passa por um gerenciador que define a severidade, suprime repetições e entrega nos canais: aqui no aplicativo, num log para auditoria e no Telegram. Estes são os alertas críticos do processamento completo, incluindo os de fusão, cada um com paciente, evidência e conduta.
>
> Uma observação: usei o Telegram para a demonstração. Num hospital, com dados reais, o mesmo mecanismo enviaria para um canal institucional, como o Teams do hospital, por causa da LGPD, sem mudar o código, só a configuração.

## Cena 7 — Resultados · 11:30 a 12:45

**Tela:** aba **Métricas**; depois o terminal com `python main.py physionet --record 14`.

**Fala:**
> Como os pacientes são sintéticos, eu sei exatamente onde estão as anomalias, então dá para medir. Nos sinais vitais, o conjunto dos três detectores encontrou todos os eventos, com 80% de precisão, melhor do que qualquer detector sozinho. No tempo real, foram 10 de 10 eventos detectados, os agudos em 2 a 6 minutos. Nas prescrições, 7 de 7 sem nenhum falso alarme, e no áudio as quatro consultas foram classificadas corretamente.
>
> Também testei com dados reais do PhysioNet, de pacientes de UTI: em 49 dos 53 registros não houve alerta, e nos outros ele pegou eventos reais, como esta taquipneia.

## Cena 8 — Encerramento · 12:45 a 13:30

**Tela:** README.

**Fala:**
> Para fechar, as limitações: os pacientes são sintéticos, os limiares de voz precisam ser recalibrados com gravações reais, e é uma ferramenta de apoio: a decisão é sempre da equipe médica.
>
> Como próximos passos: receber os sinais em streaming pelo Azure Event Hubs, treinar o YOLO com instrumental cirúrgico e adicionar confirmação de leitura nos alertas do Telegram, para escalar ao próximo plantonista se ninguém responder.
>
> O código, o relatório técnico e um notebook para o Google Colab estão no repositório, no link da descrição. Obrigado!

---

## Se o tempo apertar

Corte nesta ordem: o vídeo da área crítica (Cena 4), o PhysioNet (Cena 7) e a rolagem das mensagens no Telegram (Cena 6). **Não corte** o alerta ao vivo no Telegram (Cena 5): ele mostra o "fluxo final do alerta à equipe médica".

## Se algo falhar durante a gravação

- **A mensagem não chegou no Telegram:** confira a internet e rode `python main.py telegram-test`. A faixa vermelha no aplicativo continua valendo como alerta.
- **O alerta ao vivo demorou:** a mensagem pode levar 1 a 2 segundos. Continue falando e aponte quando ela aparecer.

Publique no YouTube como **não listado** e coloque o link no README.
