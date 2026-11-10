# Roteiro do vídeo — MedMonitor (meta: 12 min · limite: 15 min)

Cobre os quatro itens exigidos: ✅ análise de áudio e vídeo · ✅ detecção e resposta a anomalias · ✅ integração com o Azure · ✅ alerta final à equipe médica (aplicativo + Telegram).

As falas estão em português simples, com frases curtas. Os únicos nomes estrangeiros que ficaram são **nomes de produtos**, que não têm tradução. Veja como pronunciar:

| Nome | Como falar |
|---|---|
| Azure | "ájur" |
| YOLO | "iôlo" |
| OpenPose | "ôpen pôuz" |
| Telegram | "telegrâm" |
| Teams | "tíms" |
| PhysioNet | "fízio-nét" |
| Google Colab | "gúgol coláb" |
| NEWS2 | "níus dois" |

---

## Antes de gravar

```bash
cd ~/Desktop/fiap/tech-challenge-fase4
source .venv/bin/activate
python main.py azure-check      # confirme que os serviços do Azure respondem (chaves no .env)
python main.py telegram-test    # confirme que o alerta chega no celular
python main.py demo --quiet     # gera todos os resultados (envia ~15 alertas críticos ao Telegram)
python main.py gui              # deixe o aplicativo aberto
```

- Abra o **Telegram Desktop** ou o **web.telegram.org** ao lado do aplicativo, na conversa com o @Sftwa_bot.
- **Limpe a conversa do bot** depois de rodar o `demo`, para que na Cena 5 só apareça o alerta ao vivo.
- Deixe abertos também: um terminal com letra grande, o portal do Azure e a página do projeto no GitHub.
- Grave em 1080p e ensaie uma vez cronometrando.

---

## Cena 1 — Abertura · 0:00 a 1:00

**Tela:** página do projeto no GitHub, com o desenho do fluxo.

**Fala:**
> Olá, eu sou o Lucas, e este é o MedMonitor, meu projeto da Fase 4 da pós-graduação em Inteligência Artificial da FIAP.
>
> O desafio é acompanhar pacientes internados o tempo todo, usando vários tipos de informação: vídeo, áudio, texto e sinais vitais. O objetivo é perceber sinais de risco cedo e avisar a equipe médica a tempo.
>
> A ideia principal é simples: cada tipo de informação é analisado pela ferramenta mais adequada. O vídeo é analisado com o YOLO e o OpenPose. O áudio usa os serviços de fala e de texto do Azure, a nuvem da Microsoft. E os sinais vitais, as receitas de remédios e a movimentação no leito passam por detectores de anomalias.
>
> No final, o sistema cruza tudo isso no tempo e avisa a equipe pelo aplicativo e pelo Telegram, direto no celular do plantão.

## Cena 2 — Ligação com o Azure e com o Telegram · 1:00 a 2:45

**Tela:** portal do Azure com os dois recursos; depois o terminal rodando `python main.py azure-check`.

**Fala:**
> Começando pela nuvem. Aqui no portal estão os dois serviços do Azure que eu uso: um de fala e um de linguagem. O projeto tem um programa que cria esses serviços e guarda as chaves de acesso sozinho.
>
> Agora eu vou testar a ligação. O teste cria uma frase falada, como se fosse um paciente, e manda para o Azure. O Azure devolve a frase escrita, em português, com o momento exato de cada palavra. Com isso dá para medir a velocidade da fala e as pausas.
>
> Depois, o texto vai para o serviço de linguagem. Ele diz se a fala é positiva ou negativa, mostra as frases mais importantes e reconhece termos de saúde. Por exemplo: "falta de ar" é reconhecido como um sintoma. E ele entende a negação: se o paciente diz "não tenho falta de ar", isso não vira alerta.

**Tela:** terminal rodando `python main.py telegram-test`; a mensagem aparece no Telegram ao lado.

> A outra ponta é a entrega do aviso. Eu criei um robô no Telegram. Com este comando, ele manda um alerta de teste… e pronto, chegou. Tem a gravidade, o paciente, as provas e o que a equipe deve fazer. Daqui a pouco vamos ver isso acontecer ao vivo.

## Cena 3 — Análise de áudio · 2:45 a 4:30

**Tela:** aplicativo → aba **Áudio** → `P003_consulta_dispneia.wav` → ▶ Ouvir (uns 8 segundos).

**Fala:**
> Agora, a análise do áudio das consultas. Esta é a paciente P003. Prestem atenção na respiração entre as frases.
>
> O sistema faz duas análises. A primeira é sobre **o que** a paciente diz: o texto escrito pelo Azure, o tom negativo da conversa e as palavras de alerta, "falta de ar" e "cansaço para respirar".
>
> A segunda é sobre **como** ela fala. O sistema mede o ritmo, as pausas, o tom da voz, o tremor e a respiração que dá para ouvir. Com isso, ele calcula três notas, de zero a um: cansaço, dificuldade para respirar e fala enrolada. Aqui, a dificuldade para respirar deu zero vírgula oitenta e dois, bem acima do limite.
>
> Como a voz e o que ela disse concordam, o alerta sobe para crítico.

**Tela:** clique rápido em `P001` e depois em `P004`.

> Para comparar: o paciente P001 está bem. As três notas estão baixas, e o "falta de ar" aparece riscado, porque ele disse que **não** tem.
>
> Já o paciente P004 tem a fala enrolada, com tremor na voz e fala lenta. E ele conta que sente formigamento. Guardem esse paciente, porque ele volta daqui a pouco.

## Cena 4 — Análise de vídeo · 4:30 a 6:45

**Tela:** aba **Vídeo** → `P001_agachamento.mp4` → ▶ Reproduzir.

**Fala:**
> Agora, o vídeo. O exemplo é uma sessão de fisioterapia. O YOLO encontra dezessete pontos do corpo em cada imagem: ombros, quadril, joelhos e assim por diante. Com esses pontos, o sistema calcula o ângulo das articulações e o quanto o tronco está inclinado.
>
> Em cima do vídeo aparecem o ângulo do joelho e a contagem de repetições. No gráfico, cada vale é uma repetição. Foram seis.
>
> Aos nove segundos, o sistema percebeu que o tronco inclinou demais: cinquenta e seis graus, quando o limite é quarenta e cinco. Tudo isso vira um relatório automático.

**Tela:** `docs/img/comparacao_yolo_openpose.jpg`.

> O trabalho pede o OpenPose, e ele também está no projeto. Aqui estão os dois na mesma imagem. Eles contam as mesmas repetições, mas o OpenPose é umas dezessete vezes mais lento num computador comum. Por isso, o padrão é o YOLO.

**Tela:** `P005_queda_simulada.mp4` → ▶ Reproduzir até a faixa vermelha.

> O caso mais importante é a queda. Este vídeo é uma simulação: a imagem foi girada de propósito, só para testar o detector. O sistema percebe que o quadril desceu rápido demais e que o corpo ficou deitado. Na hora, ele gera um alerta crítico de queda.

**Tela:** `zona_critica.mp4` por 5 segundos.

> O YOLO também vigia áreas restritas, como a área estéril de uma cirurgia. Aqui, o limite é de duas pessoas. Quando entram seis, o sistema avisa.

## Cena 5 — Anomalias ao vivo e o aviso no Telegram · 6:45 a 9:15

**Tela:** aba **Sinais vitais** → paciente P003 → sub-aba **Tempo real** → início 14 h → 60 amostras/s. O Telegram fica visível ao lado. ▶ Iniciar.

**Fala:**
> Agora, a detecção de anomalias nos sinais vitais. Aqui o sistema usa três detectores ao mesmo tempo.
>
> O primeiro é o NEWS2, uma pontuação que os hospitais já usam para medir a gravidade do paciente. O segundo compara o paciente com ele mesmo, com o normal dele. E o terceiro é um modelo de aprendizado de máquina que procura combinações estranhas entre os sinais.
>
> Um problema só é confirmado quando dois dos três concordam por pelo menos três minutos. Assim, se o aparelho de oxigênio escorrega do dedo por um instante, não vira alarme falso.
>
> Aqui eu estou passando os dados da paciente P003, um minuto de cada vez, como chegariam do monitor do leito. Vejam o oxigênio e a respiração caindo… começam a aparecer os avisos: oxigênio baixo, respiração lenta… e quando fica grave, a faixa vermelha acende, com som e aviso no computador.

**Tela:** aponte para o Telegram quando a mensagem 🔴 chegar.

> E no mesmo momento, a mensagem chega no Telegram. É assim que a equipe fica sabendo, mesmo longe do computador.
>
> Só os alertas graves vão para o celular, para não virar barulho. E se o mesmo alerta se repetir em menos de trinta minutos, ele não é enviado de novo.

**Tela:** aba **Prescrições**, filtro em P003; depois aba **Movimentação**, paciente P005.

> E o que causou isso? Olhando as receitas: a morfina dessa paciente foi aumentada cinco vezes, às nove da noite. Meia hora depois, ela recebeu outro calmante que, junto com a morfina, pode diminuir a respiração.
>
> Na movimentação no leito, a paciente P005 ficou cinco horas e meia sem mudar de posição, o que pode causar ferida na pele. E às três da manhã houve um movimento muito forte, seguido de silêncio total. Isso indica uma possível queda da cama.

## Cena 6 — Juntando tudo: o caminho do alerta · 9:15 a 11:30

**Tela:** aba **Central de alertas**.

**Fala:**
> Tudo isso chega aqui, na central de alertas. Em cima, os números principais. Na tabela, o risco de cada paciente. O paciente P001, que está bem, ficou com risco baixo. Os outros quatro, que estão em situação grave, ficaram com risco alto.

**Tela:** selecione o alerta de fusão "Suspeita de depressão respiratória induzida por opioide".

> O grande diferencial são estes alertas em azul. Eles juntam informações de lugares diferentes.
>
> Este aqui juntou o aumento da morfina, às nove da noite, com a queda da respiração, às dez e dezessete. E concluiu: a respiração caiu provavelmente por causa da morfina. Ele mostra as provas e sugere o que fazer: diminuir a morfina e considerar o remédio que corta o efeito dela.

**Tela:** selecione "Suspeita de AVC cardioembólico".

> E este é o meu exemplo preferido. Lembram do paciente P004? Nos sinais vitais, o coração dele disparou algumas vezes. E na consulta gravada, ele estava com a fala enrolada.
>
> Separados, nenhum dos dois é grave. Mas juntos, o sistema levanta a suspeita de um AVC e recomenda chamar a equipe de AVC. É exatamente isso que a união das informações resolve.

**Tela:** Telegram, rolando as mensagens do `demo` (role a conversa ou mostre uma captura feita antes de limpar).

> Fechando o caminho do alerta: cada aviso passa por uma central que define a gravidade, segura as repetições e entrega em três lugares: aqui no aplicativo, num registro guardado para consulta e no Telegram. Estes são os alertas graves do processamento completo, cada um com o paciente, as provas e o que fazer.
>
> Uma observação importante: eu usei o Telegram só para a demonstração. Num hospital de verdade, com dados reais de pacientes, o aviso iria para um canal da própria instituição, como o Teams do hospital, por causa da lei de proteção de dados. E para trocar, não precisa mudar o programa, só a configuração.

## Cena 7 — Resultados · 11:30 a 12:45

**Tela:** aba **Métricas**; depois o terminal com `python main.py physionet --record 14`.

**Fala:**
> Como os pacientes são de teste, eu sei exatamente onde estão os problemas. Então dá para medir o acerto.
>
> Nos sinais vitais, os três detectores juntos encontraram todos os problemas, com oitenta por cento de acerto nos alertas. Isso é melhor do que qualquer um deles sozinho.
>
> Ao vivo, o sistema encontrou dez de dez problemas. Os mais urgentes foram avisados em dois a seis minutos. Nas receitas, acertou sete de sete, sem nenhum alarme falso. E no áudio, as quatro consultas foram classificadas corretamente.
>
> Eu também testei com dados reais do PhysioNet, de pacientes de terapia intensiva. Em quarenta e nove dos cinquenta e três pacientes, não houve alerta. Nos outros, ele encontrou problemas reais, como esta respiração acelerada.

## Cena 8 — Encerramento · 12:45 a 13:30

**Tela:** página do projeto no GitHub.

**Fala:**
> Para fechar, os limites do projeto. Os pacientes são de teste. A análise da voz precisa ser ajustada com gravações reais. E o sistema é uma ferramenta de apoio: a decisão é sempre da equipe médica.
>
> Como próximos passos: receber os sinais direto dos monitores pela nuvem, treinar o YOLO para reconhecer instrumentos de cirurgia e pedir confirmação de leitura no Telegram. Assim, se ninguém responder, o aviso passa para o próximo plantonista.
>
> O código, o relatório e um caderno para o Google Colab estão no GitHub, no link da descrição. Obrigado!

---

## Se o tempo apertar

Corte nesta ordem: o vídeo da área restrita (Cena 4), o PhysioNet (Cena 7) e a rolagem das mensagens no Telegram (Cena 6). **Não corte** o alerta ao vivo no Telegram (Cena 5): ele mostra o "alerta final à equipe médica".

## Se algo falhar durante a gravação

- **A mensagem não chegou no Telegram:** confira a internet e rode `python main.py telegram-test`. A faixa vermelha no aplicativo continua valendo como alerta.
- **O alerta ao vivo demorou:** a mensagem pode levar um ou dois segundos. Continue falando e aponte quando ela aparecer.

Publique no YouTube como **não listado** e coloque o link na página do projeto.
