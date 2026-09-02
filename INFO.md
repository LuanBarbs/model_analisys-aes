# INFO - Reprodução do baseline e troca para XLM-RoBERTa

Notas de decisão do trabalho de adaptação do EPITOME (empatia em texto) para PT,
em resposta ao apontamento do revisor sobre o uso de `roberta-base`.

---

## Contexto

- O revisor apontou que o `RoBERTaBASE` é um modelo **monolíngue inglês**, mas o
  diferencial do paper é o texto em **português**.
- Correção recomendada: usar `FacebookAI/xlm-roberta-base` (multilíngue, inclui PT).
- Os dados de treino (`emotional-reactions`, `interpretations`, `explorations`)
  são **traduções para PT** do dataset original do Sharma (nomes de arquivo mantidos).
- Objetivo do baseline: **reproduzir a saída "antes" (com roberta-base)** para
  contrastar com o "depois" (xlm-roberta). Não é obter um modelo bom.

## Descoberta principal: o problema não é só tokenização

- No `models.py`, o RoBERTa é o **encoder** (espinha dorsal do classificador),
  não apenas o tokenizador. A hipótese inicial de "trocar só na tokenização" é falsa.
- A troca para xlm-roberta exige: trocar tokenizador **e** encoder, **reprocessar
  os dados** e **retreinar** ER/IP/EX. Não é "trocar uma string".

## Diagnóstico crítico: o pré-processamento apaga rationales em PT

- O `process_data.py` tokeniza texto PT com o BPE inglês do `roberta-base`.
  A linha `if curr_word.startswith('Ġ')` e o `re.search(...)` de alinhamento
  falham com acento/cedilha, caindo no `except: continue` — o rationale é
  **descartado em silêncio**.
- Crosstab de `er_model.csv` (rationale vazio por nível) confirma o dano:

  | nível | rationale presente | rationale vazio |
  |-------|--------------------|-----------------|
  | 0     | 2036               | 0               |
  | 1     | 291                | 604 (67%)       |
  | 2     | 66                 | 86 (57%)        |

- O padrão está **invertido**: os níveis empáticos (1 e 2), que deveriam ter
  rationale, estão majoritariamente vazios. O sinal foi corrompido na origem.
- **Isto reforça o ponto do revisor**: o `roberta-base` contamina o pipeline
  desde o pré-processamento, não só na inferência. Se a taxa de vazios cair após
  a troca para xlm-roberta, é evidência quantitativa de que a troca importou.

## Decisões tomadas

1. **Ambiente fiel ao original** (Python 3.8, torch 1.4.0, transformers 2.7.0).
   Fidelidade > modernidade para o baseline.
   - Ajuste necessário: `pip install "scipy==1.4.1"` (compatível com numpy 1.18.2).
2. **Treinar na CPU.** A RTX 4090 (sm_89) não roda com torch 1.4.0 (kernels só até
   sm_75). `torch.cuda.is_available()` retorna `True` mas trava/falha no uso real.
   Solução: `export CUDA_VISIBLE_DEVICES=""` antes de treinar.
3. **Confiar no `process_data.py` atual.** É o único que existe; o responsável
   original saiu do projeto e não é contatável. Assumimos equivalência com a
   rodada que gerou os números publicados — **decisão a documentar formalmente**.
4. **Tratar o baseline como "antes com defeito"**, não como baseline limpo. Os
   números de rationale refletem o dano do tokenizador, não qualidade de modelo.
   Ninguém no projeto deve tratá-los como medida de desempenho.

## Fluxo real do pipeline (confirmado)

1. `process_data.py`: CSV cru (PT) → formato do modelo (alinha rationales a tokens).
2. `train.py`: treina ER / IP / EX separadamente no Reddit-PT → salva `.pth`.
3. `test.py`: aplica os 3 modelos aos posts PT (`avalia_modelos.csv`, respostas
   Gemma/Llama) → gera rótulos ER/IP/EX. **Esta é a "saída do artigo" a reproduzir.**

Observação: o README pula o `process_data.py` nos comandos do "full Reddit dataset"
(os CSVs crus não têm a coluna `rationale_labels`) — é preciso processar antes.

## Comandos do baseline

```bash
export CUDA_VISIBLE_DEVICES=""

# 1. pré-processar
python3 src/process_data.py --input_path dataset/emotional-reactions-reddit.csv --output_path dataset/er_model.csv
python3 src/process_data.py --input_path dataset/interpretations-reddit.csv   --output_path dataset/ip_model.csv
python3 src/process_data.py --input_path dataset/explorations-reddit.csv      --output_path dataset/ex_model.csv

# 2. treinar (CPU, ~30 min/mecanismo a ~8.2 s/it)
python3 src/train.py --train_path dataset/er_model.csv --lr 2e-5 --batch_size 32 --lambda_EI 1.0 --lambda_RE 0.5 --save_model --save_model_path output/reddit_ER.pth
python3 src/train.py --train_path dataset/ip_model.csv --lr 2e-5 --batch_size 32 --lambda_EI 1.0 --lambda_RE 0.5 --save_model --save_model_path output/reddit_IP.pth
python3 src/train.py --train_path dataset/ex_model.csv --lr 2e-5 --batch_size 32 --lambda_EI 1.0 --lambda_RE 0.5 --save_model --save_model_path output/reddit_EX.pth

# 3. aplicar aos posts PT
python3 src/test.py --input_path dataset/avalia_modelos.csv --output_path dataset/output/avalia_saida.csv \
  --ER_model_path output/reddit_ER.pth --IP_model_path output/reddit_IP.pth --EX_model_path output/reddit_EX.pth
```

## Próximos passos (troca para XLM-RoBERTa)

A troca **começa pelo `process_data.py`**, porque é onde o dano nasce:

1. `process_data.py`: `RobertaTokenizer` → `XLMRobertaTokenizer`; marcador de
   início-de-palavra `Ġ` (BPE) → `▁` (SentencePiece); remover `do_lower_case=True`
   (XLM-R é cased). **Reprocessar os 3 CSVs** (o `.csv` processado é específico do
   tokenizador).
2. `empathy_classifier.py` e `train.py`: trocar o tokenizador.
3. `models.py`: `from_pretrained("roberta-base")` → `"xlm-roberta-base"` (2 lugares:
   seeker e responder encoders).
4. **Retreinar** ER/IP/EX e **reaplicar** aos posts PT.
5. Comparar taxa de rationale vazio antes/depois como evidência da correção.

---

# Parte 2 — Resultados do baseline e adaptação para XLM-RoBERTa

## Resultados do baseline (roberta-base) vs. tabela do artigo

Aplicação dos 3 modelos treinados aos 20 posts PT por gerador (LLAMA, GEMMA, GPT4).
Contagens de rótulo 0/1/2 por mecanismo.

| Model | ER-0 | ER-1 | ER-2 | IP-0 | IP-1 | IP-2 | EX-0 | EX-1 | EX-2 |
|-------|------|------|------|------|------|------|------|------|------|
| Llama (artigo) | 17 | 3 | 0 | 20 | 0 | 0 | 7  | 0 | 13 |
| Llama (repro)  | 18 | 2 | 0 | 20 | 0 | 0 | 7  | 0 | 13 |
| Gemma (artigo) | 17 | 3 | 0 | 20 | 0 | 0 | 7  | 0 | 13 |
| Gemma (repro)  | 18 | 2 | 0 | 20 | 0 | 0 | 8  | 0 | 12 |
| GPT (artigo)   | 19 | 1 | 0 | 20 | 0 | 0 | 18 | 0 | 2  |
| GPT4 (repro)   | 19 | 1 | 0 | 20 | 0 | 0 | 20 | 0 | 0  |

- Diferenças pequenas (1–2 exemplos por célula), concentradas em ER e EX (fronteiras
  de decisão). IP idêntico (modelo prevê tudo 0, sem margem para variar).

## Dúvida da seed: a variação NÃO vem da avaliação

- **Inferência é 100% determinística**: `model.eval()`, sem dropout, `argmax` puro.
  Rodar `test.py` de novo sobre o mesmo `.pth` dá sempre o mesmo resultado.
- A variação vem do **treino**, mesmo com seeds fixas (`random`, `numpy`, `torch`,
  `torch.cuda`). Motivos: não-determinismo de ponto flutuante em CPU/threads
  (soma não-associativa, propaga por 4 épocas) + ambiente/versões diferentes dos
  que geraram o artigo + não se sabe qual seed a rodada original usou.
- **Conclusão**: a diferença é ruído de reprodução esperado, não bug de seed.

## Decisão sobre robustez (30 seeds) — PENDENTE com orientador

- Rodar N seeds **não reproduz** a tabela do artigo (que é rodada única). Produz
  uma tabela diferente (média ± dp).
- Faz sentido apenas se o objetivo virar **mostrar robustez** — aí é
  metodologicamente superior, mas é mudança de metodologia (aprovar com orientador).
- Custo em CPU é alto (~30–50 min/mecanismo × 3 × N seeds). 5–10 seeds costuma
  bastar; 30 tem ganho estatístico marginal a custo 3–6× maior.
- `train.py` já aceita `--seed_val`; dá para parametrizar e agregar em loop.

## Snippet para reproduzir a tabela (contagem de rótulos)

O `value_counts` sozinho omite classes ausentes. Usar `reindex` para garantir 0/1/2:

```python
import pandas as pd
df = pd.read_csv("../dataset/output/avalia_saida.csv")
df_l = df[df["id"] == "GPT4"]          # atenção: o CSV usa 'GPT4', não 'GPT'
df_l["EX_label"].value_counts().reindex([0, 1, 2], fill_value=0)
```

## Adaptação para XLM-RoBERTa — estratégia confirmada

### O encoder vendorizado NÃO serve para XLM-R
- `SeekerEncoder.from_pretrained('xlm-roberta-base')` falha com `OSError` na etapa
  de config: a `RobertaConfig` **vendorizada** só conhece os nomes do mapa hard-coded
  (`roberta-base`, etc.). O XLM-R nunca esteve nesse mapa.
- O `roberta-base` só funcionou no baseline por estar hard-coded nesse mapa.

### Solução: usar `XLMRobertaModel` da biblioteca transformers
- Teste confirmou: carrega limpo, vocab 250002, max_pos 514, hidden 768,
  forward devolve `last_hidden_state` em `[batch, seq, 768]` — mesma interface que o
  bi-encoder consome (`outputs[0]`). Troca é cirúrgica; resto da arquitetura
  (attn, norm, classifiers, forward) não muda.

### Alterações por arquivo (4 arquivos)

1. **`models.py`**: importar `XLMRobertaModel`; trocar os 2 encoders para
   `XLMRobertaModel.from_pretrained("xlm-roberta-base")`; no `forward`, trocar
   `self.*_encoder.roberta(...)` por `self.*_encoder(...)` (a lib não tem o
   atributo aninhado `.roberta`). Congelamento do seeker no `train.py` continua igual.
2. **`process_data.py`**: `RobertaTokenizer`→`XLMRobertaTokenizer`; remover
   `do_lower_case=True`; marcador `Ġ`→`▁` (U+2581, SentencePiece). **Arquivo de risco**:
   a aritmética de `response_words_position` foi escrita para BPE byte-level; trocar o
   marcador pode não bastar. Validar via crosstab.
3. **`train.py`**: `RobertaTokenizer`→`XLMRobertaTokenizer`, sem `do_lower_case`.
4. **`empathy_classifier.py`**: `RobertaTokenizer`→`XLMRobertaTokenizer`, sem
   `do_lower_case`.

### Evidência visual da correção (tokenização PT)
- XLM-R: `Não sei mais o que estou fazendo, você entende?` →
  `['▁Não','▁sei','▁mais','▁o','▁que','▁estou','▁fazendo',',','▁você','▁entende','?']`
  (palavras inteiras, acentos preservados).
- roberta-base estilhaçaria `Não`/`você` em subtokens byte-level — a causa raiz do
  `re.search` falhar e apagar rationales. Bom exemplo/figura para a resposta ao revisor.

### Ordem de execução da troca
1. Aplicar alterações nos 4 arquivos.
2. Reprocessar os 3 CSVs (salvar como `*_xlmr.csv` para preservar os do baseline).
3. **Validar com o crosstab de rationale vazio**: a inversão do baseline (67% vazio
   no nível 1, 57% no nível 2) deve cair drasticamente. Se cair → alinhamento
   consertado + evidência para o revisor. Se não cair → ajustar a aritmética do loop
   de posição em `process_data.py`.
4. Retreinar ER/IP/EX (CPU) com nomes `output/xlmr_*.pth`.
5. Reaplicar `test.py` → `avalia_saida_xlmr.csv`.
6. Comparar antes (roberta) vs depois (xlm-r): rótulos ER/IP/EX e taxa de vazios.

---

# Parte 3 — O problema do rationale e a decisão de `lambda_RE`

## Retificação importante do diagnóstico da Parte 1

A Parte 1 atribuiu os rationales vazios ao **tokenizador** (BPE inglês do
roberta-base quebrando texto PT). Isso estava **incompleto**. Investigação mais a
fundo (crosstab pós-XLM-R + inspeção dos dados crus) mostrou que a troca de
tokenizador `Ġ`→`▁` **não** consertou o alinhamento — logo, a causa raiz não era
tokenização. A causa real é o **dado-fonte**, e se divide em três casos distintos.

## Os três problemas do dataset herdado (independentes)

1. **Encoder/tokenizador errado** (roberta-base monolíngue). Resolvido: XLM-R.
   É o único dos três que afeta a saída reportada no paper (os rótulos ER/IP/EX).

2. **Alinhamento por tradução divergente** (`emotional-reactions` 59%,
   `explorations` 39% de rationales não-substring). Os rationales ESTÃO em PT, mas
   a coluna `response_post` e a coluna `rationales` foram traduzidas
   **separadamente** (GPT-4 Turbo, não-determinístico) → o mesmo trecho inglês
   virou frases PT ligeiramente diferentes nas duas colunas. Ex.: resposta diz
   "com base em minha experiência", rationale diz "de acordo com minha experiência".
   Não casa como substring. **Não corrigível por normalização de string**
   (testado: NFC, aspas, espaços — tudo `False`). Exigiria alinhamento fuzzy/semântico.

3. **Rationales não traduzidos** (`interpretations` 100% em inglês). A coluna
   `rationales` do IP ficou no original inglês enquanto `response_post` foi para PT.
   Nunca casa. Sem conserto por código — é dado incompleto.

## Descoberta decisiva: o paper NÃO usa a tarefa de rationale

Leitura do LaTeX do paper confirmou:
- A tabela de métricas automáticas (`tb:combined-evaluation-metrics`) reporta
  **apenas** contagens de rótulo ER/IP/EX (0/1/2). A regressão logística usa ER e EX.
  As métricas manuais são anotação humana. **Rationale não aparece em nenhuma
  tabela, métrica ou análise.**
- O EPITOME é usado puramente como **classificador de nível de empatia**.
- `rationale_labels` só existe no código porque a loss multi-tarefa do `train.py`
  a consome no treino — mas o output reportado vem da cabeça de empatia.

**Implicação**: o problema de rationale (casos 2 e 3) **não afeta o paper**. Não é
preciso traduzir, re-anotar nem alinhar rationale para responder ao revisor.

## Decisão: treinar com `--lambda_RE 0`

- `train.py` já expõe o peso via CLI: `--lambda_RE 0` (manter `--lambda_EI 1.0`).
  No `forward`: `loss = lambda_EI * loss_empathy + lambda_RE * loss_rationales`.
  Com `lambda_RE=0`, o gradiente vem só do classificador de empatia — o rationale
  corrompido não contamina o treino.
- A coluna `rationale_labels` **ainda precisa existir** no CSV (o script lê antes de
  treinar); apenas não influencia o gradiente. Não apagar a coluna.
- **Ressalva de honestidade**: `lambda_RE 0` produz um modelo levemente diferente do
  que gerou a tabela do paper (que usou o default `0.5`). É outra correção
  metodológica legítima decorrente do mesmo problema. Alternativa (fidelidade máxima):
  manter `0.5`, aceitando rationale-lixo no treino como sempre foi. **Escolhido: 0.**
- Nota sobre o IP: o paper especula que o IP=0 uniforme veio do limite de 100 tokens
  de saída. Mas o treino do IP tinha rationale 100% quebrado. Com `lambda_RE 0`, o IP
  passa a depender só de `level` (íntegro). Se ainda colapsar em 0, é sinal real —
  resposta mais honesta que a do paper atual.

## Trabalho futuro (adiado deliberadamente)

- Traduzir os rationales corretamente e realinhar ao texto PT.
- Reintroduzir o rationale no treino com peso do paper (`lambda_RE 0.5`) e medir se
  ajuda ou atrapalha o classificador de empatia.
- Eventualmente avaliar a própria tarefa de extração de rationale (IOU-F1 etc.),
  que hoje o paper não reporta.

## Comandos XLM-R (treino dos 3 + teste, encadeado)

`lambda_RE 0`, CPU, sufixo `_xlmr`. Encadeado com `&&` (para na primeira falha).
Rodar em `tmux`/`nohup` (dura ~2h40 em CPU; não perder por queda de SSH).

```bash
nohup bash -c '
export CUDA_VISIBLE_DEVICES="" &&
mkdir -p dataset/output &&
echo "[1/4] Treinando ER..." &&
python3 src/train.py --train_path dataset/er_model_xlmr.csv --lr 2e-5 --batch_size 32 --lambda_EI 1.0 --lambda_RE 0 --save_model --save_model_path output/xlmr_ER.pth &&
echo "[2/4] Treinando IP..." &&
python3 src/train.py --train_path dataset/ip_model_xlmr.csv --lr 2e-5 --batch_size 32 --lambda_EI 1.0 --lambda_RE 0 --save_model --save_model_path output/xlmr_IP.pth &&
echo "[3/4] Treinando EX..." &&
python3 src/train.py --train_path dataset/ex_model_xlmr.csv --lr 2e-5 --batch_size 32 --lambda_EI 1.0 --lambda_RE 0 --save_model --save_model_path output/xlmr_EX.pth &&
echo "[4/4] Rodando teste nos posts PT..." &&
python3 src/test.py --input_path dataset/avalia_modelos.csv --output_path dataset/output/avalia_saida_xlmr.csv --ER_model_path output/xlmr_ER.pth --IP_model_path output/xlmr_IP.pth --EX_model_path output/xlmr_EX.pth &&
echo "CONCLUIDO. Saida em dataset/output/avalia_saida_xlmr.csv"
' > treino_xlmr.log 2>&1 &
```

Acompanhar: `tail -f treino_xlmr.log`. Verificar vivo: `ps aux | grep train.py`.

## Estado das alterações XLM-R (feito)

- 4 arquivos editados (`models.py`, `process_data.py`, `train.py`,
  `empathy_classifier.py`) — ver Parte 2.
- 3 CSVs reprocessados como `*_xlmr.csv`. Rationale neles está corrompido (casos 2 e
  3 acima), mas com `lambda_RE 0` isso é irrelevante; a coluna `level` está íntegra.

## Pendências / a confirmar (atualizado)

- [x] Diagnóstico do rationale (3 casos) e causa raiz (dado, não tokenização).
- [x] Confirmar que o paper não usa rationale (só rótulos ER/IP/EX). Confirmado no LaTeX.
- [x] Decisão `lambda_RE`: treinar com 0.
- [ ] Rodar treino+teste XLM-R (comando acima) e comparar rótulos vs baseline/paper.
- [ ] (Opcional, recomendado) Refazer o baseline roberta com `lambda_RE 0` para uma
      comparação antes/depois limpa dos dois lados.
- [ ] Documentar formalmente (resposta ao revisor): troca roberta→XLM-R + desabilitação
      da tarefa de rationale por ausência de dados de rationale PT confiáveis.
- [ ] Alinhar com orientador: rodada única vs. N seeds para robustez.
- [ ] Trabalho futuro do rationale (tradução correta + realinhamento + avaliação).