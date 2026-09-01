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

# 1. pré-processar (já feito)
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

## Pendências / a confirmar

- [ ] Rodar o crosstab de rationale vazio em `ip_model.csv` e `ex_model.csv`
      (confirmar que o padrão invertido se repete).
- [ ] Documentar formalmente (resposta ao revisor) a decisão de confiar no
      `process_data.py` atual na ausência do responsável e dos pesos originais.
- [ ] Alinhar com o orientador: o baseline reproduz o pipeline original, mas os
      números de rationale do "antes" refletem dados corrompidos, não desempenho.
- [ ] Verificar tamanho dos `.pth` salvos (`ls -lh output/`): ~1.2–1.5 GB cada
      (guarda os dois encoders inteiros).