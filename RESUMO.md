# Resumo troca RoBERTa → XLM-RoBERTa (resposta ao revisor)

1. **Encoder/tokenizador errado** (o ponto do revisor). Corrigido com XLM-R. É o
   único dos três que afeta os resultados reportados no paper.
2. **Rationales com tradução divergente** (emotional-reactions ~59%,
   explorations ~39%): a resposta e o rationale foram traduzidos separadamente, então
   o rationale em PT quase nunca é trecho exato da resposta em PT.
3. **Rationales não traduzidos** (interpretations: 100% em inglês).

**Ponto-chave**: o paper **não usa a tarefa de rationale** — só reporta as contagens
de rótulo ER/IP/EX. Logo, os problemas 2 e 3 **não afetam os resultados do paper**.

Optei por treinar desabilitando o rationale (`lambda_RE = 0`) por enquanto -> não sabia o que fazer.

## Resultados: contagens ER / IP / EX (20 posts por modelo)

**Artigo (publicado, com roberta-base):**

| Model | ER-0 | ER-1 | ER-2 | IP-0 | IP-1 | IP-2 | EX-0 | EX-1 | EX-2 |
|-------|------|------|------|------|------|------|------|------|------|
| Llama | 17 | 3 | 0 | 20 | 0 | 0 | 7  | 0 | 13 |
| Gemma | 17 | 3 | 0 | 20 | 0 | 0 | 7  | 0 | 13 |
| GPT   | 19 | 1 | 0 | 20 | 0 | 0 | 18 | 0 | 2  |

**Minha reprodução com roberta-base (baseline "antes"):**

| Model | ER-0 | ER-1 | ER-2 | IP-0 | IP-1 | IP-2 | EX-0 | EX-1 | EX-2 |
|-------|------|------|------|------|------|------|------|------|------|
| Llama | 18 | 2 | 0 | 20 | 0 | 0 | 7  | 0 | 13 |
| Gemma | 18 | 2 | 0 | 20 | 0 | 0 | 8  | 0 | 12 |
| GPT   | 19 | 1 | 0 | 20 | 0 | 0 | 20 | 0 | 0  |

**Com xlm-roberta-base (correção do revisor, "depois"):**

| Model | ER-0 | ER-1 | ER-2 | IP-0 | IP-1 | IP-2 | EX-0 | EX-1 | EX-2 |
|-------|------|------|------|------|------|------|------|------|------|
| Llama | 19 | 1 | 0 | 20 | 0 | 0 | 7  | 0 | 13 |
| Gemma | 20 | 0 | 0 | 20 | 0 | 0 | 7  | 0 | 13 |
| GPT   | 17 | 3 | 0 | 20 | 0 | 0 | 18 | 0 | 2  |