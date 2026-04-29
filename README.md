# Tech Challenge Fase 2 - Projeto 1

## Otimizacao de modelos de diagnostico para saude da mulher

Este projeto implementa uma solucao completa em Python para otimizar hiperparametros de modelos de Machine Learning usando Algoritmos Geneticos, com foco em apoio diagnostico para saude da mulher. O caso de uso escolhido foi classificacao de lesoes mamarias com base no dataset publico **Breast Cancer Wisconsin Diagnostic**, disponivel no `scikit-learn`.

O sistema compara quatro modelos baseline e, em seguida, seleciona automaticamente a familia mais performatica por fitness no conjunto de validacao cruzada. A familia vencedora passa a ser otimizada por algoritmo genetico em tres experimentos com perfis distintos de busca, mantendo prioridade clinica em **recall/sensibilidade** para reduzir risco de falso negativo em casos malignos. Tambem inclui um modulo desacoplado de LLM para gerar explicacoes em linguagem natural com cuidado etico, sem dependencia obrigatoria de API paga.

## Objetivo do projeto

- Treinar e comparar multiplos modelos baseline de classificacao.
- Otimizar hiperparametros com algoritmo genetico.
- Comparar modelos baseline vs candidatos otimizados por algoritmo genetico.
- Gerar explicacoes em linguagem natural para apoio ao profissional de saude.
- Persistir respostas da LLM em JSONL para reuso futuro.
- Documentar arquitetura, execucao, consideracoes eticas e testes.

## Dataset escolhido

- Nome: `Breast Cancer Wisconsin Diagnostic`
- Fonte: dataset publico embarcado no `scikit-learn`
- Dominio: apoio ao diagnostico de cancer de mama
- Alvo no projeto: `1 = maligno`, `0 = benigno`

Observacao: o dataset nao possui variaveis demograficas apropriadas para uma analise robusta de equidade. O projeto, no entanto, implementa a estrutura para incluir esse calculo quando um dataset adequado estiver disponivel.

## Estrutura do projeto

```text
artifacts/
  llm_responses.jsonl
  modelos/
  graficos/
src/
  diagnostico_saude_mulher/
    config/
    data/
    evaluation/
    experiments/
    genetic_algorithm/
    llm/
    models/
    utils/
    visualization/
tests/
docs/
notebooks/
main.py
app.py
run_genetic_optimization.py
README.md
requirements.txt
.env.example
```

## Como instalar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Como executar

Execucao principal:

```bash
PYTHONPATH=src python3 main.py
```

O comando executa:

- carregamento do dataset;
- comparacao entre `RandomForestClassifier`, `LogisticRegression`, `DecisionTreeClassifier` e `KNeighborsClassifier`;
- 3 experimentos com algoritmo genetico;
- comparacao de metricas;
- geracao de explicacao pela LLM configurada;
- persistencia da resposta em `artifacts/llm_responses.jsonl`.

## Como executar a API

Subir a API local:

```bash
PYTHONPATH=src ./.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000
```

Endpoints principais:

- `GET /health`
- `GET /dataset`
- `POST /baseline`
- `POST /optimize`
- `POST /predict`
- `GET /llm/history`
- `GET /logs`

## Como rodar a otimizacao genetica com graficos

```bash
PYTHONPATH=src python3 run_genetic_optimization.py
```

Saidas geradas:

- `artifacts/experimentos_geneticos.json`
- `artifacts/graficos/comparacao_experimentos.png`
- `artifacts/graficos/resumo_experimentos.png`
- `artifacts/graficos/convergencia_ag_experimento_*.png`

## Como treinar e exportar o modelo final

Para gerar um artefato reutilizavel do modelo:

```bash
PYTHONPATH=src python3 -m diagnostico_saude_mulher.models.train_model
```

Arquivos gerados em `artifacts/modelos/`:

- `modelo_cancer_mama.joblib`
- `metricas_modelo_final.json`
- `colunas_entrada.json`

## Como abrir a interface web

```bash
streamlit run app.py
```

A interface permite:

- visualizar os modelos baseline e os experimentos geneticos;
- inspecionar graficos comparativos;
- simular uma predicao em cima de uma amostra do conjunto de teste;
- gerar uma explicacao textual via `mock` ou `http`.

## Como executar com Docker

Build e subida da interface web:

```bash
docker-compose up --build
```

Depois acesse:

- `http://localhost:8501`

O compose sobe a interface Streamlit com `LLM_PROVIDER=mock` por padrao e persiste os artefatos em `./artifacts`.

## Monitoramento e logging

O projeto grava logs em:

- `artifacts/logs/aplicacao.log`

Os logs incluem:

- treinamento do modelo;
- execucao do algoritmo genetico;
- metricas por geracao;
- chamadas e persistencia da LLM;
- erros de execucao quando propagados para logging.

Para acompanhar:

```bash
tail -f artifacts/logs/aplicacao.log
```

Na interface Streamlit existe uma aba de monitoramento que exibe o caminho do log e as ultimas linhas registradas.

## Como rodar testes

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Variaveis de ambiente

Copie o modelo:

```bash
cp .env.example .env
```

Principais variaveis:

- `LLM_PROVIDER`: `mock` ou `http`
- `LLM_MODEL`: nome do modelo
- `LLM_API_KEY`: chave da API, nunca commitar
- `LLM_ENDPOINT`: endpoint compativel com chat completions
- `LLM_OUTPUT_PATH`: caminho do arquivo JSONL de saida
- `LOG_LEVEL`: nivel de log

## Infraestrutura e escalabilidade

O projeto ja esta preparado para execucao via container com:

- [Dockerfile](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/Dockerfile)
- [docker-compose.yml](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/docker-compose.yml)

Tambem foi incluida uma estrutura inicial de Infraestrutura como Codigo em:

- [infra/terraform/main.tf](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/infra/terraform/main.tf)
- [infra/terraform/variables.tf](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/infra/terraform/variables.tf)
- [infra/terraform/outputs.tf](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/infra/terraform/outputs.tf)

Arquitetura proposta e estrategia de escala:

- [docs/cloud_architecture.md](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/docs/cloud_architecture.md)

## Como funciona o algoritmo genetico

O algoritmo genetico otimiza hiperparametros da familia de modelo selecionada automaticamente pela funcao de fitness depois da comparacao entre baselines:

- representacao genetica: dicionario de hiperparametros;
- populacao inicial: individuos aleatorios no espaco de busca;
- fitness: combinacao ponderada de recall, especificidade e F1-score, com pesos definidos por experimento;
- selecao: torneio ou roleta;
- crossover: uniforme gene a gene;
- mutacao: substituicao aleatoria de genes conforme taxa de mutacao;
- elitismo: melhores individuos sao preservados;
- evolucao: repetida por geracoes com historico salvo.

### Fitness

A funcao objetivo sempre combina as mesmas metricas:

- recall
- especificidade
- F1-score
- penalizacao opcional por gap de equidade

O que varia entre os experimentos sao os pesos dessa combinacao:

1. `AG Experimento 1`: `70%` recall, `15%` especificidade, `15%` F1-score
2. `AG Experimento 2`: `55%` recall, `25%` especificidade, `20%` F1-score
3. `AG Experimento 3`: `40%` recall, `30%` especificidade, `30%` F1-score

## Experimentos obrigatorios

Foram implementados 3 experimentos com variacao de:

- tamanho da populacao;
- taxa de mutacao;
- numero de geracoes;
- estrategia de selecao.
- taxa de crossover;
- elitismo;
- pesos da funcao de fitness.

Configuracoes executadas:

1. `AG Experimento 1`: populacao `6`, geracoes `3`, mutacao `0.10`, crossover `0.85`, elitismo `1`, torneio `2`, selecao `tournament`
2. `AG Experimento 2`: populacao `8`, geracoes `4`, mutacao `0.20`, crossover `0.90`, elitismo `2`, torneio `3`, selecao `tournament`
3. `AG Experimento 3`: populacao `12`, geracoes `6`, mutacao `0.30`, crossover `0.95`, elitismo `2`, torneio `4`, selecao `roulette`

Para aumentar a diversidade entre os cenarios, os experimentos posteriores nao aceitam repetir a melhor combinacao de hiperparametros ja encontrada por um experimento anterior. Assim, os tres resultados apresentados representam candidatos distintos dentro da mesma familia de modelo.

## Modelos baseline comparados

O projeto compara os seguintes algoritmos antes da etapa genetica:

- `RandomForestClassifier`
- `LogisticRegression`
- `DecisionTreeClassifier`
- `KNeighborsClassifier`

Na execucao atual, a `LogisticRegression` foi o modelo de referencia selecionado pelo fitness. Por isso, os experimentos geneticos passaram a otimizar essa familia, e nao mais um modelo fixado manualmente.

## Integracao com LLM

O modulo de LLM foi desacoplado por interface:

- `MockLLMClient`: uso local e testes automatizados
- `HTTPLLMClient`: integracao via endpoint HTTP compativel

As explicacoes geradas incluem:

- classificacao prevista;
- probabilidade estimada;
- interpretacao cautelosa;
- orientacoes para profissionais de saude;
- linguagem sensivel a genero;
- aviso de que o sistema nao substitui avaliacao medica.

Todas as respostas sao persistidas em JSONL para futura base de fine-tuning ou auditoria.

## Resultados esperados

- modelo base com desempenho forte e reproducivel;
- ao menos um experimento genetico com melhora de recall e/ou equilibrio entre recall e especificidade;
- historico de evolucao por geracao;
- explicacao textual salva em arquivo.

Como o algoritmo tem componente estocastico, pequenas variacoes podem ocorrer.

## Testes automatizados

Cobertura incluida para:

- funcao fitness;
- operadores geneticos;
- treinamento basico do modelo;
- integracao do pipeline principal;
- exportacao de artefatos do modelo final;
- geracao de graficos;
- geracao de prompt;
- cliente mock da LLM;
- persistencia do JSONL.

## Consideracoes eticas

- O sistema e de apoio, nao de decisao autonoma.
- Resultados nao substituem avaliacao medica.
- Priorizar recall reduz falso negativo, mas pode elevar falso positivo.
- O projeto evita expor credenciais no codigo.
- A estrutura admite monitoramento futuro de viés e equidade com datasets adequados.

## Decisoes tecnicas

- O dataset foi escolhido por ser publico, reprodutivel e aderente ao dominio de saude da mulher.
- O modelo de referencia do AG nao e fixo: ele e escolhido automaticamente pelo melhor fitness entre os baselines.
- Os hiperparametros otimizados dependem da familia vencedora no baseline, o que torna a busca evolutiva coerente com o melhor candidato encontrado.
- Os experimentos geneticos usam a mesma formula de fitness, mas com pesos diferentes por cenario para comparar estrategias mais orientadas a recall ou mais equilibradas.
- A prioridade de recall continua intencional, porque falso negativo em contexto oncologico e clinicamente mais sensivel.
- A camada de LLM foi desacoplada para permitir mock local, integracao HTTP e futura evolucao para a Fase 3.
- A arquitetura modular prepara a base para execucao assincrona, workers dedicados, API separada e servicos desacoplados na Fase 3.

## Documentacao tecnica

Detalhes adicionais estao em [docs/architecture.md](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/docs/architecture.md).
