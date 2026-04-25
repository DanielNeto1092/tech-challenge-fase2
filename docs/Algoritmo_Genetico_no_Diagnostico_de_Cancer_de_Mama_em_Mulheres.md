# Algoritmos Geneticos no Diagnostico de Cancer de Mama em Mulheres

## 1. Resumo

Este relatorio apresenta a execucao do estudo de otimizacao de hiperparametros com Algoritmos Geneticos aplicada ao dataset **Breast Cancer Wisconsin Diagnostic**. O objetivo foi comparar modelos baseline e modelos otimizados, com prioridade para **recall**, por se tratar de um problema de apoio ao diagnostico medico, em que falsos negativos sao mais criticos.

O estudo avaliou quatro familias de modelos:

- `KNeighborsClassifier`
- `DecisionTreeClassifier`
- `LogisticRegression`
- `RandomForestClassifier`

Para cada familia, foram executados tres experimentos de Algoritmo Genetico, com selecao por torneio, crossover, mutacao, elitismo e registro por geracao.

## 2. Objetivo

O estudo foi estruturado para responder quatro perguntas:

- qual o desempenho baseline de cada modelo no conjunto de teste;
- se o Algoritmo Genetico melhora o recall de cada familia;
- quais sao os trade-offs entre recall, especificidade e F1-score;
- qual modelo apresenta o melhor equilibrio para recomendacao de uso clinico.

## 3. Dataset e Preparacao

O dataset utilizado foi o **Breast Cancer Wisconsin Diagnostic**, disponibilizado no `scikit-learn`.

- total de amostras: `569`
- treino: `455`
- teste: `114`
- classe positiva: `maligno`

Etapas de preparacao:

- normalizacao com `StandardScaler`;
- split estratificado `80/20`;
- avaliacao final no conjunto de teste;
- conjunto de validacao interno para a funcao de fitness do AG.

## 4. Modelos Baseline

Os modelos baseline foram treinados com configuracoes padrao e avaliados no conjunto de teste.

| Modelo | Recall | Especificidade | F1-score | Acuracia |
|---|---:|---:|---:|---:|
| KNeighborsClassifier | 90,48% | 98,61% | 93,83% | 95,61% |
| DecisionTreeClassifier | 90,48% | 94,44% | 90,48% | 92,98% |
| LogisticRegression | 95,24% | 98,61% | 96,39% | 97,37% |
| RandomForestClassifier | 92,86% | 100,00% | 96,30% | 97,37% |

Leitura inicial:

- `LogisticRegression` teve o maior recall entre os baselines;
- `RandomForestClassifier` teve especificidade perfeita no teste;
- `KNeighborsClassifier` apresentou baseline forte, mas abaixo de `LogisticRegression` em recall;
- `DecisionTreeClassifier` foi o baseline menos equilibrado.

## 5. Algoritmo Genetico

### 5.1 Funcao de fitness

A funcao de fitness foi definida com prioridade explicita para recall:

`fitness = (0.6 * recall) + (0.3 * f1_score) + (0.1 * especificidade)`

Essa escolha busca manter foco clinico na reducao de falsos negativos, sem ignorar o equilibrio geral do classificador.

### 5.2 Operadores utilizados

- inicializacao aleatoria da populacao;
- selecao por torneio;
- crossover uniforme;
- mutacao aleatoria por gene;
- elitismo;
- registro do melhor fitness e do fitness medio por geracao.

### 5.3 Configuracoes executadas

Foram executados tres experimentos para **cada modelo**:

1. `Experimento 1 - Exploracao`
- populacao: `20`
- geracoes: `15`
- crossover: `0.80`
- mutacao: `0.30`
- torneio: `3`
- elitismo: `2`

2. `Experimento 2 - Equilibrio`
- populacao: `30`
- geracoes: `20`
- crossover: `0.85`
- mutacao: `0.15`
- torneio: `4`
- elitismo: `3`

3. `Experimento 3 - Refinamento`
- populacao: `40`
- geracoes: `25`
- crossover: `0.90`
- mutacao: `0.05`
- torneio: `5`
- elitismo: `4`

## 6. Resultados da Otimizacao

Os resultados abaixo foram extraidos da execucao real registrada em [results.json](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/artifacts/ga_study/results.json).

| Modelo | Recall Baseline | Recall Otimizado | Delta Recall | Especificidade Final | F1-score Final |
|---|---:|---:|---:|---:|---:|
| KNeighborsClassifier | 90,48% | 92,86% | 2,38 p.p. | 95,83% | 92,86% |
| DecisionTreeClassifier | 90,48% | 88,10% | -2,38 p.p. | 98,61% | 92,50% |
| LogisticRegression | 95,24% | 95,24% | 0,00 p.p. | 98,61% | 96,39% |
| RandomForestClassifier | 92,86% | 88,10% | -4,76 p.p. | 98,61% | 92,50% |

### 6.1 Melhor experimento por modelo

- `KNeighborsClassifier`: `experimento_1_exploracao`
- `DecisionTreeClassifier`: `experimento_2_equilibrio`
- `LogisticRegression`: `experimento_1_exploracao`
- `RandomForestClassifier`: `experimento_1_exploracao`

### 6.2 Hiperparametros otimizados encontrados

**KNeighborsClassifier**

- `n_neighbors = 2`
- `weights = distance`
- `p = 1`

**DecisionTreeClassifier**

- `max_depth = 9`
- `min_samples_split = 6`
- `min_samples_leaf = 1`
- `criterion = entropy`

**LogisticRegression**

- `C = 7.5`
- `max_iter = 600`
- `solver = liblinear`

**RandomForestClassifier**

- `n_estimators = 400`
- `max_depth = 30`
- `min_samples_split = 42`
- `min_samples_leaf = 3`

## 7. Analise Automatica dos Resultados

Leitura consolidada da execucao:

- o maior ganho de recall foi obtido por `KNeighborsClassifier`, com aumento de `2,38` pontos percentuais;
- `DecisionTreeClassifier` e `RandomForestClassifier` melhoraram alguns aspectos de equilibrio, mas perderam recall;
- `LogisticRegression` nao ganhou recall adicional, mas manteve o melhor equilibrio geral entre recall, especificidade e F1-score;
- a recomendacao automatica para uso clinico permaneceu em `LogisticRegression`.

Em termos praticos:

- `KNeighborsClassifier` foi o modelo com maior ganho de sensibilidade;
- `LogisticRegression` foi o modelo mais estavel;
- `RandomForestClassifier` nao sustentou no teste a vantagem observada na validacao do AG;
- `DecisionTreeClassifier` trocou recall por especificidade.

## 8. Interpretacao Tecnica

O estudo mostrou que otimizar por AG nem sempre significa melhorar o recall final em todas as familias. O comportamento foi diferente entre os modelos:

- em `KNeighborsClassifier`, o AG encontrou uma configuracao simples e eficaz, com ganho real de recall;
- em `LogisticRegression`, o baseline ja era muito forte e o AG apenas confirmou uma regiao otima, sem ganho de teste;
- em `DecisionTreeClassifier` e `RandomForestClassifier`, a otimizacao favoreceu solucoes mais conservadoras, com queda de recall e ganho de especificidade.

Isso reforca um ponto importante: em problemas clinicos, a escolha do melhor modelo nao pode depender apenas do fitness em validacao. O comportamento no conjunto de teste precisa ser analisado junto com os trade-offs.

## 9. Graficos e Evidencias Visuais

Arquivos gerados nesta execucao:

- [Comparacao baseline vs otimizado](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/artifacts/ga_study/plots/comparacao_baseline_vs_otimizado.png)
- [Convergencia KNN - experimento 1](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/artifacts/ga_study/plots/convergencia_KNeighborsClassifier_experimento_1_exploracao.png)
- [Convergencia Decision Tree - experimento 2](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/artifacts/ga_study/plots/convergencia_DecisionTreeClassifier_experimento_2_equilibrio.png)
- [Convergencia Logistic Regression - experimento 1](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/artifacts/ga_study/plots/convergencia_LogisticRegression_experimento_1_exploracao.png)
- [Convergencia Random Forest - experimento 1](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/artifacts/ga_study/plots/convergencia_RandomForestClassifier_experimento_1_exploracao.png)

Esses graficos mostram:

- melhor fitness por geracao;
- fitness medio por geracao;
- diversidade da populacao;
- individuos unicos por geracao;
- evolucao dos parametros do melhor individuo.

## 10. Interface, API e Operacao

O projeto tambem entrega componentes opcionais relevantes:

- interface em Streamlit para carregar dataset, executar baseline e AG, comparar metricas e consultar historico da LLM;
- API separada em FastAPI;
- logging em arquivo;
- persistencia de artefatos em JSON, JSONL e imagens;
- Docker e `docker-compose`;
- estrutura inicial em Terraform;
- documentacao de nuvem e escalabilidade.

## 11. Consideracoes Eticas

- o sistema e de apoio, nao de decisao autonoma;
- o foco em recall reduz falso negativo, mas pode elevar falso positivo;
- o dataset e academico e nao substitui validacao clinica real;
- a camada de LLM inclui aviso explicito de que a resposta nao substitui avaliacao medica;
- nao ha credenciais expostas no repositorio.

## 12. Conclusao

Com base nos resultados reais desta execucao:

- o AG trouxe ganho de recall apenas para `KNeighborsClassifier`;
- `LogisticRegression` permaneceu como melhor opcao geral;
- `DecisionTreeClassifier` e `RandomForestClassifier` nao sustentaram ganho de sensibilidade no teste;
- para uso clinico com o criterio adotado neste estudo, a recomendacao final e `LogisticRegression`.

O estudo cumpriu o objetivo de comparar baseline vs otimizado, registrar os melhores hiperparametros, gerar graficos de convergencia e produzir uma analise automatica coerente com as metricas observadas.
