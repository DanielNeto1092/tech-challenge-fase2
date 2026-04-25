# Roteiro de Video - 15 Minutos

## Fala por Slide

### Slide 1 - Titulo

Esta apresentacao mostra o Projeto 1 do Tech Challenge Fase 2. A proposta foi desenvolver uma solucao para apoio ao diagnostico de cancer de mama em mulheres, utilizando Machine Learning, Algoritmo Genetico para otimizacao de hiperparametros e uma camada de LLM para explicacoes em linguagem natural.

### Slide 2 - Problema e objetivo

O problema foi tratado como classificacao binaria, com foco em reduzir falsos negativos. Por isso, o projeto priorizou recall como metrica principal. O objetivo foi comparar modelos baseline, definir uma referencia para otimizacao e avaliar o impacto do Algoritmo Genetico no desempenho do modelo.

### Slide 3 - Dataset

O dataset utilizado foi o Breast Cancer Wisconsin Diagnostic, disponivel no scikit-learn. Ele possui 569 amostras e foi ajustado no projeto para que a classe positiva representasse malignidade. Isso ajuda a alinhar o recall com o objetivo clinico de identificar corretamente casos malignos.

### Slide 4 - Arquitetura

A solucao foi organizada em modulos separados para dados, modelos, avaliacao, algoritmo genetico, explicabilidade, visualizacao, API e interface web. Alem do pipeline principal, o projeto inclui Streamlit, FastAPI, logs, geracao de artefatos e preparacao para containerizacao e nuvem.

### Slide 5 - Algoritmo Genetico

O Algoritmo Genetico foi implementado com populacao inicial, fitness, selecao, crossover, mutacao, elitismo e historico por geracao. A funcao de fitness foi definida com 55% de peso para recall, 25% para especificidade e 20% para F1-score.

### Slide 6 - Experimentos

Foram executados tres experimentos geneticos, variando populacao, numero de geracoes, taxa de mutacao e estrategia de selecao. Isso permitiu observar nao apenas o melhor resultado final, mas tambem o comportamento da busca ao longo das geracoes.

### Slide 7 - Comparacao de baselines

Antes da etapa genetica, o projeto comparou quatro modelos baseline: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier e KNeighborsClassifier.

Na execucao atual, a LogisticRegression foi o melhor baseline, com recall de 95,24%, especificidade de 98,61%, F1-score de 96,39% e acuracia de 97,37%.

### Slide 8 - Baseline de referencia do AG

O RandomForest foi mantido como modelo de referencia para a etapa genetica. No conjunto de teste, ele apresentou recall de 92,86%, especificidade de 98,61%, F1-score de 95,12% e acuracia de 96,49%. A escolha dele para o AG se justifica pela existencia de um espaco de hiperparametros mais adequado para busca evolutiva.

### Slide 9 - Hiperparametros otimizados

No experimento 2, que foi o melhor dentro da familia RandomForest, o AG encontrou a seguinte configuracao: n_estimators igual a 300, max_depth igual a 6, min_samples_split igual a 6, min_samples_leaf igual a 2 e max_features igual a log2.

### Slide 10 - Comparacao baseline x AG

Nesta comparacao, o ponto principal e que o AG nao aumentou o recall em relacao ao RandomForest baseline, que permaneceu em 92,86%. Mas ele melhorou o equilibrio geral do modelo ao aumentar a especificidade para 100% e o F1-score para 96,30%.

### Slide 11 - Resumo dos experimentos

O grafico de resumo mostra o fitness final dos tres experimentos. O experimento 2 teve o maior fitness final, com valor aproximado de 0,9510. Isso indica que ele foi o melhor equilibrio encontrado segundo a funcao de fitness definida no projeto.

### Slide 12 - Convergencia do AG

Neste grafico de convergencia, vemos como o fitness evolui ao longo das geracoes. No experimento 2, o melhor fitness sai de aproximadamente 0,9471 e chega a 0,9510, mostrando melhora ao longo da busca.

### Slide 13 - Fitness por hiperparametro

Este grafico mostra como o fitness se comportou em relacao aos hiperparametros observados ao longo das geracoes. A ideia aqui nao e apenas mostrar o melhor ponto final, mas evidenciar que houve variacao real durante a busca genetica.

### Slide 14 - Metricas por geracao

Aqui vemos a evolucao de recall, especificidade e F1-score por geracao. Isso ajuda a entender se a melhoria veio de um ganho isolado ou de um equilibrio progressivo entre as metricas.

### Slide 15 - LLM e explicabilidade

O projeto tambem inclui uma camada de LLM desacoplada, com suporte a mock e integracao HTTP. Ela gera explicacoes em linguagem natural a partir da classificacao, da probabilidade e das metricas do modelo, sempre com cautela e com aviso de que o sistema nao substitui avaliacao medica.

### Slide 16 - Interface web

A interface Streamlit permite visualizar os modelos baseline, os experimentos geneticos, os graficos, simular predicoes e consultar o historico das respostas geradas pela LLM.

### Slide 17 - API, logging e nuvem

A solucao tambem conta com API em FastAPI, logging em arquivo, monitoramento na interface, Docker e estrutura inicial com Terraform. Isso deixa o projeto mais preparado para evolucao e implantacao futura.

### Slide 18 - Conclusoes

Como conclusao, o projeto entregou comparacao entre multiplos baselines, otimizacao genetica com rastreabilidade por geracao, integracao com LLM, interface, API, logs e documentacao.

Na execucao atual, a LogisticRegression foi o baseline mais forte. Ja o Algoritmo Genetico melhorou o equilibrio do RandomForest, com destaque para o experimento 2.
