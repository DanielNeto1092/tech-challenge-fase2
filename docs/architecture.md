# Arquitetura do Sistema

## Visao geral

O sistema foi organizado em modulos coesos e desacoplados para suportar treinamento, otimizacao genetica, avaliacao e explicabilidade por LLM no contexto de saude da mulher.

```text
Dataset -> Preparacao -> Modelo Base -> Avaliacao
                        -> Algoritmo Genetico -> Melhor Modelo -> Avaliacao
                                                       -> LLM -> JSONL
```

## Modulos principais

### `config`

- Centraliza configuracoes de aplicacao e LLM.
- Mantem credenciais fora do codigo.

### `data`

- Carrega o dataset publico.
- Faz split estratificado treino/teste.
- Normaliza a semantica do alvo para `1 = maligno`.

### `models`

- Define schemas de resultados e metricas.
- Treina modelo base e modelo otimizado.
- Executa validacao cruzada e avaliacao final.
- Exporta o modelo final e artefatos auxiliares para `artifacts/modelos/`.

### `evaluation`

- Calcula recall, especificidade, F1-score, ROC AUC, precisao e acuracia.
- Implementa a funcao fitness com prioridade para recall.
- Possui estrutura para gap de equidade.

### `genetic_algorithm`

- Implementa representacao genetica dos hiperparametros.
- Gera populacao inicial.
- Executa selecao, crossover, mutacao, elitismo e evolucao.
- Registra os resultados por geracao.

### `llm`

- Expõe uma interface desacoplada para clientes de LLM.
- Suporta `mock` e `http`.
- Gera prompts especializados para saude da mulher.
- Persiste respostas em JSONL.

### `utils`

- Configura logging padronizado.

## Fluxo de execucao

1. Carregamento do dataset de cancer de mama.
2. Treinamento do modelo base `RandomForestClassifier`.
3. Execucao de tres experimentos com configuracoes distintas do AG.
4. Reavaliacao dos melhores hiperparametros no conjunto de teste.
5. Escolha do melhor experimento com foco em recall no teste.
6. Inferencia em uma amostra de teste.
7. Geracao de explicacao em linguagem natural.
8. Persistencia da resposta em arquivo JSONL.
9. Exportacao opcional do modelo final treinado.

## Decisoes tecnicas

- Dataset embarcado no `scikit-learn` para garantir execucao local.
- `RandomForestClassifier` por robustez, interpretabilidade operacional e hiperparametros adequados ao AG.
- Validacao cruzada estratificada para reduzir risco de superajuste na fitness.
- Fitness orientada a recall por conta do custo clinico de falso negativo.
- Cliente `mock` para eliminar dependencia obrigatoria de API externa.

## Privacidade, vies e equidade

- O projeto nao usa dados sensiveis locais nem depende de envio real de prontuarios.
- Nenhuma credencial e armazenada no codigo.
- O dataset escolhido nao contem atributo demografico confiavel para auditoria de equidade.
- Ainda assim, o pipeline foi desenhado para calcular gap de recall entre grupos quando esse tipo de dado existir.
- A explicacao da LLM inclui linguagem cautelosa e nao prescritiva.

## Riscos e limites

- O dataset e academico e nao substitui validacao em ambiente clinico real.
- A explicacao textual depende da qualidade do modelo e do contexto fornecido.
- Uma estrategia focada em recall pode aumentar encaminhamentos desnecessarios.
- O AG melhora hiperparametros, mas nao corrige vies do dataset por si so.

## Evolucoes futuras

- Suporte a datasets com atributos demograficos apropriados.
- Comparacao com outros algoritmos como XGBoost e Logistic Regression.
- Dashboards para historico de geracoes e importancia de atributos.
- Registro versionado de prompts e respostas para auditoria.
