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

### `experiments`

- Orquestra baseline, execucao dos experimentos geneticos, escolha do melhor modelo e comparacao final.
- Permite reaproveitar a mesma logica em `main.py`, `run_genetic_optimization.py` e `app.py`.

### `visualization`

- Gera graficos de convergencia, comparacao e resumo dos experimentos.

### `utils`

- Configura logging padronizado.

## Fluxo de execucao

1. Carregamento do dataset de cancer de mama.
2. Treinamento e avaliacao dos modelos baseline suportados pelo projeto.
3. Escolha automatica do modelo de referencia do AG com base na funcao de fitness.
4. Execucao de tres experimentos com configuracoes distintas do AG sobre a familia selecionada.
5. Reavaliacao dos melhores hiperparametros no conjunto de teste.
6. Escolha do melhor experimento com foco em recall, F1-score e especificidade.
7. Inferencia em uma amostra de teste.
8. Geracao de explicacao em linguagem natural.
9. Persistencia da resposta em arquivo JSONL.
10. Exportacao opcional do modelo final treinado.
11. Geracao opcional de graficos para acompanhamento dos experimentos.

## Decisoes tecnicas

- Dataset embarcado no `scikit-learn` para garantir execucao local.
- Comparacao entre multiplos modelos baseline para evitar fixar a etapa genetica em uma unica familia sem evidencia empirica.
- O modelo de referencia do AG nao e definido manualmente; ele e escolhido automaticamente pelo melhor valor de fitness entre os baselines.
- A familia otimizada pelo AG depende do melhor candidato encontrado na etapa baseline, o que torna a busca mais coerente com o desempenho observado.
- Validacao estratificada na avaliacao do pipeline para reduzir risco de distorcoes entre treino e teste.
- Fitness orientada a recall por conta do custo clinico de falso negativo.
- Cliente `mock` para eliminar dependencia obrigatoria de API externa.

## Interface, API e operacao

- A interface Streamlit permite carregar o dataset padrao ou enviar CSV, executar baseline, disparar a otimizacao genetica, comparar metricas e consultar o historico da LLM.
- A API separada em FastAPI prepara a base para desacoplamento entre interface, inferencia e execucao de cargas mais pesadas.
- O projeto persiste artefatos em `artifacts/`, incluindo logs, respostas da LLM, graficos e modelos exportados.
- A arquitetura atual permite migrar a etapa genetica para jobs assíncronos ou workers dedicados sem alterar a camada de apresentacao.

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
- Ampliacao da comparacao com outras familias de modelos, como XGBoost e SVM.
- Dashboards para historico de geracoes e importancia de atributos.
- Registro versionado de prompts e respostas para auditoria.
- Execucao da otimizacao genetica em background com fila e worker dedicado.
