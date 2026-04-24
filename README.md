# Tech Challenge Fase 2 - Projeto 1

## Otimizacao de modelos de diagnostico para saude da mulher

Este projeto implementa uma solucao completa em Python para otimizar hiperparametros de modelos de Machine Learning usando Algoritmos Geneticos, com foco em apoio diagnostico para saude da mulher. O caso de uso escolhido foi classificacao de lesoes mamarias com base no dataset publico **Breast Cancer Wisconsin Diagnostic**, disponivel no `scikit-learn`.

O sistema compara um modelo base com modelos otimizados por algoritmo genetico, priorizando **recall/sensibilidade** para reduzir risco de falso negativo em casos malignos. Tambem inclui um modulo desacoplado de LLM para gerar explicacoes em linguagem natural com cuidado etico, sem dependencia obrigatoria de API paga.

## Objetivo do projeto

- Treinar um modelo base de classificacao.
- Otimizar hiperparametros com algoritmo genetico.
- Comparar modelo base vs modelos otimizados.
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
src/
  diagnostico_saude_mulher/
    config/
    data/
    evaluation/
    genetic_algorithm/
    llm/
    models/
    utils/
tests/
docs/
notebooks/
main.py
run_project.py
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
python3 main.py
```

ou

```bash
python3 run_project.py
```

O comando executa:

- carregamento do dataset;
- treinamento do modelo base;
- 3 experimentos com algoritmo genetico;
- comparacao de metricas;
- geracao de explicacao pela LLM configurada;
- persistencia da resposta em `artifacts/llm_responses.jsonl`.

## Como treinar e exportar o modelo final

Para gerar um artefato reutilizavel do modelo:

```bash
PYTHONPATH=src python3 -m diagnostico_saude_mulher.models.train_model
```

Arquivos gerados em `artifacts/modelos/`:

- `modelo_cancer_mama.joblib`
- `metricas_modelo_final.json`
- `colunas_entrada.json`

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

## Como funciona o algoritmo genetico

O algoritmo genetico otimiza hiperparametros de um `RandomForestClassifier`:

- representacao genetica: dicionario de hiperparametros;
- populacao inicial: individuos aleatorios no espaco de busca;
- fitness: combinacao ponderada de recall, especificidade e F1-score;
- selecao: torneio ou roleta;
- crossover: uniforme gene a gene;
- mutacao: substituicao aleatoria de genes conforme taxa de mutacao;
- elitismo: melhores individuos sao preservados;
- evolucao: repetida por geracoes com historico salvo.

### Fitness

A funcao objetivo prioriza recall:

- `55%` recall
- `25%` especificidade
- `20%` F1-score
- penalizacao opcional por gap de equidade

## Experimentos obrigatorios

Foram implementados 3 experimentos com variacao de:

- tamanho da populacao;
- taxa de mutacao;
- numero de geracoes;
- estrategia de selecao.

Configuracoes executadas:

1. Populacao `6`, geracoes `3`, mutacao `0.10`, selecao `tournament`
2. Populacao `8`, geracoes `4`, mutacao `0.15`, selecao `tournament`
3. Populacao `10`, geracoes `5`, mutacao `0.20`, selecao `roulette`

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
- geracao de prompt;
- cliente mock da LLM;
- persistencia do JSONL.

## Consideracoes eticas

- O sistema e de apoio, nao de decisao autonoma.
- Resultados nao substituem avaliacao medica.
- Priorizar recall reduz falso negativo, mas pode elevar falso positivo.
- O projeto evita expor credenciais no codigo.
- A estrutura admite monitoramento futuro de viés e equidade com datasets adequados.

## Documentacao tecnica

Detalhes adicionais estao em [docs/architecture.md](/mnt/c/desenvolvimento/repositorio/tech-challenge-fase2/docs/architecture.md).
