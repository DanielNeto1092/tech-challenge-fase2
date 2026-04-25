from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import streamlit as st

from diagnostico_saude_mulher.config.settings import AppSettings, LLMSettings
from diagnostico_saude_mulher.data.datasets import carregar_dataset_cancer_mama, construir_dataset_de_dataframe
from diagnostico_saude_mulher.experiments.runner import (
    build_comparison_payload,
    build_genetic_configs,
    choose_best_experiment,
    run_experiments,
)
from diagnostico_saude_mulher.llm.clients import build_llm_client
from diagnostico_saude_mulher.llm.service import DiagnosticExplanationService, load_explanation_history
from diagnostico_saude_mulher.models.training import criar_modelo_otimizado_por_nome, prever_amostra
from diagnostico_saude_mulher.ui.viewmodels import experiment_results_to_dataframe, llm_history_to_dataframe
from diagnostico_saude_mulher.utils.logging_utils import configure_logging
from diagnostico_saude_mulher.visualization.plots import plot_convergence, plot_experiment_comparison, plot_experiment_summary

st.set_page_config(page_title="Diagnostico em Saude da Mulher", layout="wide")
st.title("Diagnostico assistido para saude da mulher")
st.caption("Machine Learning, Algoritmo Genetico, monitoramento e explicabilidade em linguagem natural.")

settings = AppSettings()
configure_logging(settings.log_level, settings.log_output_dir)
llm_settings = LLMSettings()

if "dataset_bundle" not in st.session_state:
    st.session_state.dataset_bundle = carregar_dataset_cancer_mama(settings)
if "baseline_result" not in st.session_state:
    st.session_state.baseline_result = None
if "baseline_models" not in st.session_state:
    st.session_state.baseline_models = None
if "genetic_results" not in st.session_state:
    st.session_state.genetic_results = None
if "best_result" not in st.session_state:
    st.session_state.best_result = None


def _render_metrics(result, prefix: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Recall {prefix}", f"{result.metrics_teste.recall:.2%}")
    col2.metric(f"Especificidade {prefix}", f"{result.metrics_teste.specificity:.2%}")
    col3.metric(f"F1 {prefix}", f"{result.metrics_teste.f1_score:.2%}")
    col4.metric(f"ROC AUC {prefix}", f"{result.metrics_teste.roc_auc:.2%}")


with st.sidebar:
    st.header("Configuracao")
    origem = st.radio("Dataset", ["Dataset padrao", "Upload CSV"])
    uploaded = st.file_uploader("Enviar CSV", type=["csv"]) if origem == "Upload CSV" else None
    if st.button("Carregar dataset"):
        try:
            if origem == "Dataset padrao":
                st.session_state.dataset_bundle = carregar_dataset_cancer_mama(settings)
            elif uploaded is not None:
                dataframe = pd.read_csv(uploaded)
                st.session_state.dataset_bundle = construir_dataset_de_dataframe(dataframe, settings, nome=uploaded.name)
            st.session_state.baseline_result = None
            st.session_state.baseline_models = None
            st.session_state.genetic_results = None
            st.session_state.best_result = None
            st.success("Dataset carregado com sucesso.")
        except Exception as exc:
            st.error(f"Falha ao carregar dataset: {exc}")

    st.markdown("---")
    if st.button("Executar modelo base", use_container_width=True):
        dataset = st.session_state.dataset_bundle
        baseline_models, baseline, _ = run_experiments(settings, dataset, [])
        st.session_state.baseline_models = baseline_models
        st.session_state.baseline_result = baseline
        st.success("Comparacao de modelos baseline executada.")

    if st.button("Executar otimizacao genetica", use_container_width=True):
        dataset = st.session_state.dataset_bundle
        baseline_models, baseline, experimentos = run_experiments(settings, dataset, build_genetic_configs())
        st.session_state.baseline_models = baseline_models
        st.session_state.baseline_result = baseline
        st.session_state.genetic_results = experimentos
        st.session_state.best_result = choose_best_experiment(experimentos)
        st.success("Otimizacao genetica concluida.")


dataset = st.session_state.dataset_bundle
baseline_state = st.session_state.baseline_result
baseline_models = st.session_state.baseline_models
genetic_results = st.session_state.genetic_results
best_result = st.session_state.best_result

tab_dataset, tab_modelos, tab_predicao, tab_llm, tab_monitoramento = st.tabs(
    ["Dataset", "Modelos", "Predicao", "LLM", "Monitoramento"]
)

with tab_dataset:
    st.subheader("Dataset carregado")
    st.write({"nome": dataset.nome, "amostras": len(dataset.atributos), "atributos": len(dataset.atributos.columns)})
    st.dataframe(dataset.atributos.head(10), use_container_width=True)
    st.caption("Se usar upload CSV, inclua a coluna alvo como `diagnostico_maligno`, `target`, `Outcome` ou `diagnosis`.")

with tab_modelos:
    st.subheader("Resultados")
    if baseline_state is None and genetic_results is None:
        st.info("Execute o modelo base ou a otimizacao genetica na barra lateral.")
    if baseline_state is not None:
        st.markdown("**Modelo de referencia para o AG, selecionado pelo fitness**")
        _render_metrics(baseline_state, baseline_state.modelo)
    if baseline_models:
        st.markdown("**Comparacao entre modelos baseline**")
        st.dataframe(experiment_results_to_dataframe(baseline_models), width="stretch")
    if genetic_results:
        st.markdown("**Experimentos geneticos**")
        tabela = experiment_results_to_dataframe(genetic_results)
        st.dataframe(tabela, width="stretch")
        if best_result is not None and baseline_state is not None and hasattr(baseline_state, "metrics_teste"):
            comparacao = build_comparison_payload(baseline_state, best_result)
            st.json(comparacao)
            output_dir = ROOT / "artifacts" / "graficos_streamlit"
            st.image(str(plot_experiment_comparison(baseline_state, genetic_results, output_dir)))
            st.image(str(plot_experiment_summary(genetic_results, output_dir)))
            for experimento in genetic_results:
                with st.expander(f"Convergencia - {experimento.rotulo_exibicao}"):
                    st.image(str(plot_convergence(experimento, output_dir)))

with tab_predicao:
    st.subheader("Predicao assistida")
    modelo_resultado = best_result or baseline_state
    if modelo_resultado is None:
        st.info("Execute primeiro o modelo base ou a otimizacao genetica.")
    else:
        amostra_idx = st.selectbox("Amostra do conjunto de teste", options=list(range(len(dataset.X_teste))), index=0)
        linha = dataset.X_teste.iloc[amostra_idx].copy()
        edited_values: dict[str, float] = {}
        cols = st.columns(2)
        for idx, coluna in enumerate(dataset.X_teste.columns[:8]):
            with cols[idx % 2]:
                edited_values[coluna] = st.number_input(coluna, value=float(linha[coluna]))
        for coluna, valor in edited_values.items():
            linha[coluna] = valor
        amostra_df = pd.DataFrame([linha])
        modelo = criar_modelo_otimizado_por_nome(modelo_resultado.modelo, modelo_resultado.parametros, settings.random_seed)
        modelo.fit(dataset.X_treino, dataset.y_treino)
        predicao, probabilidade = prever_amostra(modelo, amostra_df)
        classificacao = "maligno" if int(predicao[0]) == settings.positive_label else "benigno"
        st.metric("Classificacao prevista", classificacao)
        st.metric("Probabilidade de malignidade", f"{float(probabilidade[0]):.2%}")
        st.session_state.current_prediction = {
            "classificacao": classificacao,
            "probabilidade": float(probabilidade[0]),
            "metrics": (best_result or baseline_state).metrics_teste,
        }

with tab_llm:
    st.subheader("Explicacoes e historico")
    current_prediction = st.session_state.get("current_prediction")
    if current_prediction is None:
        st.info("Gere uma predicao na aba anterior para habilitar a explicacao.")
    else:
        contexto = st.text_area(
            "Contexto clinico",
            value="Paciente em avaliacao complementar de lesao mamaria.",
            height=120,
        )
        if st.button("Gerar explicacao com LLM/mock"):
            service = DiagnosticExplanationService(build_llm_client(llm_settings), llm_settings.output_path)
            explicacao = service.generate_explanation(
                classificacao=current_prediction["classificacao"],
                probabilidade=current_prediction["probabilidade"],
                metrics=current_prediction["metrics"],
                contexto_clinico=contexto,
            )
            st.text_area("Explicacao gerada", explicacao.resposta, height=220)

    history = load_explanation_history(llm_settings.output_path)
    if history:
        st.markdown("**Historico de respostas da LLM**")
        history_df = llm_history_to_dataframe(history)
        st.dataframe(history_df, width="stretch")
    else:
        st.caption("Nenhuma resposta registrada ainda.")

with tab_monitoramento:
    st.subheader("Logs e monitoramento")
    log_file = settings.log_output_dir / "aplicacao.log"
    st.write({"arquivo_log": str(log_file), "jsonl_llm": str(llm_settings.output_path)})
    if log_file.exists():
        content = log_file.read_text(encoding="utf-8")
        st.text_area("Ultimas linhas do log", "\n".join(content.splitlines()[-80:]), height=320)
    else:
        st.caption("Log ainda nao gerado.")
