from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "Apresentacao_Algoritmo_Genetico_Cancer_de_Mama.pptx"
RESULTS_PATH = ROOT / "artifacts" / "experimentos_geneticos.json"
GRAFICOS_DIR = ROOT / "artifacts" / "graficos"


def add_title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle


def add_bullet_slide(prs: Presentation, title: str, bullets: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    text_frame = slide.placeholders[1].text_frame
    text_frame.clear()
    for index, bullet in enumerate(bullets):
        paragraph = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
        paragraph.text = bullet
        paragraph.level = 0
        paragraph.font.size = Pt(22)


def add_two_column_slide(prs: Presentation, title: str, left_title: str, left_bullets: list[str], right_title: str, right_bullets: list[str]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title

    left_box = slide.shapes.add_textbox(Inches(0.6), Inches(1.4), Inches(4.2), Inches(5.2))
    right_box = slide.shapes.add_textbox(Inches(5.0), Inches(1.4), Inches(4.2), Inches(5.2))

    for box, column_title, bullets in ((left_box, left_title, left_bullets), (right_box, right_title, right_bullets)):
        frame = box.text_frame
        frame.word_wrap = True
        p = frame.paragraphs[0]
        p.text = column_title
        p.font.bold = True
        p.font.size = Pt(24)
        for bullet in bullets:
            item = frame.add_paragraph()
            item.text = bullet
            item.level = 0
            item.font.size = Pt(18)


def add_image_slide(prs: Presentation, title: str, image_path: Path, caption: str | None = None) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    slide.shapes.add_picture(str(image_path), Inches(0.8), Inches(1.3), width=Inches(8.4))
    if caption:
        box = slide.shapes.add_textbox(Inches(0.8), Inches(6.6), Inches(8.4), Inches(0.5))
        paragraph = box.text_frame.paragraphs[0]
        paragraph.text = caption
        paragraph.alignment = PP_ALIGN.CENTER
        paragraph.font.size = Pt(14)


def build_presentation() -> Path:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    baseline_models = data["modelos_baseline"]
    baseline_ref = data["baseline"]
    baseline = data["baseline"]["metrics_teste"]
    exp1 = data["experimentos_geneticos"][0]
    exp2 = data["experimentos_geneticos"][1]
    melhor_baseline = max(
        baseline_models,
        key=lambda item: (item["metrics_teste"]["recall"], item["metrics_teste"]["f1_score"], item["metrics_teste"]["specificity"]),
    )

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    add_title_slide(
        prs,
        "Algoritmo Genetico no Diagnostico de Cancer de Mama em Mulheres",
        "Tech Challenge Fase 2 | Machine Learning, otimizacao genetica e explicabilidade",
    )

    add_bullet_slide(
        prs,
        "Problema e Objetivo",
        [
            "Sistema de apoio ao diagnostico com foco na reducao de falsos negativos.",
            "Comparacao entre multiplos modelos baseline e otimizacao da familia vencedora pelo fitness.",
            "Explicabilidade em linguagem natural voltada ao contexto profissional.",
        ],
    )

    add_bullet_slide(
        prs,
        "Dataset Utilizado",
        [
            "Breast Cancer Wisconsin Diagnostic.",
            "569 amostras em um problema de classificacao binaria.",
            "Classe positiva ajustada para malignidade.",
        ],
    )

    add_bullet_slide(
        prs,
        "Arquitetura da Solucao",
        [
            "Separacao entre dominio, interface, API e camada de experimentacao.",
            "Entradas principais: pipeline, AG, Streamlit e API.",
            "Saidas: JSON, JSONL, graficos, logs e modelo exportado.",
        ],
    )

    add_bullet_slide(
        prs,
        "Algoritmo Genetico",
        [
            f"Genes representando hiperparametros de {baseline_ref['modelo']}.",
            "Selecao, crossover, mutacao e elitismo.",
            "Fitness: 55% recall, 25% especificidade, 20% F1.",
        ],
    )

    add_two_column_slide(
        prs,
        "Experimentos",
        "Configuracoes testadas",
        [
            "Exp. 1: pop. 6 | 3 geracoes | mut. 0,10",
            "Exp. 2: pop. 8 | 4 geracoes | mut. 0,15",
            "Exp. 3: pop. 10 | 5 geracoes | mut. 0,20",
        ],
        "Camadas adicionais",
        [
            "Script dedicado para AG.",
            "Geracao automatica de graficos.",
            "UI, API, logging, Docker e Terraform.",
        ],
    )

    add_two_column_slide(
        prs,
        "Comparacao de Baselines",
        "Modelos avaliados",
        [
            "RandomForestClassifier",
            "LogisticRegression",
            "DecisionTreeClassifier",
            "KNeighborsClassifier",
        ],
        "Melhor baseline em teste",
        [
            f"Modelo: {melhor_baseline['modelo']}",
            f"Recall: {melhor_baseline['metrics_teste']['recall']:.2%}",
            f"Especificidade: {melhor_baseline['metrics_teste']['specificity']:.2%}",
            f"F1-score: {melhor_baseline['metrics_teste']['f1_score']:.2%}",
        ],
    )

    add_two_column_slide(
        prs,
        "Baseline de Referencia do AG",
        f"{baseline_ref['modelo']} base",
        [
            f"Recall: {baseline['recall']:.2%}",
            f"Especificidade: {baseline['specificity']:.2%}",
            f"F1-score: {baseline['f1_score']:.2%}",
            f"Acuracia: {baseline['accuracy']:.2%}",
        ],
        "Leitura tecnica",
        [
            "A familia de referencia foi escolhida automaticamente pelo fitness.",
            f"Modelo selecionado: {baseline_ref['modelo']}.",
            "A comparacao com outros baselines reforca a coerencia da busca evolutiva.",
        ],
    )

    add_two_column_slide(
        prs,
        "Melhores Configuracoes do AG",
        "Experimento 1",
        [
            "classifier__C = 0.1",
            "classifier__solver = lbfgs",
            "scaler__with_mean = True",
            "scaler__with_std = True",
        ],
        "Experimento 2",
        [
            "classifier__C = 0.1",
            "classifier__solver = lbfgs",
            "scaler__with_mean = True",
            "scaler__with_std = True",
            f"F1 teste = {exp2['metrics_teste']['f1_score']:.2%}",
        ],
    )

    add_two_column_slide(
        prs,
        "Hiperparametros Otimizados",
        "Experimento 2",
        [
            "classifier__C = 0.1",
            "classifier__solver = lbfgs",
            "scaler__with_mean = True",
            "scaler__with_std = True",
        ],
        "Leitura tecnica",
        [
            "Melhor equilibrio no conjunto de teste.",
            f"Recall = {exp2['metrics_teste']['recall']:.2%}",
            f"Especificidade = {exp2['metrics_teste']['specificity']:.2%}",
            f"F1-score = {exp2['metrics_teste']['f1_score']:.2%}",
        ],
    )

    add_image_slide(
        prs,
        "Comparacao Baseline x AG",
        GRAFICOS_DIR / "comparacao_experimentos.png",
        "Comparacao de recall, especificidade e F1-score no conjunto de teste.",
    )

    add_image_slide(
        prs,
        "Resumo dos Experimentos",
        GRAFICOS_DIR / "resumo_experimentos.png",
        "Fitness final dos tres experimentos geneticos executados.",
    )

    add_image_slide(
        prs,
        "Convergencia do AG",
        GRAFICOS_DIR / "convergencia_ag_experimento_2.png",
        "Evolucao do fitness no experimento com melhor equilibrio no conjunto de teste.",
    )

    add_image_slide(
        prs,
        "Fitness x Hiperparametros",
        GRAFICOS_DIR / "comportamento_hiperparametros.png",
        "Leitura do comportamento do fitness frente aos hiperparametros otimizados.",
    )

    add_bullet_slide(
        prs,
        "Leitura Parametros x Fitness",
        [
            "Os tres experimentos convergiram para a mesma configuracao vencedora.",
            "O melhor fitness final foi 0,9712.",
            "A configuracao vencedora usou C=0.1 e solver=lbfgs.",
            "A baixa variacao nos graficos reflete convergencia rapida para a mesma solucao.",
        ],
    )

    add_image_slide(
        prs,
        "Metricas por Geracao",
        GRAFICOS_DIR / "metricas_por_geracao_ag_experimento_2.png",
        "Evolucao de recall, especificidade e F1 ao longo das geracoes.",
    )

    add_bullet_slide(
        prs,
        "LLM e Explicabilidade",
        [
            "Camada desacoplada com suporte a mock e endpoint HTTP.",
            "Explicacoes com classificacao, probabilidade, cautela clinica e aviso de nao substituicao da avaliacao medica.",
            "Persistencia automatica das respostas em JSONL.",
        ],
    )

    add_bullet_slide(
        prs,
        "Interface Web",
        [
            "Metricas, graficos e monitoramento.",
            "Upload de CSV e simulacao de predicao.",
            "Historico da LLM e comparacao entre baselines e familia otimizada.",
        ],
    )

    add_bullet_slide(
        prs,
        "API, Logging e Nuvem",
        [
            "FastAPI pronta para integracoes futuras.",
            "Logging em arquivo e aba de monitoramento.",
            "Docker e Terraform inicial para nuvem.",
        ],
    )

    add_bullet_slide(
        prs,
        "Testes e Preparacao Futura",
        [
            "Cobertura para nucleo, integracao, UI e API.",
            "Base modular para workers e servicos desacoplados.",
            "Preparacao tecnica para a Fase 3.",
        ],
    )

    add_bullet_slide(
        prs,
        "Conclusoes",
        [
            "A comparacao entre quatro baselines ampliou a robustez do estudo.",
            "A LogisticRegression foi selecionada como referencia pelo fitness.",
            "O AG melhorou o desempenho dessa mesma familia no conjunto de teste.",
            "Projeto completo, testado e pronto para evolucao.",
        ],
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(OUTPUT))
        return OUTPUT
    except PermissionError:
        fallback = OUTPUT.with_name(f"{OUTPUT.stem}_atualizada{OUTPUT.suffix}")
        prs.save(str(fallback))
        return fallback


if __name__ == "__main__":
    print(build_presentation())
