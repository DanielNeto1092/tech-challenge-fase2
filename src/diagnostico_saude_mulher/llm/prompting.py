from __future__ import annotations

from diagnostico_saude_mulher.models.schemas import ClassificationMetrics


def construir_system_prompt() -> str:
    """Prompt de sistema com as restrições éticas do domínio."""

    return (
        "Voce e uma assistente para apoio a diagnostico em saude da mulher. "
        "Escreva em portugues, com linguagem sensivel a genero, cautelosa e apropriada para profissionais de saude. "
        "Nao afirme diagnostico definitivo. Explique limites do modelo e sempre reforce que a decisao depende de avaliacao medica."
    )


def construir_user_prompt(
    classificacao: str,
    probabilidade: float,
    metrics: ClassificationMetrics,
    contexto_clinico: str,
) -> str:
    """Gera o prompt do usuário para interpretação do resultado do modelo."""

    return (
        "Gere uma explicacao clinica objetiva para profissional de saude.\n"
        f"Classificacao prevista: {classificacao}.\n"
        f"Probabilidade estimada de malignidade: {probabilidade:.2%}.\n"
        f"Recall do modelo: {metrics.recall:.2%}.\n"
        f"Especificidade do modelo: {metrics.specificity:.2%}.\n"
        f"F1-score do modelo: {metrics.f1_score:.2%}.\n"
        f"Contexto clinico: {contexto_clinico}.\n"
        "A resposta deve conter:\n"
        "- interpretacao cautelosa;\n"
        "- orientacoes praticas para a equipe clinica;\n"
        "- linguagem sensivel a genero;\n"
        "- aviso de que o sistema nao substitui avaliacao medica."
    )

