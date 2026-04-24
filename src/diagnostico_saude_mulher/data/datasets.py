from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from diagnostico_saude_mulher.config.settings import AppSettings


@dataclass(frozen=True)
class DatasetBundle:
    """Representa o dataset de treino e teste já preparado."""

    nome: str
    descricao: str
    atributos: pd.DataFrame
    alvo: pd.Series
    X_treino: pd.DataFrame
    X_teste: pd.DataFrame
    y_treino: pd.Series
    y_teste: pd.Series


def carregar_dataset_cancer_mama(settings: AppSettings) -> DatasetBundle:
    """Carrega o dataset Breast Cancer Wisconsin e prepara treino/teste.

    O alvo original do scikit-learn é 0=maligno, 1=benigno.
    Aqui invertemos para 1=maligno, 0=benigno, pois o recall do caso
    positivo deve priorizar a sensibilidade a casos malignos.
    """

    raw = load_breast_cancer(as_frame=True)
    atributos = raw.data.copy()
    alvo = (1 - raw.target).rename(settings.target_name)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        atributos,
        alvo,
        test_size=settings.test_size,
        stratify=alvo,
        random_state=settings.random_seed,
    )

    return DatasetBundle(
        nome="Breast Cancer Wisconsin Diagnostic",
        descricao=raw.DESCR.splitlines()[0],
        atributos=atributos,
        alvo=alvo,
        X_treino=X_treino,
        X_teste=X_teste,
        y_treino=y_treino,
        y_teste=y_teste,
    )

