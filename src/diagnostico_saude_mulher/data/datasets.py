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


def _split_dataset(atributos: pd.DataFrame, alvo: pd.Series, settings: AppSettings, nome: str, descricao: str) -> DatasetBundle:
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        atributos,
        alvo,
        test_size=settings.test_size,
        stratify=alvo,
        random_state=settings.random_seed,
    )
    return DatasetBundle(
        nome=nome,
        descricao=descricao,
        atributos=atributos,
        alvo=alvo,
        X_treino=X_treino,
        X_teste=X_teste,
        y_treino=y_treino,
        y_teste=y_teste,
    )


def carregar_dataset_cancer_mama(settings: AppSettings) -> DatasetBundle:
    """Carrega o dataset Breast Cancer Wisconsin e prepara treino/teste.

    O alvo original do scikit-learn é 0=maligno, 1=benigno.
    Aqui invertemos para 1=maligno, 0=benigno, pois o recall do caso
    positivo deve priorizar a sensibilidade a casos malignos.
    """

    raw = load_breast_cancer(as_frame=True)
    atributos = raw.data.copy()
    alvo = (1 - raw.target).rename(settings.target_name)
    return _split_dataset(
        atributos,
        alvo,
        settings,
        nome="Breast Cancer Wisconsin Diagnostic",
        descricao=raw.DESCR.splitlines()[0],
    )


def construir_dataset_de_dataframe(dataframe: pd.DataFrame, settings: AppSettings, nome: str = "Dataset carregado") -> DatasetBundle:
    """Constroi um DatasetBundle a partir de um CSV enviado pela interface."""

    target_candidates = [settings.target_name, "target", "Target", "Outcome", "diagnosis"]
    target_column = next((column for column in target_candidates if column in dataframe.columns), None)
    if target_column is None:
        raise ValueError(
            "O dataset enviado precisa ter uma coluna alvo. Use uma destas: "
            f"{', '.join(target_candidates)}."
        )

    atributos = dataframe.drop(columns=[target_column]).copy()
    alvo = dataframe[target_column].copy()
    if alvo.dtype == object:
        normalizado = alvo.astype(str).str.lower().str.strip()
        mapping = {"maligno": 1, "benigno": 0, "m": 1, "b": 0}
        if set(normalizado.unique()).issubset(set(mapping)):
            alvo = normalizado.map(mapping)
    alvo = alvo.astype(int).rename(settings.target_name)
    return _split_dataset(
        atributos=atributos,
        alvo=alvo,
        settings=settings,
        nome=nome,
        descricao="Dataset fornecido pela interface do usuario.",
    )
