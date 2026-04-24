de# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:19:37 2023

@author: SérgioPolimante
"""

import random
from typing import Optional, Tuple, Sequence

def order_crossover(parent1: Sequence[Tuple[float, float]], parent2: Sequence[Tuple[float, float]]) -> list[Tuple[float, float]]:
    """
    Perform order crossover (OX) between two parent sequences to create a child sequence.

    Parameters:
    - parent1 (List[Tuple[float, float]]): The first parent sequence.
    - parent2 (List[Tuple[float, float]]): The second parent sequence.

    Returns:
    List[Tuple[float, float]]: The child sequence resulting from the order crossover.
    """
    length = len(parent1)

    # Choose two random indices for the crossover
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Inicializa o filho com valores None para indicar posições não preenchidas
    child: list[Optional[Tuple[float, float]]] = [None] * length

    # Copia o segmento de parent1 para o filho nas mesmas posições
    for i in range(start_index, end_index):
        child[i] = parent1[i]

    # Obtém os genes de parent2 que ainda não estão no filho, preservando sua ordem
    remaining_genes = [gene for gene in parent2 if gene not in child]

    # Preenche as posições restantes, começando de end_index e retornando ao início se necessário
    child_index = end_index
    for gene in remaining_genes:
        # Retorna para o início se chegarmos ao fim
        if child_index >= length:
            child_index = 0
        # Pula posições que já estão preenchidas
        while child[child_index] is not None:
            child_index = (child_index + 1) % length
        # Coloca o gene
        child[child_index] = gene
        child_index = (child_index + 1) % length

    # Todos os valores None foram substituídos neste ponto
    return child


# # Example usage:
# parent1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# parent2 = [9, 3, 7, 8, 2, 6, 5, 1, 4]

# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# Example usage:
parent1 = [(1, 1), (2, 2), (3, 3), (4,4), (5,5), (6, 6)]
parent2 = [(6, 6), (5, 5), (4, 4), (3, 3),  (2, 2), (1, 1)]

# parent1 = [1, 2, 3, 4, 5, 6]
# parent2 = [6, 5, 4, 3, 2, 1]


child = order_crossover(parent1, parent2)
print("Parent 1:", parent1)
print("Parent 2:", parent2)
print("Child   :", child)


