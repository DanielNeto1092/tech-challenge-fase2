# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:41:21 2023

@author: SérgioPolimante
"""

import random
import copy
from typing import List, Tuple

#### MUTATION ###
def mutate(solution: List[Tuple[float, float]], mutation_probability: float) -> List[Tuple[float, float]]:
    """
    Mutate a solution using inversion mutation with a given probability.
    
    Inversion mutation is highly effective for TSP: it selects two random positions
    and reverses the segment between them. This preserves most of the tour structure
    while potentially improving edge crossings.

    Parameters:
    - solution (List[Tuple[float, float]]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation occurring (0.0 to 1.0).

    Returns:
    List[Tuple[float, float]]: The mutated solution sequence.
    """
    # Check if mutation should occur
    if random.random() >= mutation_probability:
        return solution
    
    # Ensure there are at least two cities to perform inversion
    if len(solution) < 2:
        return solution
    
    # Create a copy of the solution
    mutated_solution = solution.copy()
    
    # Selecionar dois índices aleatórios para o segmento a ser invertido.
    index1 = random.randint(0, len(solution) - 1)
    index2 = random.randint(0, len(solution) - 1)
    
    # Garantir que index1 < index2
    start_index = min(index1, index2)
    end_index = max(index1, index2)
    
    # Inverta o segmento entre start_index e end_index (inclusive).
    if start_index != end_index:
        mutated_solution[start_index:end_index + 1] = reversed(mutated_solution[start_index:end_index + 1])
    
    return mutated_solution

        
        
    
# Example usage:
original_solution =[(99, 100), (2, 50), (1, 71), (3, 20), (4, 40), (5, 60), (6, 80), (7, 90)]
mutation_probability = 1  # 50% chance to mutate

mutated_solution = mutate(original_solution, mutation_probability)
print("Original Solution:", original_solution)
print("Mutated Solution:", mutated_solution)