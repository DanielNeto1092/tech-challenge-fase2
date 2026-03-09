

import random
import math
import copy 
from typing import List, Tuple, Optional

default_problems = {
5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
10:[(470, 169), (602, 202), (754, 239), (476, 233), (468, 301), (522, 29), (597, 171), (487, 325), (746, 232), (558, 136)],
12:[(728, 67), (560, 160), (602, 312), (712, 148), (535, 340), (720, 354), (568, 300), (629, 260), (539, 46), (634, 343), (491, 135), (768, 161)],
15:[(512, 317), (741, 72), (552, 50), (772, 346), (637, 12), (589, 131), (732, 165), (605, 15), (730, 38), (576, 216), (589, 381), (711, 387), (563, 228), (494, 22), (787, 288)]
}

def generate_random_population(cities_location: List[Tuple[float, float]], population_size: int) -> List[List[Tuple[float, float]]]:
    """
    Generate a random population of routes for a given set of cities.

    Parameters:
    - cities_location (List[Tuple[float, float]]): A list of tuples representing the locations of cities,
      where each tuple contains the latitude and longitude.
    - population_size (int): The size of the population, i.e., the number of routes to generate.

    Returns:
    List[List[Tuple[float, float]]]: A list of routes, where each route is represented as a list of city locations.
    """
    return [random.sample(cities_location, len(cities_location)) for _ in range(population_size)]


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1 (Tuple[float, float]): The coordinates of the first point.
    - point2 (Tuple[float, float]): The coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """
    Calculate the fitness of a given path based on the total Euclidean distance.

    Parameters:
    - path (List[Tuple[float, float]]): A list of tuples representing the path,
      where each tuple contains the coordinates of a point.

    Returns:
    float: The total Euclidean distance of the path.
    """
    distance = 0
    n = len(path)
    for i in range(n):
        distance += calculate_distance(path[i], path[(i + 1) % n])

    return distance


def order_crossover(parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
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
    child: List[Optional[Tuple[float, float]]] = [None] * length

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
    return child  # type: ignore

### demonstration: crossover test code
# Example usage:
# parent1 = [(1, 1), (2, 2), (3, 3), (4,4), (5,5), (6, 6)]
# parent2 = [(6, 6), (5, 5), (4, 4), (3, 3),  (2, 2), (1, 1)]

# # parent1 = [1, 2, 3, 4, 5, 6]
# # parent2 = [6, 5, 4, 3, 2, 1]


# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# # Example usage:
# population = generate_random_population(5, 10)

# print(calculate_fitness(population[0]))


# population = [(random.randint(0, 100), random.randint(0, 100))
#           for _ in range(3)]



# TODO: implement a mutation_intensity and invert pieces of code instead of just swamping two. 
def mutate(solution:  List[Tuple[float, float]], mutation_probability: float) ->  List[Tuple[float, float]]:
    """
    Mutate a solution by inverting a segment of the sequence with a given mutation probability.

    Parameters:
    - solution (List[int]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation for each individual in the solution.

    Returns:
    List[int]: The mutated solution sequence.
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
    
    # Inverte o segmento entre start_index e end_index (inclusive).
    if start_index != end_index:
        mutated_solution[start_index:end_index + 1] = reversed(mutated_solution[start_index:end_index + 1])
    
    return mutated_solution

### Demonstration: mutation test code    
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(population: List[List[Tuple[float, float]]], fitness: List[float]) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """
    Sort a population based on fitness values.

    Parameters:
    - population (List[List[Tuple[float, float]]]): The population of solutions, where each solution is represented as a list.
    - fitness (List[float]): The corresponding fitness values for each solution in the population.

    Returns:
    Tuple[List[List[Tuple[float, float]]], List[float]]: A tuple containing the sorted population and corresponding sorted fitness values.
    """
    # Combine lists into pairs
    combined_lists = list(zip(population, fitness))

    # Sort based on the values of the fitness list
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Separate the sorted pairs back into individual lists
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)

    return sorted_population, sorted_fitness


if __name__ == '__main__':
    N_CITIES = 10
    
    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3
    cities_locations = [(random.randint(0, 100), random.randint(0, 100))
              for _ in range(N_CITIES)]
    
    # CREATE INITIAL POPULATION
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # Lists to store best fitness and generation for plotting
    best_fitness_values = []
    best_solutions = []
    
    for generation in range(N_GENERATIONS):
  
        
        population_fitness = [calculate_fitness(individual) for individual in population]    
        
        population, population_fitness = sort_population(population,  population_fitness)
        
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]
           
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)    

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        new_population = [population[0]]  # Keep the best individual: ELITISM
        
        while len(new_population) < POPULATION_SIZE:
            
            # SELECTION
            parent1, parent2 = random.choices(population[:10], k=2)  # Select parents from the top 10 individuals
            
            # CROSSOVER
            child1 = order_crossover(parent1, parent2)
            
            ## MUTATION
            child1 = mutate(child1, MUTATION_PROBABILITY)
            
            new_population.append(child1)
            
    
        print('generation: ', generation)
        population = new_population
    
def generate_population_with_heuristics(cities_location: List[Tuple[float, float]], 
                                       population_size: int, 
                                       heuristic_ratio: float = 0.2) -> List[List[Tuple[float, float]]]:
    """
    Gera uma população híbrida combinando soluções baseadas em heurísticas e soluções aleatórias.
    
    Esta abordagem melhora a qualidade da população inicial ao incluir algumas boas soluções vindas de heurísticas, 
    enquanto mantém a diversidade com soluções aleatórias.

    Parameters:
    - cities_location (List[Tuple[float, float]]): Lista de coordenadas das cidades.
    - population_size (int): Tamanho total da população.
    - heuristic_ratio (float): Fração da população a ser gerada usando heurísticas (0.0 a 1.0).
                               Default é 0.2 (20% heurística, 80% aleatória).

    Returns:
    List[List[Tuple[float, float]]]: Uma população híbrida de rotas.
    """
    population = []
    
    # Calcular quantas soluções devem usar heurísticas
    n_heuristic = max(1, int(population_size * heuristic_ratio))
    n_random = population_size - n_heuristic
    
    # Gerar soluções baseadas em heurísticas (Nearest Neighbour a partir de diferentes cidades iniciais)
    for _ in range(n_heuristic):
        # Usar diferentes cidades iniciais para obter variedade nas soluções heurísticas
        tour = nearest_neighbour(cities_location)
        population.append(tour)
    
    # Preencher o restante com soluções aleatórias para diversidade
    population.extend(generate_random_population(cities_location, n_random))
    
    return population

def nearest_neighbour(cities_location: List[Tuple[float, float]], start_city: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
    """
    Gera uma rota utilizando a heurística do Vizinho Mais Próximo.
    
    Este algoritmo guloso começa a partir de uma cidade e sempre se move para a cidade não visitada mais próxima.
    Embora não seja ótimo, geralmente produz boas soluções iniciais para o TSP.

    Parameters:
    - cities_location (List[Tuple[float, float]]): Lista de coordenadas das cidades.
    - start_city (Optional[Tuple[float, float]]): Cidade inicial. Se None, escolhe aleatoriamente.

    Returns:
    List[Tuple[float, float]]: Uma rota gerada utilizando a heurística do Vizinho Mais Próximo.
    """
    if not cities_location:
        return []
    
    # Escolher cidade inicial
    if start_city is None:
        current_city = random.choice(cities_location)
    else:
        current_city = start_city
    
    tour = [current_city]
    unvisited = set(cities_location) - {current_city}
    
    # Construir rota sempre indo para a cidade não visitada mais próxima
    while unvisited:
        nearest_city = min(unvisited, key=lambda city: calculate_distance(current_city, city))
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
    
    return tour

