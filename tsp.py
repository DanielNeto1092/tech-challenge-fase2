import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import mutate, order_crossover, generate_random_population, calculate_fitness, sort_population, default_problems, generate_population_with_heuristics, nearest_neighbour
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import numpy as np
from benchmark_att48 import *
import json
import os


# Define constant values
# pygame
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450
HEADER_HEIGHT = 70  # Espaço para o título
FOOTER_HEIGHT = 90  # Espaço para as informações do rodapé

# GA
N_CITIES = 15
POPULATION_SIZE = 100
N_GENERATIONS = None
MUTATION_PROBABILITY = 0.5

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# Initialize problem
# Using Random cities generation
# cities_locations = [(random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS), random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
#                     for _ in range(N_CITIES)]


# # # Using Deault Problems: 10, 12 or 15
# WIDTH, HEIGHT = 800, 400
# cities_locations = default_problems[15]


# Using att48 benchmark
WIDTH, HEIGHT = 1500, 800
att_cities_locations = np.array(att_48_cities_locations)
max_x = max(point[0] for point in att_cities_locations)
max_y = max(point[1] for point in att_cities_locations)

# Adicionar margens para evitar cidades nas bordas
MARGIN_RIGHT = NODE_RADIUS * 2
MARGIN_TOP = NODE_RADIUS * 2
MARGIN_BOTTOM = NODE_RADIUS * 2

# Ajustar escala considerando margens
available_width = WIDTH - PLOT_X_OFFSET - MARGIN_RIGHT
available_height = HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

scale_x = available_width / max_x
scale_y = available_height / max_y

cities_locations = [(float(point[0] * scale_x + PLOT_X_OFFSET),
                     float(point[1] * scale_y + HEADER_HEIGHT + MARGIN_TOP)) for point in att_cities_locations]
target_solution = [cities_locations[i-1] for i in att_48_cities_order]
fitness_target_solution = calculate_fitness(target_solution)
print(f"Best Solution: {fitness_target_solution}")
# ----- Using att48 benchmark


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 28)
title_font = pygame.font.Font(None, 30)  # Fonte maior para o título
generation_counter = itertools.count(start=1)  # Start the counter at 1
start_time = pygame.time.get_ticks()

# Create Initial Population
# population = generate_random_population(cities_locations, POPULATION_SIZE)
# Utilizando abordagem híbrida: 50% baseada em heurística (Vizinho Mais Próximo), 50% aleatória para diversidade.
population = generate_population_with_heuristics(cities_locations, POPULATION_SIZE, heuristic_ratio=0.5)
best_fitness_values = []
best_solutions = []

# Rastrear tempo desde última melhoria
last_improvement_time = start_time
previous_best_fitness = float('inf')
time_to_last_improvement = 0  # Tempo que levou para encontrar a última melhoria
best_fitness_generation = 0
best_fitness_time = "00:00:00"
best_fitness_took = "00:00"

# Arquivo para salvar o melhor resultado
BEST_RESULT_FILE = "best_solution.json"

# Carregar melhor resultado anterior (se existir)
global_best_fitness = float('inf')
if os.path.exists(BEST_RESULT_FILE):
    try:
        with open(BEST_RESULT_FILE, 'r') as f:
            saved_data = json.load(f)
            global_best_fitness = saved_data.get('fitness', float('inf'))
            print(f"Loaded previous best: {global_best_fitness}")
    except Exception as e:
        print(f"Could not load previous best: {e}")


def save_screenshot(screen):
    """Captura e salva a tela do pygame."""
    os.makedirs("screenshots", exist_ok=True)
    timestamp = pygame.time.get_ticks()
    screenshot_filename = f"screenshots/screenshot_{timestamp}.png"
    pygame.image.save(screen, screenshot_filename)
    print(f"Screenshot saved: {screenshot_filename}")


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            save_screenshot(screen)
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                save_screenshot(screen)
                running = False

    elapsed_time = (pygame.time.get_ticks() - start_time) / 1000  # Time in seconds

    # Converter para horas, minutos e segundos
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    generation = next(generation_counter)

    screen.fill(WHITE)

    # Desenhar título centralizado no topo
    title_text = title_font.render("Mudança 3: Ajuste Cruzamento, substituição de child.insert() e Mutação Invertida; Probabilidade de Mutação = 0.5; Inicialização: Nearest Neighbour", True, BLACK)
    title_rect = title_text.get_rect(center=(WIDTH // 2, HEADER_HEIGHT // 2))
    screen.blit(title_text, title_rect)

    population_fitness = [calculate_fitness(
        individual) for individual in population]

    population, population_fitness = sort_population(
        population,  population_fitness)

    best_fitness = calculate_fitness(population[0])
    best_solution = population[0]

    # Verificar se houve melhoria
    if best_fitness < previous_best_fitness:
        current_time = pygame.time.get_ticks()
        # Calcular quanto tempo levou para encontrar esta melhoria
        time_to_last_improvement = (current_time - last_improvement_time) / 1000
        # Atualizar tempo da última melhoria
        last_improvement_time = current_time
        previous_best_fitness = best_fitness

        # Tempo total de execução quando o novo best foi encontrado
        best_elapsed_time = (current_time - start_time) / 1000
        best_hours = int(best_elapsed_time // 3600)
        best_minutes = int((best_elapsed_time % 3600) // 60)
        best_seconds = int(best_elapsed_time % 60)

        # Tempo que levou para encontrar esta melhoria
        took_minutes = int(time_to_last_improvement // 60)
        took_seconds = int(time_to_last_improvement % 60)

        best_fitness_generation = generation
        best_fitness_time = f"{best_hours:02d}:{best_minutes:02d}:{best_seconds:02d}"
        best_fitness_took = f"{took_minutes:02d}:{took_seconds:02d}"

        print(
            f"Generation {generation}: Best fitness = {round(best_fitness, 2)}; "
            f"Time: {best_hours:02d}:{best_minutes:02d}:{best_seconds:02d}; "
            f"Discovery Time: {took_minutes:02d}:{took_seconds:02d}"
        )
        
        # Salvar se for melhor que o recorde global
        if best_fitness < global_best_fitness:
            global_best_fitness = best_fitness
            try:
                result_data = {
                    'fitness': float(best_fitness),
                    'generation': generation,
                    'time': best_fitness_time,
                    'took': best_fitness_took,
                    'solution': [(float(x), float(y)) for x, y in best_solution]
                }
                with open(BEST_RESULT_FILE, 'w') as f:
                    json.dump(result_data, f, indent=2)
                print(f"★ NEW RECORD SAVED: {round(best_fitness, 2)} ★")
            except Exception as e:
                print(f"Error saving best solution: {e}")
    
    # Calcular tempo desde última melhoria (reseta a cada melhoria)
    time_since_improvement = (pygame.time.get_ticks() - last_improvement_time) / 1000
    since_minutes = int(time_since_improvement // 60)
    since_seconds = int(time_since_improvement % 60)
    
    # Tempo que levou para encontrar a última melhoria (não reseta)
    took_minutes = int(time_to_last_improvement // 60)
    took_seconds = int(time_to_last_improvement % 60)

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distance (pxls)",
              position=(0, HEADER_HEIGHT))

    draw_cities(screen, [(int(x), int(y)) for x, y in cities_locations], RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)
    draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

    # Linha separadora abaixo do título (desenhada por cima para ficar visível)
    pygame.draw.line(screen, BLACK, (0, HEADER_HEIGHT), (WIDTH, HEADER_HEIGHT), 2)

    # Renderizar informações no rodapé em duas linhas
    footer_padding_bottom = 8
    footer_line_spacing = 4
    line_height = font.get_linesize()
    line_2_y = HEIGHT - footer_padding_bottom - line_height
    line_1_y = line_2_y - footer_line_spacing - line_height

    line_1 = (
        f"Time Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d} "
        f"Current Generation: {generation} "
        f"Since: {since_minutes:02d}:{since_seconds:02d}"
    )
    line_2 = (
        f"Generation: {best_fitness_generation}: "
        f"Best Fitness Distance = {round(previous_best_fitness, 2)}; "
        f"Time: {best_fitness_time}; "
        f"Discovery Time: {best_fitness_took}"
    )

    line_1_text = font.render(line_1, True, BLACK)
    line_2_text = font.render(line_2, True, RED)
    screen.blit(line_1_text, (10, line_1_y))
    screen.blit(line_2_text, (10, line_2_y))

    new_population = [population[0]]  # Keep the best individual: ELITISM

    while len(new_population) < POPULATION_SIZE:

        # selection
        # simple selection based on first 10 best solutions
        # parent1, parent2 = random.choices(population[:10], k=2)

        # solution based on fitness probability
        probability = 1 / np.array(population_fitness)
        parent1, parent2 = random.choices(population, weights=probability, k=2)

        child1 = order_crossover(parent1, parent2)
        # child1 = order_crossover(parent1, parent1)

        child1 = mutate(child1, MUTATION_PROBABILITY)

        new_population.append(child1)

    population = new_population

    pygame.display.flip()
    clock.tick(FPS)


# exit software
pygame.quit()
sys.exit()
