# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:03:11 2023

@author: SérgioPolimante
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pygame
from typing import List, Tuple


def draw_plot(
    screen: pygame.Surface,
    x: list,
    y: list,
    x_label: str = "Generation",
    y_label: str = "Fitness",
    position: tuple = (0, 0),
) -> None:
    """
    Draw a plot on a Pygame screen using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.plot(x, y)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    raw_data = canvas.buffer_rgba()
    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    screen.blit(surf, position)

    plt.close(fig)


def draw_cities(
    screen: pygame.Surface,
    cities_locations: List[Tuple[int, int]],
    rgb_color: Tuple[int, int, int],
    node_radius: int,
) -> None:
    """
    Draw circles representing cities on the given Pygame screen.
    """
    for city_location in cities_locations:
        pygame.draw.circle(screen, rgb_color, city_location, node_radius)


def draw_paths(
    screen: pygame.Surface,
    path: List[Tuple[int, int]],
    rgb_color: Tuple[int, int, int],
    width: int = 1,
) -> None:
    """
    Draw a path on a Pygame screen.
    """
    pygame.draw.lines(screen, rgb_color, True, path, width=width)


def draw_text(
    screen: pygame.Surface,
    text: str,
    color: pygame.Color,
    position: Tuple[int, int] = (10, 10),
    font_size: int = 15,
) -> None:
    """
    Draw text on a Pygame screen.
    """
    pygame.font.init()
    my_font = pygame.font.SysFont("Arial", font_size)
    text_surface = my_font.render(text, True, color)
    screen.blit(text_surface, position)