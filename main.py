import pygame
import numpy as np
from simulation import Map, Food, Colony, render_pheromones, draw_food

pygame.init()
clock = pygame.time.Clock()

# Display configuration.
WIDTH, HEIGHT = 600, 400
CELL_SIZE = 4
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

DIRT_COLOR = (160, 82, 45)
WHITE = (255, 255, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ant Colony Simulator - Modular Version")
font = pygame.font.SysFont("Arial", 20)


def main():
    dt = 0.45  # Initial time step.
    sim_time = 0.0  # Simulation elapsed time.

    # Initialize pheromone maps.
    pher_home = Map(GRID_WIDTH, GRID_HEIGHT)
    pher_food = Map(GRID_WIDTH, GRID_HEIGHT)

    # Create the ant colony at the center.
    colony = Colony(WIDTH // 2, HEIGHT // 2, 100, pher_home, pher_food)

    # Initialize food distribution.
    food = Food(GRID_WIDTH, GRID_HEIGHT)
    food.add_food(400, 300)
    food.add_food(150, 250)
    food.add_food(300, 100)
    food.add_food(500, 200)

    delivered_printed = False
    running = True
    while running:
        clock.tick(60)
        sim_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Add food via mouse click.
                x, y = pygame.mouse.get_pos()
                food.add_food(x, y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    dt *= 1.1
                elif event.key == pygame.K_DOWN:
                    dt /= 1.1

        screen.fill(DIRT_COLOR)
        render_pheromones(screen, pher_home, pher_food, CELL_SIZE)
        draw_food(screen, food, CELL_SIZE)
        colony.update(food, dt)
        colony.resolve_collisions()
        colony.draw(screen)

        # Display simulation statistics.
        delivered_text = font.render(
            f"Food Delivered: {colony.food_delivered}", True, WHITE
        )
        time_text = font.render(f"Time Elapsed: {sim_time:.2f}", True, WHITE)
        dt_text = font.render(f"dt: {dt:.3f}", True, WHITE)
        screen.blit(delivered_text, (10, 10))
        screen.blit(time_text, (10, 30))
        screen.blit(dt_text, (10, 50))

        # Check if all food is delivered and no ant carries food.
        if np.sum(food.map_vals) == 0 and all(not ant.has_food for ant in colony.ants):
            if not delivered_printed:
                print(f"Total Food Delivered: {colony.food_delivered}")
                print(f"Total Time Elapsed: {sim_time:.2f} seconds")
                delivered_printed = True

        pher_home.step(dt)
        pher_food.step(dt)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
