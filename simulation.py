import pygame
import numpy as np
from scipy.integrate import ode
import math
from rng import ManualRNG

rng = ManualRNG()


class Map:
    def __init__(self, grid_width, grid_height, max_val=100, evaporation_rate=0.999):
        self.width = grid_width
        self.height = grid_height
        self.max_val = max_val  # Maximum pheromone value for normalization
        self.evaporation_rate = evaporation_rate  # Determines decay per time unit
        self.map_vals = np.zeros((self.height, self.width), dtype=np.float32)

    def step(self, dt):
        # Update the pheromone map over a time step dt using an ODE solver.
        # The decay is modeled as: dP/dt = -k * P, with k = -ln(evaporation_rate)
        k = -np.log(self.evaporation_rate)
        y0 = self.map_vals.flatten()  # Flatten to 1D

        def dP_dt(t, y):
            return -k * y

        solver = ode(dP_dt)
        solver.set_integrator("dop853")
        solver.set_initial_value(y0, 0)
        solver.integrate(dt)
        self.map_vals = solver.y.reshape(self.map_vals.shape)

    def set_value(self, x, y, val):
        gx = x // 4
        gy = y // 4
        if 0 <= gx < self.width and 0 <= gy < self.height:
            if val > self.map_vals[gy, gx]:
                self.map_vals[gy, gx] = val

    def get_value(self, x, y):
        gx = x // 4
        gy = y // 4
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.map_vals[gy, gx]
        return -1

    def get_weighted_direction(self, x, y):
        gx = x // 4
        gy = y // 4
        sum_dx = 0.0
        sum_dy = 0.0
        total = 0.0
        directions = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]
        for dx, dy in directions:
            nx = gx + dx
            ny = gy + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                value = self.map_vals[ny, nx]
                sum_dx += dx * value
                sum_dy += dy * value
                total += value
        if total > 0:
            return [sum_dx / total, sum_dy / total]
        else:
            return [0, 0]


class Food:
    def __init__(self, grid_width, grid_height):
        self.width = grid_width
        self.height = grid_height
        # Boolean grid: True indicates food presence.
        self.map_vals = np.zeros((self.height, self.width), dtype=bool)

    def add_food(self, x, y):
        gx = x // 4
        gy = y // 4
        # Add food in a 5x5 block.
        for i in range(gy, min(gy + 5, self.height)):
            for j in range(gx, min(gx + 5, self.width)):
                self.map_vals[i, j] = True

    def bite(self, x, y):
        gx = x // 4
        gy = y // 4
        if 0 <= gx < self.width and 0 <= gy < self.height:
            self.map_vals[gy, gx] = False

    def get_value(self, x, y):
        gx = x // 4
        gy = y // 4
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.map_vals[gy, gx]
        return False


class Ant:
    def __init__(self, x, y, home_map, food_map):
        # Initialize velocity with Gaussian noise.
        self.dx = rng.normal(0, 1)
        self.dy = rng.normal(0, 1)
        self.x = float(x)
        self.y = float(y)
        self.int_x = x
        self.int_y = y
        self.last_x = x
        self.last_y = y
        self.has_food = False
        self.home_pher = 100
        self.food_pher = 100
        self.use_rate = 0.995
        self.wander_chance = 0.92
        self.bored = 0
        self.home_map = home_map
        self.food_map = food_map

    def step(self, dt):
        # Use Gaussian noise to update velocity.
        if rng.random() > self.wander_chance:
            self.dx += rng.normal(0, 1) * dt
        if rng.random() > self.wander_chance:
            self.dy += rng.normal(0, 1) * dt
        if rng.random() > 0.99:
            self.bored += rng.randint(0, 15)
        if self.bored > 0:
            self.bored -= 1
        else:
            if self.has_food:
                direction = self.home_map.get_weighted_direction(self.int_x, self.int_y)
                self.dx += direction[0] * rng.uniform(0, 1.5) * dt
                self.dy += direction[1] * rng.uniform(0, 1.5) * dt
            else:
                direction = self.food_map.get_weighted_direction(self.int_x, self.int_y)
                self.dx += direction[0] * rng.uniform(0, 1.5) * dt
                self.dy += direction[1] * rng.uniform(0, 1.5) * dt

        # Boundary conditions.
        if self.x < 2:
            self.dx = 1
        if self.x > 600 - 2:
            self.dx = -1
        if self.y < 2:
            self.dy = 1
        if self.y > 400 - 2:
            self.dy = -1

        self.dx = max(min(self.dx, 1), -1)
        self.dy = max(min(self.dy, 1), -1)

        # Update position.
        self.x += self.dx * dt
        self.y += self.dy * dt
        self.int_x = int(self.x)
        self.int_y = int(self.y)

        # Deposit pheromones if cell changed.
        if self.last_x != self.int_x or self.last_y != self.int_y:
            if self.has_food:
                self.food_pher *= self.use_rate
                self.food_map.set_value(self.int_x, self.int_y, self.food_pher)
            else:
                self.home_pher *= self.use_rate
                self.home_map.set_value(self.int_x, self.int_y, self.home_pher)
        self.last_x = self.int_x
        self.last_y = self.int_y


class Colony:
    def __init__(self, x, y, count, home_map, food_map):
        self.ants = [Ant(x, y, home_map, food_map) for _ in range(count)]
        self.x = x
        self.y = y
        self.food_delivered = 0

    def update(self, food, dt):
        for ant in self.ants:
            ant.step(dt)
            # If ant with food reaches nest region (within 20 pixels)
            if ant.has_food:
                if abs(ant.int_x - self.x) < 20 and abs(ant.int_y - self.y) < 20:
                    ant.has_food = False
                    ant.home_pher = 100
                    self.food_delivered += 1
            else:
                if food.get_value(ant.int_x, ant.int_y):
                    ant.has_food = True
                    ant.food_pher = 100
                    food.bite(ant.int_x, ant.int_y)

    def resolve_collisions(self):
        # Check pairwise collisions; ants are circles with radius 4.
        collision_distance = 8
        n = len(self.ants)
        for i in range(n):
            for j in range(i + 1, n):
                ant1 = self.ants[i]
                ant2 = self.ants[j]
                dx = ant1.x - ant2.x
                dy = ant1.y - ant2.y
                dist = math.hypot(dx, dy)
                if dist < collision_distance:
                    if dist == 0:
                        angle = rng.uniform(0, 2 * math.pi)
                        normal = [math.cos(angle), math.sin(angle)]
                        dist = 0.1
                    else:
                        normal = [dx / dist, dy / dist]
                    penetration = collision_distance - dist
                    correction = 0.5 * penetration
                    ant1.x += correction * normal[0]
                    ant1.y += correction * normal[1]
                    ant2.x -= correction * normal[0]
                    ant2.y -= correction * normal[1]
                    rel_vx = ant1.dx - ant2.dx
                    rel_vy = ant1.dy - ant2.dy
                    rel_dot = rel_vx * normal[0] + rel_vy * normal[1]
                    if rel_dot < 0:
                        e = 0.5  # Coefficient of restitution
                        impulse = -(1 + e) * rel_dot / 2  # Equal mass assumption
                        ant1.dx += impulse * normal[0]
                        ant1.dy += impulse * normal[1]
                        ant2.dx -= impulse * normal[0]
                        ant2.dy -= impulse * normal[1]

    def draw(self, surface):
        for ant in self.ants:
            color = (218, 165, 32) if ant.has_food else (50, 30, 20)
            if abs(ant.dx) > abs(ant.dy):
                pygame.draw.rect(surface, color, (int(ant.x), int(ant.y), 6, 4))
            else:
                pygame.draw.rect(surface, color, (int(ant.x), int(ant.y), 4, 6))


def render_pheromones(surface, home_map, food_map, cell_size=4):
    pheromone_surface = pygame.Surface(
        (surface.get_width(), surface.get_height()), pygame.SRCALPHA
    )
    for gy in range(home_map.height):
        for gx in range(home_map.width):
            x_pixel = gx * cell_size
            y_pixel = gy * cell_size
            home_val = home_map.map_vals[gy, gx]
            food_val = food_map.map_vals[gy, gx]
            if home_val > 0 or food_val > 0:
                home_alpha = (home_val / home_map.max_val) * 255
                food_alpha = (food_val / home_map.max_val) * 255
                pixel_r = int(80 * (home_alpha / 255) + 160 * (1 - home_alpha / 255))
                pixel_g = int(70 * (home_alpha / 255) + 82 * (1 - home_alpha / 255))
                pixel_b = int(60 * (home_alpha / 255) + 45 * (1 - home_alpha / 255))
                pixel_r = int(
                    255 * (food_alpha / 255) + pixel_r * (1 - food_alpha / 255)
                )
                pixel_g = int(
                    255 * (food_alpha / 255) + pixel_g * (1 - food_alpha / 255)
                )
                pixel_b = int(
                    255 * (food_alpha / 255) + pixel_b * (1 - food_alpha / 255)
                )
                pygame.draw.rect(
                    pheromone_surface,
                    (pixel_r, pixel_g, pixel_b, 255),
                    (x_pixel, y_pixel, cell_size, cell_size),
                )
    surface.blit(pheromone_surface, (0, 0))


def draw_food(surface, food, cell_size=4):
    for gy in range(food.height):
        for gx in range(food.width):
            if food.map_vals[gy, gx]:
                x_pixel = gx * cell_size
                y_pixel = gy * cell_size
                pygame.draw.rect(
                    surface, (218, 165, 32), (x_pixel, y_pixel, cell_size, cell_size)
                )
