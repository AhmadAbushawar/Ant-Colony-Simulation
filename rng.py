import math
import time


class ManualRNG:
    def __init__(self, seed=None):
        # If no seed is provided, generate one using the current system time.
        if seed is None:
            seed = int(time.time() * 1000) % 2147483647
        self.m = 2147483647  # modulus (a large prime)
        self.a = 16807  # multiplier
        self.c = 0  # increment
        self.state = seed

    def random(self):
        # Returns a pseudo-random float in the range [0, 1)
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m

    def uniform(self, a, b):
        # Returns a pseudo-random float in the range [a, b)
        return a + (b - a) * self.random()

    def randint(self, a, b):
        # Returns a pseudo-random integer in the range [a, b]
        return a + int(self.random() * (b - a + 1))

    def normal(self, mean=0, std=1):
        # Generate a normally distributed random number using the Box-Muller transform.
        u1 = self.random()
        u2 = self.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mean + std * z0
