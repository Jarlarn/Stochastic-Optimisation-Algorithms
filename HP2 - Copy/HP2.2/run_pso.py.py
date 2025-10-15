import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()


def himmelblau(x, y):
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2


class ParticleSwarmOptimizer:
    def __init__(
        self,
        swarm_size,
        dimensions,
        position_min,
        position_max,
        velocity_max,
        inertia_weight,
        inertia_min,
        inertia_decay,
    ):
        self.swarm_size = swarm_size
        self.dimensions = dimensions
        self.position_min = np.asarray(position_min)
        self.position_max = np.asarray(position_max)
        self.velocity_max = velocity_max
        self.inertia_weight = inertia_weight
        self.inertia_min = inertia_min
        self.inertia_decay = inertia_decay
        random_positions = rng.uniform(0, 1, size=(swarm_size, dimensions))
        self.positions = self.position_min + random_positions * (
            self.position_max - self.position_min
        )
        self.velocities = -(
            (self.position_max - self.position_min) / 2
        ) + random_positions * (self.position_max - self.position_min)
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.array(self.evaluate_swarm(self.positions))
        best_particle_index = int(np.argmin(self.personal_best_values))
        self.global_best_position = self.personal_best_positions[
            best_particle_index
        ].copy()
        self.global_best_value = float(self.personal_best_values[best_particle_index])

    def evaluate_particle(self, position):
        x1, x2 = position[0], position[1]
        term1 = (x1**2 + x2 - 11) ** 2
        term2 = (x1 + x2**2 - 7) ** 2
        return term1 + term2

    def evaluate_swarm(self, positions):
        return np.array([self.evaluate_particle(pos) for pos in positions])

    def update_best_positions(self):
        for i in range(self.swarm_size):
            fitness = self.evaluate_particle(self.positions[i])
            if fitness < self.personal_best_values[i]:
                self.personal_best_values[i] = fitness
                self.personal_best_positions[i] = self.positions[i].copy()
        best_particle_index = np.argmin(self.personal_best_values)
        if self.personal_best_values[best_particle_index] < self.global_best_value:
            self.global_best_value = self.personal_best_values[best_particle_index]
            self.global_best_position = self.personal_best_positions[
                best_particle_index
            ].copy()

    def update_velocity_and_position(self, cognitive_coefficient, social_coefficient):
        for i in range(self.swarm_size):
            cognitive_random = rng.uniform(0, 1, size=self.dimensions)
            social_random = rng.uniform(0, 1, size=self.dimensions)
            cognitive_component = (
                cognitive_coefficient
                * cognitive_random
                * (self.personal_best_positions[i] - self.positions[i])
            )
            social_component = (
                social_coefficient
                * social_random
                * (self.global_best_position - self.positions[i])
            )
            self.velocities[i] = (
                self.inertia_weight * self.velocities[i]
                + cognitive_component
                + social_component
            )
            if self.velocity_max is not None:
                vmax = np.asarray(self.velocity_max)
                self.velocities[i] = np.clip(self.velocities[i], -vmax, vmax)
            self.positions[i] += self.velocities[i]
        if self.inertia_weight > self.inertia_min:
            self.inertia_weight = max(
                self.inertia_weight * self.inertia_decay, self.inertia_min
            )


def plot_himmelblau():
    x = np.linspace(-6, 6, 600)
    y = np.linspace(-6, 6, 600)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau(X, Y)

    plt.figure(figsize=(7, 6))
    levels = np.logspace(0, 5, 35, base=10)
    cs = plt.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=0.8)
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    plt.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.6)
    plt.colorbar(label="f(x1, x2)")
    plt.title("Himmelblau Function Contours")
    plt.xlabel("x1")
    plt.ylabel("x2")

    known_minima = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ]
    for a, b in known_minima:
        plt.plot(a, b, "r*", markersize=10)
        plt.text(
            a + 0.15,
            b + 0.15,
            f"({a:.3f}, {b:.3f})",
            fontsize=12,
            color="red",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0.3),
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig("contour_with_known_minima.png", dpi=600, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    swarm_size = 50
    dimensions = 2
    position_min = -5
    position_max = 5
    velocity_max = 4
    cognitive_coefficient = 2
    social_coefficient = 2
    inertia_weight = 1.4
    inertia_min = 0.35
    inertia_decay = 0.995
    iterations = 200

    results = []
    for _ in range(50):
        pso = ParticleSwarmOptimizer(
            swarm_size,
            dimensions,
            position_min,
            position_max,
            velocity_max,
            inertia_weight,
            inertia_min,
            inertia_decay,
        )
        for t in range(iterations):
            pso.update_best_positions()
            pso.update_velocity_and_position(cognitive_coefficient, social_coefficient)
        x1, x2 = pso.global_best_position
        best_value = pso.global_best_value
        results.append((x1, x2, best_value))
        print(f"({x1:.6f}, {x2:.6f}, {best_value:.6f})")

    plot_himmelblau()
