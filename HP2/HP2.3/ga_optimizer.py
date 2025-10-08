import numpy as np
import random
import time
import json
import os
from typing import List, Dict, Tuple, Callable, Any, Optional
import matplotlib.pyplot as plt

from truck_model import Truck
from nn_controller import create_controller_from_chromosome, SIGMOID_C

# Type aliases
Population = List[List[float]]
FitnessFunction = Callable[[List[float]], float]


class GeneticAlgorithm:
    """Genetic Algorithm for optimizing neural network weights for truck control"""

    def __init__(
        self,
        pop_size: int = 50,
        chromosome_length: int = None,
        ni: int = 4,
        nh: int = 6,
        no: int = 2,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        selection_pressure: float = 1.5,
        elitism: int = 1,
        seed: int = None,
    ):
        """
        Initialize the GA optimizer.

        Args:
            pop_size: Population size
            chromosome_length: Length of chromosome (if None, calculated from ni, nh, no)
            ni: Number of NN inputs
            nh: Number of hidden neurons
            no: Number of outputs
            mutation_rate: Probability of gene mutation
            crossover_rate: Probability of crossover
            selection_pressure: Pressure for rank selection
            elitism: Number of best individuals to preserve
            seed: Random seed
        """
        self.pop_size = pop_size
        self.ni = ni
        self.nh = nh
        self.no = no

        # Calculate chromosome length if not provided
        if chromosome_length is None:
            w_i_h_size = nh * (ni + 1)
            w_h_o_size = no * (nh + 1)
            self.chromosome_length = w_i_h_size + w_h_o_size
        else:
            self.chromosome_length = chromosome_length

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        self.elitism = elitism

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize population with random chromosomes
        self.population = self._init_population()

        # Statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = -float("inf")

    def _init_population(self) -> Population:
        """Initialize random population"""
        return [
            np.random.uniform(-1.0, 1.0, self.chromosome_length).tolist()
            for _ in range(self.pop_size)
        ]

    def _evaluate_population(
        self, fitness_fn: FitnessFunction
    ) -> List[Tuple[List[float], float]]:
        """
        Evaluate all individuals in the population

        Returns:
            List of (individual, fitness) tuples, sorted by descending fitness
        """
        evaluated = []
        for ind in self.population:
            fitness = fitness_fn(ind)
            evaluated.append((ind, fitness))

            # Update best individual
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = ind.copy()

        # Sort by fitness (descending)
        evaluated.sort(key=lambda x: x[1], reverse=True)
        return evaluated

    def _selection(self, evaluated_pop: List[Tuple[List[float], float]]) -> List[float]:
        """
        Rank-based selection
        """
        # Extract individuals
        individuals = [ind for ind, _ in evaluated_pop]

        # Calculate selection probabilities based on rank
        n = len(individuals)
        ranks = np.arange(n, 0, -1)  # n, n-1, ..., 1

        # Linear ranking probabilities
        selection_probs = (2 - self.selection_pressure) / n + 2 * ranks * (
            self.selection_pressure - 1
        ) / (n * (n - 1))

        # Normalize probabilities
        selection_probs = selection_probs / np.sum(selection_probs)

        # Select an individual based on probabilities
        idx = np.random.choice(n, p=selection_probs)
        return individuals[idx].copy()

    def _crossover(
        self, parent1: List[float], parent2: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Perform uniform crossover between two parents
        """
        if random.random() < self.crossover_rate:
            child1 = parent1.copy()
            child2 = parent2.copy()

            # Uniform crossover
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]

            return child1, child2
        else:
            # No crossover
            return parent1.copy(), parent2.copy()

    def _mutate(self, individual: List[float]) -> List[float]:
        """
        Perform mutation on an individual
        """
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutated[i] += random.gauss(0, 0.2)
                # Clamp values to reasonable range
                mutated[i] = max(-3.0, min(3.0, mutated[i]))
        return mutated

    def evolve(
        self, fitness_fn: FitnessFunction, generations: int = 100, verbose: bool = True
    ) -> Tuple[List[float], float]:
        """
        Run the evolutionary process for a specified number of generations

        Args:
            fitness_fn: Function that evaluates fitness of an individual
            generations: Number of generations to evolve
            verbose: Whether to print progress

        Returns:
            Tuple of (best_individual, best_fitness)
        """
        start_time = time.time()

        for gen in range(generations):
            # Evaluate current population
            evaluated_pop = self._evaluate_population(fitness_fn)

            # Extract fitness values for statistics
            fitnesses = [fit for _, fit in evaluated_pop]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_gen_fitness = fitnesses[0]

            # Store statistics
            self.best_fitness_history.append(best_gen_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Print progress
            if verbose and (gen % 10 == 0 or gen == generations - 1):
                elapsed = time.time() - start_time
                print(
                    f"Gen {gen:3d}/{generations}: Best={best_gen_fitness:.4f}, Avg={avg_fitness:.4f}, Time={elapsed:.1f}s"
                )

            # Stop if we've reached the final generation
            if gen == generations - 1:
                break

            # Create new population
            new_population = []

            # Elitism: copy best individuals to new population
            for i in range(self.elitism):
                if i < len(evaluated_pop):
                    new_population.append(evaluated_pop[i][0].copy())

            # Fill the rest of the population with offspring
            while len(new_population) < self.pop_size:
                # Select parents
                parent1 = self._selection(evaluated_pop)
                parent2 = self._selection(evaluated_pop)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                # Add children to new population
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)

            # Replace old population
            self.population = new_population

        return self.best_individual, self.best_fitness

    def plot_fitness_history(self, save_path: Optional[str] = None):
        """Plot the fitness history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, "b-", label="Best Fitness")
        plt.plot(self.avg_fitness_history, "r-", label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness History")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_best(self, filename: str = "best_chromosome.json"):
        """Save the best individual to a file"""
        data = {
            "chromosome": self.best_individual,
            "fitness": self.best_fitness,
            "ni": self.ni,
            "nh": self.nh,
            "no": self.no,
            "history": {
                "best_fitness": self.best_fitness_history,
                "avg_fitness": self.avg_fitness_history,
            },
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Best individual saved to {filename}")


def create_fitness_function(
    truck_params: Dict = None,
    slope_indices: List[int] = [0, 1],
    data_set_indices: List[int] = [0],
    ni: int = 3,  # Changed to 3 inputs per spec
    nh: int = 6,
    no: int = 2,
    max_distance: float = 1000.0,  # Changed to 1000m per spec
    v_max: float = 25.0,
    v_min: float = 1.0,
) -> FitnessFunction:
    """
    Create a fitness function for evaluating truck controllers

    Args:
        truck_params: Parameters for truck model
        slope_indices: List of slope profiles to test on
        data_set_indices: List of data sets to test on
        ni, nh, no: Neural network dimensions
        max_distance: Maximum distance for simulations (1000m)
        v_max: Maximum allowed velocity
        v_min: Minimum allowed velocity

    Returns:
        Fitness function that evaluates a chromosome
    """
    # Default truck parameters
    if truck_params is None:
        truck_params = {
            "mass": 20000.0,  # kg
            "base_engine_brake_coeff": 3000.0,  # Cb = 3000 N per spec
            "max_brake_temp": 750.0,  # Tmax = 750 K per spec
            "temp_cooling_tau": 30.0,  # tau = 30s per spec
            "temp_heating_ch": 40.0,  # Ch = 40 K/s per spec
            "ambient_temp": 283.0,  # Tamb = 283 K per spec
        }

    def fitness_function(chromosome: List[float]) -> float:
        # Create controller from chromosome
        controller = create_controller_from_chromosome(chromosome, ni, nh, no)

        # Create truck instance
        truck = Truck(**truck_params)

        # Run simulations for different slopes and datasets
        total_fitness = 0.0
        num_scenarios = len(slope_indices) * len(data_set_indices)

        for slope_idx in slope_indices:
            for dataset_idx in data_set_indices:
                # Reset truck with starting conditions from assignment:
                # x = 0, v = 20 m/s, gear = 7, Tb = 500 K
                truck.reset(position=0.0, velocity=20.0, gear=7, tb_total=500.0)

                # Simulate descent
                try:
                    history = truck.simulate(
                        controller=controller,
                        slope_index=slope_idx,
                        data_set_index=dataset_idx,
                        max_distance=max_distance,
                        auto_gear=False,  # Use NN's gear change
                    )

                    # Extract metrics for fitness calculation
                    final_velocity = history["velocity"][-1]
                    max_velocity = max(history["velocity"])
                    min_velocity = min(history["velocity"])
                    max_brake_temp = max(history["brake_temp"])
                    avg_velocity = sum(history["velocity"]) / len(history["velocity"])
                    distance = history["position"][-1]

                    # Check if constraints were violated
                    constraints_violated = (
                        max_velocity > v_max
                        or min_velocity < v_min
                        or max_brake_temp > truck_params["max_brake_temp"]
                    )

                    # Simple fitness according to spec: F = v*d
                    # If constraints violated, use actual distance traveled
                    # If completed without violations, use full distance
                    scenario_fitness = avg_velocity * distance

                    # Add penalty if constraints violated (optional)
                    if constraints_violated:
                        # Scale down fitness significantly when constraints are violated
                        scenario_fitness *= 0.5

                    # Add to total fitness
                    total_fitness += scenario_fitness

                except Exception as e:
                    print(f"Simulation error: {e}")
                    total_fitness += 0.0  # Zero fitness for failed simulations

        # Average fitness across all scenarios
        return total_fitness / num_scenarios

    return fitness_function


def run_optimization():
    """Run the optimization process using the parameters from the assignment"""
    # Assignment parameters
    Tmax = 750.0  # K
    M = 20000.0  # kg
    tau = 30.0  # s
    Ch = 40.0  # K/s
    Tamb = 283.0  # K
    Cb = 3000.0  # N (engine brake base coeff)
    vmax = 25.0  # m/s
    vmin = 1.0  # m/s
    alpha_max = 10.0  # degrees

    # Neural network configuration
    ni = 3  # Normalized inputs: v/vmax, α/αmax, Tb/Tmax
    nh = 6  # Hidden neurons (you may adjust this between 3-10)
    no = 2  # Two outputs: brake_pedal, gear_change

    # Truck parameters
    truck_params = {
        "mass": M,
        "base_engine_brake_coeff": Cb,
        "max_brake_temp": Tmax,
        "temp_cooling_tau": tau,
        "temp_heating_ch": Ch,
        "ambient_temp": Tamb,
    }

    # Create fitness functions for training and validation
    fitness_tr = create_fitness_function(
        truck_params=truck_params,
        slope_indices=list(range(1, 11)),  # Slopes 1-10
        data_set_indices=[1],  # Training set
        ni=ni,
        nh=nh,
        no=no,
        max_distance=1000.0,
        v_max=vmax,
        v_min=vmin,
    )

    fitness_val = create_fitness_function(
        truck_params=truck_params,
        slope_indices=list(range(1, 6)),  # Slopes 1-5
        data_set_indices=[2],  # Validation set
        ni=ni,
        nh=nh,
        no=no,
        max_distance=1000.0,
        v_max=vmax,
        v_min=vmin,
    )

    # GA parameters
    ga = GeneticAlgorithm(
        pop_size=100,
        ni=ni,
        nh=nh,
        no=no,
        mutation_rate=0.05,
        crossover_rate=0.8,
        selection_pressure=1.5,
        elitism=2,
        seed=42,
    )

    # Run optimization with early stopping based on validation fitness
    max_generations = 200
    patience = 20
    best_val_fitness = float("-inf")
    best_chromosome = None
    no_improvement = 0

    print("Starting optimization...")
    print("Generation | Training Fitness | Validation Fitness")

    for gen in range(max_generations):
        # Evolve for one generation using training fitness
        ga.evolve(fitness_fn=fitness_tr, generations=1, verbose=False)

        # Evaluate best chromosome on validation set
        val_fitness = fitness_val(ga.best_individual)

        print(f"{gen:10d} | {ga.best_fitness:16.6f} | {val_fitness:18.6f}")

        # Check for improvement
        if val_fitness > best_val_fitness:
            best_val_fitness = val_fitness
            best_chromosome = ga.best_individual.copy()
            no_improvement = 0

            # Save best chromosome as Python file
            with open("best_chromosome.py", "w") as f:
                f.write("# Best chromosome found through GA optimization\n")
                f.write(f"CHROMOSOME = {best_chromosome}\n")
                f.write(f"NI = {ni}  # Number of inputs\n")
                f.write(f"NH = {nh}  # Number of hidden neurons\n")
                f.write(f"NO = {no}  # Number of outputs\n")
                f.write(f"SIGMOID_C = {SIGMOID_C}  # Sigmoid parameter c\n")
        else:
            no_improvement += 1

        # Early stopping
        if no_improvement >= patience:
            print(
                f"Early stopping after {gen} generations (no improvement for {patience} generations)"
            )
            break

    print("\nOptimization completed!")
    print(f"Best validation fitness: {best_val_fitness:.6f}")

    # Also save as JSON for test program
    import json

    with open("best_chromosome.json", "w") as f:
        json.dump(
            {
                "chromosome": best_chromosome,
                "ni": ni,
                "nh": nh,
                "no": no,
                "sigmoid_c": SIGMOID_C,
                "fitness": best_val_fitness,
            },
            f,
            indent=2,
        )

    print("Best chromosome saved to best_chromosome.py and best_chromosome.json")

    # Plot fitness history if available
    if hasattr(ga, "best_fitness_history") and len(ga.best_fitness_history) > 0:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(ga.best_fitness_history, label="Training Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness History")
        plt.legend()
        plt.grid(True)
        plt.savefig("fitness_history.png")
        plt.show()


if __name__ == "__main__":
    run_optimization()
