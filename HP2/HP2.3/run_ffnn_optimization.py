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
        tournament_size: int = 3,  # NEW: tournament size for selection
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
        self.tournament_size = tournament_size
        self.tournament_probability = 0.75

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
        """Initialize random population with values in [0.0, 1.0]"""
        population = []
        for _ in range(self.pop_size):
            # Initialize genes in [0,1] as expected by decode_chromosome
            chromosome = np.random.uniform(0.0, 1.0, self.chromosome_length).tolist()
            population.append(chromosome)
        return population

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
        Probabilistic tournament selection:
        - sample k contestants from evaluated_pop
        - sort them by fitness descending
        - choose the i-th best with probability p*(1-p)**i (try best first with prob p)
        - if none selected (unlikely), return the worst of the contestants
        """
        if not evaluated_pop:
            return []

        n = len(evaluated_pop)
        k = max(1, min(self.tournament_size, n))

        # Sample k distinct competitors (tuples of (individual, fitness))
        competitors = random.sample(evaluated_pop, k)

        # Sort competitors by fitness descending (best first)
        competitors.sort(key=lambda t: t[1], reverse=True)

        p = getattr(self, "tournament_probability", 0.75)
        # Probabilistic pick: try best, then second-best, ...
        for ind, fit in competitors:
            if random.random() < p:
                return ind.copy()
            # if not picked, reduce effective pick chance for next by multiplying (1-p)
            # (implementation: loop continues so next has same p but conditioned on previous fails)
        # Fallback: return worst competitor if nobody sampled
        return competitors[-1][0].copy()

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
        """Perform mutation on an individual"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutated[i] += random.gauss(0, 0.1)  # Smaller std dev
                # Clamp values to [0,1] range for proper decoding
                mutated[i] = max(0.0, min(1.0, mutated[i]))
        return mutated

    def evolve(
        self, fitness_fn: FitnessFunction, generations: int = 100, verbose: bool = True
    ) -> Tuple[List[float], float]:
        """
        Run the evolutionary process for a specified number of generations
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
    truck_params, slopes, v_min=1.0, v_max=25.0, max_distance=10000.0, Tmax=750.0
):
    def fitness_function(chromosome):
        truck = Truck(**truck_params)
        total_fitness = 0.0

        for slope_idx, data_set_idx in slopes:
            controller = create_controller_from_chromosome(chromosome)
            truck.reset(position=0.0, velocity=20.0, gear=7, tb_total=500.0)

            result = truck.simulate(
                controller=controller,
                slope_index=slope_idx,
                data_set_index=data_set_idx,
                max_distance=max_distance,
                v_min=v_min,
                v_max=v_max,
            )

            # Check if the truck completed the entire distance
            distance_traveled = result["position"][-1]
            if distance_traveled < max_distance:
                return 0.0  # Invalidate solution if the truck didn't complete the path

            # Check for constraint violations
            if "velocity" in result and "brake_temp" in result:
                velocities = result["velocity"]
                temps = result["brake_temp"]

                if any(t > Tmax for t in temps):  # Brake temperature exceeds Tmax
                    return 0.0  # Invalidate solution
                if any(
                    v < v_min or v > v_max for v in velocities
                ):  # Velocity out of bounds
                    return 0.0  # Invalidate solution

            # Calculate average velocity
            avg_velocity = sum(result["velocity"]) / len(result["velocity"])

            # Reward for average velocity * distance traveled
            velocity_distance_reward = avg_velocity * distance_traveled
            total_fitness += velocity_distance_reward

        return total_fitness / len(slopes)

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
        slopes=[(i, 1) for i in range(1, 11)],  # Fixed: Slopes 1-10 in dataset 1
        v_min=vmin,
        v_max=vmax,
        max_distance=1000.0,
    )

    fitness_val = create_fitness_function(
        truck_params=truck_params,
        slopes=[(i, 2) for i in range(1, 6)],  # Fixed: Slopes 1-5 in dataset 2
        v_min=vmin,
        v_max=vmax,
        max_distance=1000.0,
    )

    # GA parameters
    ga_pop_size = 100
    ga = GeneticAlgorithm(
        pop_size=ga_pop_size,
        ni=ni,
        nh=nh,
        no=no,
        mutation_rate=(1.0 / ga_pop_size),  # mutation rate = 1 / population_size
        crossover_rate=0.8,
        selection_pressure=1.2,
        elitism=1,
        tournament_size=3,
        seed=42,
    )

    # Run optimization with early stopping based on validation fitness
    max_generations = 200
    patience = 30  # Stop after 30 generations with no improvement
    best_val_fitness = float("-inf")
    best_chromosome = None
    no_improvement = 0

    checkpoint_interval = 10  # Save best chromosome every 10 generations
    report_interval = 5  # Print detailed stats every 5 generations
    start_time = time.time()

    print("Starting optimization...")
    print("Generation | Training Fitness | Validation Fitness | Time (s) | Progress")

    for gen in range(max_generations):
        gen_start_time = time.time()

        # Evolve for one generation using training fitness
        ga.evolve(fitness_fn=fitness_tr, generations=1, verbose=False)

        # Evaluate best chromosome on validation set
        val_fitness = fitness_val(ga.best_individual)

        # Check for improvement
        if val_fitness > best_val_fitness:
            best_val_fitness = val_fitness
            best_chromosome = ga.best_individual.copy()
            no_improvement = 0  # Reset counter when improvement found

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

        # Calculate elapsed time and estimate remaining
        elapsed = time.time() - start_time
        gen_time = time.time() - gen_start_time
        estimated_remaining = gen_time * (max_generations - gen - 1)

        # Print progress with time information
        print(
            f"{gen:10d} | {ga.best_fitness:16.6f} | {val_fitness:18.6f} | {elapsed:8.1f} | {gen/max_generations*100:5.1f}%"
        )

        # Detailed report every report_interval generations
        if gen % report_interval == 0:
            # Calculate improvement rate (over last 10 generations or fewer if early)
            history_window = min(10, len(ga.best_fitness_history))
            if history_window > 1:
                recent_history = ga.best_fitness_history[-history_window:]
                improvement_rate = (
                    recent_history[-1] - recent_history[0]
                ) / history_window
                print(f"  Recent improvement rate: {improvement_rate:.6f}/gen")
                print(f"  Est. completion in: {estimated_remaining/60:.1f} minutes")
                print(
                    f"  Population diversity: {np.mean(np.std(np.array(ga.population), axis=0)):.6f}"
                )

        # Save checkpoint every checkpoint_interval generations
        if gen % checkpoint_interval == 0:
            # Save best chromosome as Python file with timestamp
            with open(f"checkpoints/checkpoint_gen{gen}.py", "w") as f:
                f.write("# Checkpoint chromosome from GA optimization\n")
                f.write(f"CHROMOSOME = {ga.best_individual}\n")
                f.write(f"NI = {ni}  # Number of inputs\n")
                f.write(f"NH = {nh}  # Number of hidden neurons\n")
                f.write(f"NO = {no}  # Number of outputs\n")
                f.write(f"SIGMOID_C = {SIGMOID_C}  # Sigmoid parameter c\n")
                f.write(f"TRAINING_FITNESS = {ga.best_fitness}  # Training fitness\n")
                f.write(f"VALIDATION_FITNESS = {val_fitness}  # Validation fitness\n")
                f.write(f"GENERATION = {gen}  # Generation number\n")

            print(f"  Checkpoint saved to checkpoint_gen{gen}.py")

    # End of generational loop -----------------------------------------------------------------
    # Save final best chromosome so run_test.py can load and rerun it
    if ga.best_individual is not None:
        # Save as Python module for easy import by run_test.py
        with open("best_chromosome.py", "w") as f:
            f.write("# Best chromosome saved by run_ffnn_optimization\n")
            f.write(f"CHROMOSOME = {ga.best_individual}\n")
            f.write(f"NI = {ni}\n")
            f.write(f"NH = {nh}\n")
            f.write(f"NO = {no}\n")
            f.write(f"SIGMOID_C = {SIGMOID_C}\n")
            f.write(f"TRAINING_FITNESS = {ga.best_fitness}\n")


run_optimization()
