import numpy as np
import random
import time
import os
from typing import List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
import math
from truck_model import Truck
from nn_controller import create_controller_from_chromosome, SIGMOID_C

Population = List[List[float]]
FitnessFunction = Callable[[List[float]], float]


class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int = 50,
        chromosome_length: Optional[int] = None,
        ni: int = 4,
        nh: int = 5,
        no: int = 2,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        elitism: int = 1,
        seed: Optional[int] = None,
        tournament_size: int = 3,
    ):
        self.pop_size = pop_size
        self.ni = ni
        self.nh = nh
        self.no = no
        if chromosome_length is None:
            w_i_h_size = nh * (ni + 1)
            w_h_o_size = no * (nh + 1)
            self.chromosome_length = w_i_h_size + w_h_o_size
        else:
            self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.tournament_probability = 0.75
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.population = self._init_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = -float("inf")

    def _init_population(self) -> Population:
        population = []
        for _ in range(self.pop_size):
            chromosome = np.random.uniform(0.0, 1.0, self.chromosome_length).tolist()
            population.append(chromosome)
        return population

    def _evaluate_population(
        self, fitness_fn: FitnessFunction
    ) -> List[Tuple[List[float], float]]:
        evaluated = []
        for ind in self.population:
            fitness = fitness_fn(ind)
            evaluated.append((ind, fitness))
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = ind.copy()
        evaluated.sort(key=lambda x: x[1], reverse=True)
        return evaluated

    def _selection(self, evaluated_pop: List[Tuple[List[float], float]]) -> List[float]:
        if not evaluated_pop:
            return []
        n = len(evaluated_pop)
        k = max(1, min(self.tournament_size, n))
        competitors = random.sample(evaluated_pop, k)
        competitors.sort(key=lambda t: t[1], reverse=True)
        p = getattr(self, "tournament_probability", 0.75)
        for ind, fit in competitors:
            if random.random() < p:
                return ind.copy()
        return competitors[-1][0].copy()

    def _crossover(
        self, parent1: List[float], parent2: List[float]
    ) -> Tuple[List[float], List[float]]:
        if random.random() < self.crossover_rate:
            child1 = parent1.copy()
            child2 = parent2.copy()
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def _mutate(self, individual: List[float]) -> List[float]:
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += random.gauss(0, 0.1)
                mutated[i] = max(0.0, min(1.0, mutated[i]))
        return mutated

    def evolve(
        self, fitness_fn: FitnessFunction, generations: int = 100, verbose: bool = True
    ) -> Tuple[Optional[List[float]], float]:
        start_time = time.time()
        for gen in range(generations):
            evaluated_pop = self._evaluate_population(fitness_fn)
            fitnesses = [fit for _, fit in evaluated_pop]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_gen_fitness = fitnesses[0]
            self.best_fitness_history.append(best_gen_fitness)
            self.avg_fitness_history.append(avg_fitness)
            if verbose and (gen % 10 == 0 or gen == generations - 1):
                elapsed = time.time() - start_time
                print(
                    f"Gen {gen:3d}/{generations}: Best={best_gen_fitness:.4f}, Avg={avg_fitness:.4f}, Time={elapsed:.1f}s"
                )
            new_population = []
            for i in range(self.elitism):
                if i < len(evaluated_pop):
                    new_population.append(evaluated_pop[i][0].copy())
            while len(new_population) < self.pop_size:
                parent1 = self._selection(evaluated_pop)
                parent2 = self._selection(evaluated_pop)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.append(child1)
                if len(new_population) < self.pop_size:
                    new_population.append(child2)
            self.population = new_population
        return self.best_individual, self.best_fitness

    def plot_fitness_history(self, save_path: Optional[str] = None):
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


def create_fitness_function(
    truck_params, slopes, v_min=1.0, v_max=25.0, max_distance=1000.0, Tmax=750.0
):
    def fitness_function(chromosome):
        truck = Truck(**truck_params)
        total_fitness = 0.0
        temp_penalty_k = 2.0
        speed_bonus_weight = 0.2
        pedal_reward_weight = 8
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
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            velocities = result.get("velocity", [])
            pedals = result.get("pedal", [])
            temps = result.get("brake_temp", [])
            distance_traveled = result["position"][-1]
            avg_speed = sum(velocities) / len(velocities) if velocities else 0.0
            base_fi = avg_speed * distance_traveled
            max_tb = max(temps) if temps else truck_params.get("max_brake_temp", Tmax)
            if metrics.get("constraint_violated", False) or metrics.get(
                "termination_reason", None
            ) in ("v_max_exceeded", "v_min_violated", "brake_temp_exceeded"):
                excess = max(0.0, max_tb - Tmax) / Tmax
                v_excess = max(0.0, max(velocities) - v_max) / v_max
                penalty = math.exp(-temp_penalty_k * excess) * math.exp(
                    -30.0 * v_excess
                )
            else:
                penalty = 1.0
            pedal_reward = 0.0
            for v, pedal in zip(velocities, pedals):
                if v > 0.9 * v_max:
                    pedal_reward += pedal * pedal_reward_weight
            speed_bonus = 1.0 + speed_bonus_weight * (avg_speed / v_max)
            timesteps_above_vmax = sum(1 for v in velocities if v > v_max)
            time_penalty = math.exp(-0.05 * timesteps_above_vmax)
            fitness_i = base_fi * penalty * speed_bonus * time_penalty + pedal_reward
            total_fitness += fitness_i
        return total_fitness / len(slopes)

    return fitness_function


def run_optimization():
    Tmax = 750.0
    M = 20000.0
    tau = 30.0
    Ch = 40.0
    Tamb = 283.0
    Cb = 3000.0
    vmax = 25.0
    vmin = 1.0
    ni = 3
    nh = 5
    no = 2
    truck_params = {
        "mass": M,
        "base_engine_brake_coeff": Cb,
        "max_brake_temp": Tmax,
        "temp_cooling_tau": tau,
        "temp_heating_ch": Ch,
        "ambient_temp": Tamb,
    }
    fitness_tr = create_fitness_function(
        truck_params=truck_params,
        slopes=[(i, 1) for i in range(1, 11)],
        v_min=vmin,
        v_max=vmax,
        max_distance=1000.0,
    )
    fitness_val = create_fitness_function(
        truck_params=truck_params,
        slopes=[(i, 2) for i in range(1, 6)],
        v_min=vmin,
        v_max=vmax,
        max_distance=1000.0,
    )
    chromosome_length = nh * (ni + 1) + no * (nh + 1)
    ga_pop_size = 300
    ga = GeneticAlgorithm(
        pop_size=ga_pop_size,
        ni=ni,
        nh=nh,
        no=no,
        mutation_rate=(1.0 / chromosome_length),
        crossover_rate=0.8,
        elitism=1,
        tournament_size=3,
    )
    max_generations = 200
    best_val_fitness = float("-inf")
    best_chromosome = None
    validation_history = []
    checkpoint_interval = 10
    report_interval = 5
    start_time = time.time()
    print("Starting optimization...")
    print("Generation | Training Fitness | Validation Fitness | Time (s) | Progress")
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    for gen in range(max_generations):
        gen_start_time = time.time()
        ga.evolve(fitness_fn=fitness_tr, generations=1, verbose=False)
        val_fitness = fitness_val(ga.best_individual)
        validation_history.append(val_fitness)
        if val_fitness > best_val_fitness:
            best_val_fitness = val_fitness
            best_chromosome = ga.best_individual.copy()
            with open("best_chromosome.py", "w") as f:
                f.write("# Best chromosome found through GA optimization\n")
                f.write(f"CHROMOSOME = {best_chromosome}\n")
                f.write(f"NI = {ni}  # Number of inputs\n")
                f.write(f"NH = {nh}  # Number of hidden neurons\n")
                f.write(f"NO = {no}  # Number of outputs\n")
                f.write(f"SIGMOID_C = {SIGMOID_C}  # Sigmoid parameter c\n")
        elapsed = time.time() - start_time
        gen_time = time.time() - gen_start_time
        estimated_remaining = gen_time * (max_generations - gen - 1)
        print(
            f"{gen:10d} | {ga.best_fitness:16.6f} | {val_fitness:18.6f} | {elapsed:8.1f} | {gen/max_generations*100:5.1f}%"
        )
        if gen % report_interval == 0:
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
        if gen % checkpoint_interval == 0:
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
    if ga.best_individual is not None:
        with open("best_chromosome.py", "w") as f:
            f.write("# Best chromosome saved by run_ffnn_optimization\n")
            f.write(f"CHROMOSOME = {ga.best_individual}\n")
            f.write(f"NI = {ni}\n")
            f.write(f"NH = {nh}\n")
            f.write(f"NO = {no}\n")
            f.write(f"SIGMOID_C = {SIGMOID_C}\n")
            f.write(f"TRAINING_FITNESS = {ga.best_fitness}\n")
    try:
        plt.figure(figsize=(10, 6))
        gens = list(range(len(ga.best_fitness_history)))
        plt.plot(
            gens, ga.best_fitness_history, "-b", label="Training: max fitness (per gen)"
        )
        plt.plot(
            gens, validation_history, "-r", label="Validation: best individual fitness"
        )
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("GA fitness: training max and validation fitness vs generation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("fitness_history.png")
        print("Saved fitness history plot to fitness_history.png")
        plt.show()
    except Exception as e:
        print("Could not plot fitness history:", e)


run_optimization()
