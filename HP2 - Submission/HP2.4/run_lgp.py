import random
import function_data
import math

data_samples = function_data.load_function_data()
input_values = [x for x, y in data_samples]
target_values = [y for x, y in data_samples]

POPULATION_SIZE = 250
MIN_CHROMOSOME_LENGTH = 10
MAX_CHROMOSOME_LENGTH = 30
INSTRUCTION_SIZE = 4
MAX_INSTRUCTIONS = 100
VARIABLE_REGISTER_COUNT = 4
CONSTANT_REGISTER_COUNT = 1
CROSSOVER_PROBABILITY = 0.8

HIGH_ERROR_THRESHOLD = 0.20
LOW_ERROR_THRESHOLD = 0.01
MUTATION_RATE_AT_HIGH_ERROR = 6.0
MUTATION_RATE_AT_LOW_ERROR = 0.5
STAGNATION_GENERATION_LIMIT = 75
MUTATION_HEAT_UP_FACTOR = 1.5
MUTATION_COOL_DOWN_FACTOR = 0.9
IMPROVEMENT_THRESHOLD_RATIO = 0.001

OPERATORS = ["+", "-", "*", "/"]
OPERATOR_TO_INDEX = {op: i for i, op in enumerate(OPERATORS)}
INDEX_TO_OPERATOR = {i: op for i, op in enumerate(OPERATORS)}
CONSTANT_VALUES = [1.0, -1.0, 2.0, 0.0]

MAX_ERROR_VALUE = 1e10

TARGET_RMSE = 0.01


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = None
        self.error = None

    def evaluate(self):
        self.error = self.calculate_rmse()
        self.fitness = 1.0 / (self.error + 1e-8)

    def calculate_rmse(self):
        sample_count = len(input_values)
        squared_error_sum = 0.0
        for i in range(sample_count):
            x_value = input_values[i]
            y_true = target_values[i]
            y_pred = execute_chromosome(self.chromosome, x_value)
            if not math.isfinite(y_pred):
                return MAX_ERROR_VALUE
            error = y_pred - y_true
            squared_error = error * error
            if not math.isfinite(squared_error) or abs(squared_error) > MAX_ERROR_VALUE:
                return MAX_ERROR_VALUE
            squared_error_sum += squared_error
        rmse = math.sqrt(squared_error_sum / sample_count)
        if not math.isfinite(rmse) or rmse > MAX_ERROR_VALUE:
            return MAX_ERROR_VALUE
        return rmse


def generate_random_gene(gene_position):
    if gene_position % INSTRUCTION_SIZE == 0:
        return random.randint(0, 3)
    elif gene_position % INSTRUCTION_SIZE == 1:
        return random.randint(0, VARIABLE_REGISTER_COUNT - 1)
    else:
        return random.randint(0, VARIABLE_REGISTER_COUNT + CONSTANT_REGISTER_COUNT - 1)


def generate_random_chromosome():
    instruction_count = random.randint(MIN_CHROMOSOME_LENGTH, MAX_CHROMOSOME_LENGTH)
    chromosome = []
    for gene_index in range(instruction_count * INSTRUCTION_SIZE):
        chromosome.append(generate_random_gene(gene_index))
    return chromosome


def decode_instruction(chromosome, instruction_index):
    base_index = instruction_index * INSTRUCTION_SIZE
    operator = INDEX_TO_OPERATOR[chromosome[base_index]]
    destination_register = chromosome[base_index + 1]
    operand1_register = chromosome[base_index + 2]
    operand2_register = chromosome[base_index + 3]
    return (operator, destination_register, operand1_register, operand2_register)


def execute_chromosome(chromosome, input_value):
    variable_registers = [0.0] * VARIABLE_REGISTER_COUNT
    variable_registers[0] = input_value
    constant_registers = CONSTANT_VALUES
    instruction_count = len(chromosome) // INSTRUCTION_SIZE
    for instruction_index in range(instruction_count):
        operator, destination_register, operand1_register, operand2_register = (
            decode_instruction(chromosome, instruction_index)
        )

        def get_register_value(register_index):
            if register_index < VARIABLE_REGISTER_COUNT:
                return variable_registers[register_index]
            else:
                return constant_registers[register_index - VARIABLE_REGISTER_COUNT]

        operand1_value = get_register_value(operand1_register)
        operand2_value = get_register_value(operand2_register)
        try:
            if operator == "+":
                variable_registers[destination_register] = (
                    operand1_value + operand2_value
                )
            elif operator == "-":
                variable_registers[destination_register] = (
                    operand1_value - operand2_value
                )
            elif operator == "*":
                variable_registers[destination_register] = (
                    operand1_value * operand2_value
                )
            elif operator == "/":
                variable_registers[destination_register] = (
                    operand1_value / operand2_value
                    if abs(operand2_value) > 1e-6
                    else 1.0
                )
        except Exception:
            variable_registers[destination_register] = 0.0
    return variable_registers[0]


def two_point_crossover(parent1, parent2):
    chromosome1 = parent1.chromosome
    chromosome2 = parent2.chromosome
    instruction_count1 = len(chromosome1) // INSTRUCTION_SIZE
    instruction_count2 = len(chromosome2) // INSTRUCTION_SIZE
    if (
        instruction_count1 < 2
        or instruction_count2 < 2
        or random.random() > CROSSOVER_PROBABILITY
    ):
        return Individual(chromosome1[:]), Individual(chromosome2[:])
    crossover_points1 = sorted(random.sample(range(1, instruction_count1), 2))
    crossover_points2 = sorted(random.sample(range(1, instruction_count2), 2))
    child1_chromosome = (
        chromosome1[: crossover_points1[0] * INSTRUCTION_SIZE]
        + chromosome2[
            crossover_points2[0]
            * INSTRUCTION_SIZE : crossover_points2[1]
            * INSTRUCTION_SIZE
        ]
        + chromosome1[crossover_points1[1] * INSTRUCTION_SIZE :]
    )
    child2_chromosome = (
        chromosome2[: crossover_points2[0] * INSTRUCTION_SIZE]
        + chromosome1[
            crossover_points1[0]
            * INSTRUCTION_SIZE : crossover_points1[1]
            * INSTRUCTION_SIZE
        ]
        + chromosome2[crossover_points2[1] * INSTRUCTION_SIZE :]
    )
    return Individual(child1_chromosome), Individual(child2_chromosome)


def tournament_selection(population, tournament_size=5):
    candidates = random.sample(population, tournament_size)
    candidates.sort(key=lambda individual: individual.fitness, reverse=True)
    for candidate in candidates:
        if random.random() < 0.8:
            return candidate
    return candidates[-1]


def mutate_chromosome_with_rate(individual, mutation_probability):
    chromosome = individual.chromosome
    for gene_index in range(len(chromosome)):
        if random.random() < mutation_probability:
            gene_position = gene_index % INSTRUCTION_SIZE
            if gene_position == 0:
                chromosome[gene_index] = random.randint(0, 3)
            elif gene_position == 1:
                chromosome[gene_index] = random.randint(0, VARIABLE_REGISTER_COUNT - 1)
            else:
                chromosome[gene_index] = random.randint(
                    0, VARIABLE_REGISTER_COUNT + CONSTANT_REGISTER_COUNT - 1
                )
    return individual


def save_best_chromosome(individual):
    with open("best_chromosome_4.py", "w") as file:
        file.write("BEST_CHROMOSOME = " + repr(individual.chromosome) + "\n")
        file.write("CONSTANTS = " + repr(CONSTANT_VALUES) + "\n")
        file.write("VARIABLE_REGISTER_COUNT = " + repr(VARIABLE_REGISTER_COUNT) + "\n")
        file.write("CONSTANT_REGISTER_COUNT = " + repr(CONSTANT_REGISTER_COUNT) + "\n")


def main():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = generate_random_chromosome()
        individual = Individual(chromosome)
        individual.evaluate()
        population.append(individual)

    best_individual = max(population, key=lambda individual: individual.fitness)
    best_fitness = best_individual.fitness
    best_error = best_individual.error

    adaptive_mutation_multiplier = 1.0
    generations_without_improvement = 0
    best_overall_individual = best_individual

    generation = 0
    while best_error > TARGET_RMSE:

        if best_error > HIGH_ERROR_THRESHOLD:
            global_mutation_rate = MUTATION_RATE_AT_HIGH_ERROR
        elif best_error < LOW_ERROR_THRESHOLD:
            global_mutation_rate = MUTATION_RATE_AT_LOW_ERROR
        else:
            progress = (HIGH_ERROR_THRESHOLD - best_error) / (
                HIGH_ERROR_THRESHOLD - LOW_ERROR_THRESHOLD
            )
            global_mutation_rate = MUTATION_RATE_AT_HIGH_ERROR - progress * (
                MUTATION_RATE_AT_HIGH_ERROR - MUTATION_RATE_AT_LOW_ERROR
            )

        improvement_threshold = (
            best_overall_individual.fitness * IMPROVEMENT_THRESHOLD_RATIO
        )
        if best_fitness > best_overall_individual.fitness + improvement_threshold:
            adaptive_mutation_multiplier = max(
                0.5, adaptive_mutation_multiplier * MUTATION_COOL_DOWN_FACTOR
            )
            best_overall_individual = best_individual
            generations_without_improvement = 0
            current_mutation_rate = global_mutation_rate * adaptive_mutation_multiplier
            print(
                f"[Generation {generation}] Improvement detected! "
                f"New best fitness: {best_fitness:.6f}, "
                f"RMSE: {best_error:.6f}, "
                f"Chromosome length: {len(best_individual.chromosome)//INSTRUCTION_SIZE} instructions. "
                f"Mutation rate adjusted to {current_mutation_rate:.3f} "
                f"(base: {global_mutation_rate:.2f}, multiplier: {adaptive_mutation_multiplier:.2f})"
            )
        else:
            generations_without_improvement += 1
            if generations_without_improvement > STAGNATION_GENERATION_LIMIT:
                adaptive_mutation_multiplier = min(
                    2.0, adaptive_mutation_multiplier * MUTATION_HEAT_UP_FACTOR
                )
                generations_without_improvement = 0
                current_mutation_rate = (
                    global_mutation_rate * adaptive_mutation_multiplier
                )
                print(
                    f"[Generation {generation}] No significant improvement for {STAGNATION_GENERATION_LIMIT} generations. "
                    f"Increasing mutation rate to {current_mutation_rate:.3f} "
                    f"(base: {global_mutation_rate:.2f}, multiplier: {adaptive_mutation_multiplier:.2f})"
                )

        current_mutation_rate = global_mutation_rate * adaptive_mutation_multiplier

        if generation % 20 == 0:
            print(
                f"[Generation {generation}] Status update: "
                f"Current RMSE: {best_error:.6f}, "
                f"Mutation rate: {current_mutation_rate:.3f} "
                f"(base: {global_mutation_rate:.2f}, multiplier: {adaptive_mutation_multiplier:.2f})"
            )

        population.sort(key=lambda individual: individual.fitness, reverse=True)
        new_population = [population[0]]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            if random.random() < CROSSOVER_PROBABILITY:
                child1, child2 = two_point_crossover(parent1, parent2)
            else:
                child1 = Individual(parent1.chromosome[:])
                child2 = Individual(parent2.chromosome[:])

            if (len(child1.chromosome) // INSTRUCTION_SIZE) > MAX_INSTRUCTIONS or (
                len(child2.chromosome) // INSTRUCTION_SIZE
            ) > MAX_INSTRUCTIONS:
                continue

            mutation_probability_child1 = current_mutation_rate / (
                len(child1.chromosome) + 1
            )
            mutation_probability_child2 = current_mutation_rate / (
                len(child2.chromosome) + 1
            )
            child1 = mutate_chromosome_with_rate(child1, mutation_probability_child1)
            child2 = mutate_chromosome_with_rate(child2, mutation_probability_child2)

            child1.evaluate()
            child2.evaluate()

            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population
        best_individual = max(population, key=lambda individual: individual.fitness)
        best_fitness = best_individual.fitness
        best_error = best_individual.error
        generation += 1

    print("\n=== Evolution Complete ===")
    print(f"Final best fitness score: {best_fitness:.6f}")
    print(f"Final best RMSE: {best_error:.6f}")
    print(f"Total generations executed: {generation}")
    save_best_chromosome(best_individual)


if __name__ == "__main__":
    main()
