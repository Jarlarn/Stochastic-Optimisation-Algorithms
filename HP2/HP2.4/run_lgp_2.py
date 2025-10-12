import numpy as np
import random
import copy
import os
import time
from function_data import load_function_data

# LGP algorithm parameters
POPULATION_SIZE = 500
MAX_GENERATIONS = 10000  # Safety limit to prevent infinite loops
ERROR_THRESHOLD = 0.01  # Run until error is below this threshold
TOURNAMENT_SIZE = 7
INITIAL_MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.9
MAX_CHROMOSOME_LENGTH = 400  # Hard limit: 100 instructions (4 genes each)
SOFT_LIMIT_LENGTH = 300  # Soft limit with penalty
PENALTY_FACTOR = 0.8  # Penalty for exceeding soft limit
TIME_WINDOW_SIZE = 30  # Window size for mutation probability adaptation
PROBABILITY_CHANGE_FACTOR = 1.5  # Factor to change mutation probability

# Checkpoint settings
CHECKPOINT_INTERVAL = 1000  # Save checkpoint every 1000 generations
CHECKPOINT_MAX_AGE = 10000  # Delete checkpoints older than 10000 generations
CHECKPOINT_DIR = "checkpoints2"

# Register configuration
NUM_VARIABLE_REGISTERS = 6  # r[0] is input x, r[1] will be output
NUM_CONSTANT_REGISTERS = 6

# Operators
OPERATORS = ["+", "-", "*", "/"]


def initialize_constants():
    """Initialize constant registers with useful values for this problem"""
    constants = [1.0, 2.0, 3.0, 0.5, 0.1, -1.0]
    return constants


def create_instruction():
    """Create a random instruction [operator, destination, source1, source2]"""
    operator = random.choice(OPERATORS)
    dest_register = random.randint(
        1, NUM_VARIABLE_REGISTERS - 1
    )  # Skip register 0 (input)

    # Source registers can be variable or constant
    source1_type = random.choice([0, 1])  # 0: variable, 1: constant
    source2_type = random.choice([0, 1])

    if source1_type == 0:
        source1 = random.randint(0, NUM_VARIABLE_REGISTERS - 1)
    else:
        source1 = random.randint(0, NUM_CONSTANT_REGISTERS - 1) + NUM_VARIABLE_REGISTERS

    if source2_type == 0:
        source2 = random.randint(0, NUM_VARIABLE_REGISTERS - 1)
    else:
        source2 = random.randint(0, NUM_CONSTANT_REGISTERS - 1) + NUM_VARIABLE_REGISTERS

    return [operator, dest_register, source1, source2]


def initialize_chromosome():
    """Initialize a random chromosome with a set of instructions"""
    length = random.randint(4, 40)  # Start with reasonable length
    chromosome = []
    for _ in range(length):
        chromosome.append(create_instruction())
    return chromosome


def initialize_population(size):
    """Create an initial population of chromosomes"""
    return [initialize_chromosome() for _ in range(size)]


def execute_chromosome(chromosome, x, constants):
    """Execute the chromosome instructions and return the result"""
    registers = [0.0] * NUM_VARIABLE_REGISTERS
    registers[0] = x  # Input x goes into register 0

    for instruction in chromosome:
        operator, dest, src1, src2 = instruction

        # Get source values
        if src1 < NUM_VARIABLE_REGISTERS:
            val1 = registers[src1]
        else:
            val1 = constants[src1 - NUM_VARIABLE_REGISTERS]

        if src2 < NUM_VARIABLE_REGISTERS:
            val2 = registers[src2]
        else:
            val2 = constants[src2 - NUM_VARIABLE_REGISTERS]

        # Execute the operation
        if operator == "+":
            result = val1 + val2
        elif operator == "-":
            result = val1 - val2
        elif operator == "*":
            result = val1 * val2
        elif operator == "/":
            if abs(val2) < 1e-10:  # Avoid division by zero
                result = val1  # Keep the first value if division by zero
            else:
                result = val1 / val2

        registers[dest] = result

    return registers[1]  # Output is in register 1


def evaluate_fitness(chromosome, data, constants):
    """Calculate fitness based on RMSE between predictions and actual values"""
    error_sum = 0.0

    for x, y in data:
        try:
            y_pred = execute_chromosome(chromosome, x, constants)
            error_sum += (y_pred - y) ** 2
        except:
            # If any error occurs during execution, give a poor fitness
            return 0.001

    rmse = np.sqrt(error_sum / len(data))

    # Apply penalty if chromosome exceeds soft limit
    if len(chromosome) > SOFT_LIMIT_LENGTH // 4:
        return 1.0 / (rmse / PENALTY_FACTOR)

    return 1.0 / rmse


def tournament_selection(population, fitness_values):
    """Select a chromosome using tournament selection"""
    tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return copy.deepcopy(population[winner_index])


def two_point_crossover(parent1, parent2):
    """Perform two-point crossover between parents"""
    # Make sure crossover points are between instructions (not within)
    if len(parent1) < 2 or len(parent2) < 2:
        # If parents are too small, return copies
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    point1_parent1 = random.randint(1, len(parent1) - 1)
    point2_parent1 = random.randint(1, len(parent1) - 1)
    while point1_parent1 == point2_parent1:
        point2_parent1 = random.randint(1, len(parent1) - 1)

    if point1_parent1 > point2_parent1:
        point1_parent1, point2_parent1 = point2_parent1, point1_parent1

    point1_parent2 = random.randint(1, len(parent2) - 1)
    point2_parent2 = random.randint(1, len(parent2) - 1)
    while point1_parent2 == point2_parent2:
        point2_parent2 = random.randint(1, len(parent2) - 1)

    if point1_parent2 > point2_parent2:
        point1_parent2, point2_parent2 = point2_parent2, point1_parent2

    # Create children
    child1 = (
        parent1[:point1_parent1]
        + parent2[point1_parent2:point2_parent2]
        + parent1[point2_parent1:]
    )
    child2 = (
        parent2[:point1_parent2]
        + parent1[point1_parent1:point2_parent1]
        + parent2[point2_parent2:]
    )

    # Check if children exceed maximum length
    if (
        len(child1) <= MAX_CHROMOSOME_LENGTH // 4
        and len(child2) <= MAX_CHROMOSOME_LENGTH // 4
    ):
        return child1, child2
    else:
        # Try again if children are too long
        return two_point_crossover(parent1, parent2)


def mutate(chromosome, mutation_rate):
    """Apply mutations to the chromosome"""
    mutated = copy.deepcopy(chromosome)

    for i in range(len(mutated)):
        instruction = mutated[i]

        # Mutate operator
        if random.random() < mutation_rate:
            instruction[0] = random.choice(OPERATORS)

        # Mutate destination register
        if random.random() < mutation_rate:
            instruction[1] = random.randint(1, NUM_VARIABLE_REGISTERS - 1)

        # Mutate source1
        if random.random() < mutation_rate:
            source1_type = random.choice([0, 1])  # 0: variable, 1: constant
            if source1_type == 0:
                instruction[2] = random.randint(0, NUM_VARIABLE_REGISTERS - 1)
            else:
                instruction[2] = (
                    random.randint(0, NUM_CONSTANT_REGISTERS - 1)
                    + NUM_VARIABLE_REGISTERS
                )

        # Mutate source2
        if random.random() < mutation_rate:
            source2_type = random.choice([0, 1])
            if source2_type == 0:
                instruction[3] = random.randint(0, NUM_VARIABLE_REGISTERS - 1)
            else:
                instruction[3] = (
                    random.randint(0, NUM_CONSTANT_REGISTERS - 1)
                    + NUM_VARIABLE_REGISTERS
                )

    # Add instruction with small probability
    if random.random() < mutation_rate and len(mutated) < MAX_CHROMOSOME_LENGTH // 4:
        insert_pos = random.randint(0, len(mutated))
        mutated.insert(insert_pos, create_instruction())

    # Remove instruction with small probability
    if random.random() < mutation_rate and len(mutated) > 1:
        remove_pos = random.randint(0, len(mutated) - 1)
        mutated.pop(remove_pos)

    return mutated


def mutation_probability_change(
    previous_mutation_probability,
    all_maximum_fitnesses,
    number_of_instructions,
    time_window_size,
    probability_change_factor=2.0,
):
    initial_mutation_probability = 1.0 / (number_of_instructions * 4)
    if len(all_maximum_fitnesses) < time_window_size:
        new_mutation_probability = previous_mutation_probability
    else:
        fitness_difference = (
            all_maximum_fitnesses[-1] - all_maximum_fitnesses[-time_window_size]
        )
        if fitness_difference > 1e-6:
            new_mutation_probability = (
                previous_mutation_probability * probability_change_factor
            )
        elif fitness_difference < 1e-12:
            new_mutation_probability = initial_mutation_probability * 1.5
        else:
            new_mutation_probability = initial_mutation_probability
    return new_mutation_probability


def save_best_chromosome(chromosome, constants):
    """Save the best chromosome and constants to a file"""
    with open("best_chromosome.py", "w") as f:
        f.write("# Best chromosome found by LGP\n\n")
        f.write("def get_best_chromosome():\n")
        f.write("    return " + str(chromosome) + "\n\n")
        f.write("def get_constants():\n")
        f.write("    return " + str(constants) + "\n")


def save_checkpoint(generation, chromosome, constants):
    """Save a checkpoint of the best chromosome at the current generation"""
    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_gen_{generation}.py")
    with open(checkpoint_path, "w") as f:
        f.write(f"# Checkpoint from generation {generation}\n\n")
        f.write("def get_best_chromosome():\n")
        f.write("    return " + str(chromosome) + "\n\n")
        f.write("def get_constants():\n")
        f.write("    return " + str(constants) + "\n")

    print(f"Checkpoint saved at generation {generation}")


def cleanup_old_checkpoints(current_generation):
    """Delete checkpoints that are older than CHECKPOINT_MAX_AGE generations"""
    if not os.path.exists(CHECKPOINT_DIR):
        return

    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.startswith("checkpoint_gen_"):
            try:
                # Extract generation number from filename
                gen_number = int(filename.split("_")[2].split(".")[0])

                # Delete if older than the max age
                if current_generation - gen_number > CHECKPOINT_MAX_AGE:
                    file_path = os.path.join(CHECKPOINT_DIR, filename)
                    os.remove(file_path)
                    print(f"Removed old checkpoint from generation {gen_number}")
            except (ValueError, IndexError):
                # Skip files with unexpected naming pattern
                continue


def run_lgp():
    """Main LGP algorithm that runs until error threshold is reached"""
    # Load data
    data = load_function_data()

    # Initialize constants
    constants = initialize_constants()

    # Initialize population
    population = initialize_population(POPULATION_SIZE)

    # Evaluate initial population
    fitness_values = [evaluate_fitness(chrom, data, constants) for chrom in population]

    best_fitness = max(fitness_values)
    best_chromosome = copy.deepcopy(population[fitness_values.index(best_fitness)])

    # Keep track of all maximum fitnesses for adaptive mutation
    all_max_fitnesses = [best_fitness]

    # Initialize variables
    generation = 0
    current_error = 1.0 / best_fitness

    # Initial mutation rate
    avg_instructions = sum(len(chrom) for chrom in population) / len(population)
    mutation_rate = 1.0 / (avg_instructions * 4)  # Start with theoretical optimal rate

    # Run until error threshold is reached or max generations
    while current_error > ERROR_THRESHOLD and generation < MAX_GENERATIONS:
        generation += 1

        # Create new population
        new_population = []

        # Elitism: keep the best individual
        new_population.append(copy.deepcopy(best_chromosome))

        # Fill the rest of the population
        while len(new_population) < POPULATION_SIZE:
            if random.random() < CROSSOVER_RATE:
                # Crossover
                parent1 = tournament_selection(population, fitness_values)
                parent2 = tournament_selection(population, fitness_values)
                child1, child2 = two_point_crossover(parent1, parent2)

                # Mutate children
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)

                # Add to new population
                new_population.append(child1)
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(child2)
            else:
                # Just select and mutate
                parent = tournament_selection(population, fitness_values)
                child = mutate(parent, mutation_rate)
                new_population.append(child)

        # Update population
        population = new_population

        # Evaluate new population
        fitness_values = [
            evaluate_fitness(chrom, data, constants) for chrom in population
        ]

        # Check for new best solution
        current_best_fitness = max(fitness_values)
        current_best_chromosome = population[fitness_values.index(current_best_fitness)]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_chromosome = copy.deepcopy(current_best_chromosome)
            current_error = 1.0 / best_fitness
            print(
                f"Generation {generation}: New best fitness = {best_fitness:.6f}, Error = {current_error:.6f}, Length = {len(best_chromosome)}"
            )

        # Store the best fitness of this generation
        all_max_fitnesses.append(current_best_fitness)

        # Update mutation rate using the new function
        avg_instructions = sum(len(chrom) for chrom in population) / len(population)
        mutation_rate = mutation_probability_change(
            mutation_rate,
            all_max_fitnesses,
            avg_instructions,
            TIME_WINDOW_SIZE,
            PROBABILITY_CHANGE_FACTOR,
        )

        # Keep mutation rate within reasonable bounds
        mutation_rate = max(0.001, min(0.3, mutation_rate))

        # Print progress periodically
        if generation % 10 == 0:
            avg_fitness = sum(fitness_values) / len(fitness_values)
            print(
                f"Generation {generation}: Avg fitness = {avg_fitness:.6f}, Best fitness = {best_fitness:.6f}, Error = {current_error:.6f}, Mutation rate = {mutation_rate:.6f}"
            )

        # Save checkpoint every CHECKPOINT_INTERVAL generations
        if generation % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(generation, best_chromosome, constants)
            cleanup_old_checkpoints(generation)

    # Final output
    if current_error <= ERROR_THRESHOLD:
        print(f"Success! Error threshold reached after {generation} generations.")
    else:
        print(f"Maximum generations reached. Best error: {current_error:.6f}")

    print(f"Final best fitness: {best_fitness:.6f}, Error: {current_error:.6f}")
    print(f"Best chromosome length: {len(best_chromosome)}")

    # Save best chromosome
    save_best_chromosome(best_chromosome, constants)

    return best_chromosome, constants


if __name__ == "__main__":
    best_chromosome, constants = run_lgp()
