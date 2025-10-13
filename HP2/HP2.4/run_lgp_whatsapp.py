# run_lgp.py

import random
import math
import copy
from function_data import load_function_data

# --- Constants and Parameters ---
# GA Parameters
POPULATION_SIZE = 250
MAX_GENERATIONS = 2000  
TOURNAMENT_SIZE = 5
P_CROSSOVER = 0.8

# --- Hybrid Adaptive Control Parameters ---
HIGH_ERROR_THRESHOLD = 0.20    # Any error above this is "bad"
LOW_ERROR_THRESHOLD = 0.01     # Any error below this is "good"
MUTATION_AT_HIGH_ERROR = 6.0   # The default mutation rate when solutions are bad
MUTATION_AT_LOW_ERROR = 0.5    # The default rate when we are close

# Progress-Based Adaptation (as multipliers)
STAGNATION_THRESHOLD = 75      # Generations without improvement before heating up
HEAT_UP_MULTIPLIER = 1.5       # Multiplier to boost the current base rate
COOL_DOWN_MULTIPLIER = 0.9     # Multiplier to cool the current base rate

# --- Parameters for significant improvement ---
# Only consider an improvement "real" if it's at least 0.1% better
SIGNIFICANT_IMPROVEMENT_THRESHOLD = 0.001 

# LGP Parameters
NUM_VARIABLE_REGISTERS = 4  # M
NUM_CONSTANT_REGISTERS = 4  # N
CONSTANT_VALUES = [1.0, -1.0, 2.0, 0.0]

# Chromosome length constraints
MIN_INITIAL_INSTRUCTIONS = 10
MAX_INITIAL_INSTRUCTIONS = 30
MAX_INSTRUCTIONS = 100 

# --- Operator Set ---
def protected_division(a, b):
    if abs(b) < 1e-6:
        return 1.0
    return a / b

OPERATORS = {
    0: ('+', lambda a, b: a + b),
    1: ('-', lambda a, b: a - b),
    2: ('*', lambda a, b: a * b),
    3: ('/', protected_division)
}
NUM_OPERATORS = len(OPERATORS)


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = 0.0

    def __len__(self):
        return len(self.chromosome)

# --- Core LGP Functions ---

def create_random_instruction():
    op = random.randint(0, NUM_OPERATORS - 1)
    dest_reg = random.randint(0, NUM_VARIABLE_REGISTERS - 1)
    op1_reg = random.randint(0, NUM_VARIABLE_REGISTERS + NUM_CONSTANT_REGISTERS - 1)
    op2_reg = random.randint(0, NUM_VARIABLE_REGISTERS + NUM_CONSTANT_REGISTERS - 1)
    return [op, dest_reg, op1_reg, op2_reg]

def create_random_chromosome():
    num_instructions = random.randint(MIN_INITIAL_INSTRUCTIONS, MAX_INITIAL_INSTRUCTIONS)
    return [gene for _ in range(num_instructions) for gene in create_random_instruction()]

def initialize_population():
    return [Individual(create_random_chromosome()) for _ in range(POPULATION_SIZE)]

def get_operand_value(reg_index, var_regs, const_regs):
    if reg_index < NUM_VARIABLE_REGISTERS:
        return var_regs[reg_index]
    else:
        return const_regs[reg_index - NUM_VARIABLE_REGISTERS]

def evaluate_chromosome(chromosome, x_input):
    variable_registers = [0.0] * NUM_VARIABLE_REGISTERS
    variable_registers[0] = x_input 

    num_instructions = len(chromosome) // 4
    for i in range(num_instructions):
        instr = chromosome[i*4 : (i+1)*4]
        op_idx, dest_idx, op1_idx, op2_idx = instr

        op_func = OPERATORS[op_idx][1]
        val1 = get_operand_value(op1_idx, variable_registers, CONSTANT_VALUES)
        val2 = get_operand_value(op2_idx, variable_registers, CONSTANT_VALUES)

        result = op_func(val1, val2)
        variable_registers[dest_idx] = result
        
    return variable_registers[0] 

def calculate_fitness(individual, data):
    squared_error_sum = 0.0
    MAX_ERROR = 1e10  
    
    for x, y_true in data:
        y_pred = evaluate_chromosome(individual.chromosome, x)
        
        # Protect against overflow and invalid values
        if not math.isfinite(y_pred):
            individual.fitness = 0.0
            return
        
        error = y_pred - y_true
        # Cap the error to prevent overflow in squaring
        if abs(error) > MAX_ERROR:
            squared_error_sum += MAX_ERROR ** 2
        else:
            squared_error_sum += error ** 2
    
    rmse = math.sqrt(squared_error_sum / len(data))
    
    # Add a small epsilon to avoid division by zero for a perfect fit
    individual.fitness = 1.0 / (rmse + 1e-9)

def tournament_selection(population):
    tournament = random.sample(population, TOURNAMENT_SIZE)
    return max(tournament, key=lambda ind: ind.fitness)

def crossover(parent1, parent2):
    p1_genes = parent1.chromosome
    p2_genes = parent2.chromosome

    len1_instr = len(p1_genes) // 4
    len2_instr = len(p2_genes) // 4
    
    if len1_instr < 2 or len2_instr < 2: 
        return Individual(p1_genes), Individual(p2_genes)

    pt1_1, pt1_2 = sorted(random.sample(range(1, len1_instr), 2))
    pt2_1, pt2_2 = sorted(random.sample(range(1, len2_instr), 2))

    pt1_1 *= 4; pt1_2 *= 4
    pt2_1 *= 4; pt2_2 *= 4

    child1_genes = p1_genes[:pt1_1] + p2_genes[pt2_1:pt2_2] + p1_genes[pt1_2:]
    child2_genes = p2_genes[:pt2_1] + p1_genes[pt1_1:pt1_2] + p2_genes[pt2_2:]
    
    return Individual(child1_genes), Individual(child2_genes)


def mutate_with_rate(individual, p_mutation_gene):
    for i in range(len(individual.chromosome)):
        if random.random() < p_mutation_gene:
            # Re-generate the gene based on its position in the instruction
            gene_pos = i % 4
            if gene_pos == 0: # Operator
                individual.chromosome[i] = random.randint(0, NUM_OPERATORS - 1)
            elif gene_pos == 1: # Destination register
                individual.chromosome[i] = random.randint(0, NUM_VARIABLE_REGISTERS - 1)
            else: # Operand register
                individual.chromosome[i] = random.randint(0, NUM_VARIABLE_REGISTERS + NUM_CONSTANT_REGISTERS - 1)


def main():
    """Main LGP execution loop."""
    data = load_function_data()
    population = initialize_population()
    best_overall_individual = None
    
    generations_without_improvement = 0
    adaptive_multiplier = 1.0 

    for gen in range(MAX_GENERATIONS):
        # Evaluate fitness of the entire population
        for ind in population:
            calculate_fitness(ind, data)
        
        # Find the best individual in the current generation
        best_gen_individual = max(population, key=lambda ind: ind.fitness)

       
        # 1. Determine the GLOBAL base rate based on current quality
        current_best_error = 1.0 / best_overall_individual.fitness if best_overall_individual else float('inf')

        if current_best_error > HIGH_ERROR_THRESHOLD:
            global_base_rate = MUTATION_AT_HIGH_ERROR
        elif current_best_error < LOW_ERROR_THRESHOLD:
            global_base_rate = MUTATION_AT_LOW_ERROR
        else:
            # Linearly interpolate between the high and low rates
            progress = (HIGH_ERROR_THRESHOLD - current_best_error) / (HIGH_ERROR_THRESHOLD - LOW_ERROR_THRESHOLD)
            global_base_rate = MUTATION_AT_HIGH_ERROR - progress * (MUTATION_AT_HIGH_ERROR - MUTATION_AT_LOW_ERROR)

        # 2. Check for stagnation/progress to adjust the LOCAL multiplier
        improvement_threshold = best_overall_individual.fitness * SIGNIFICANT_IMPROVEMENT_THRESHOLD if best_overall_individual else 0
        if best_overall_individual is None or best_gen_individual.fitness > best_overall_individual.fitness + improvement_threshold:
            # PROGRESS - cool down the adaptive multiplier
            adaptive_multiplier = max(0.5, adaptive_multiplier * COOL_DOWN_MULTIPLIER)  # Don't let it go to zero
            best_overall_individual = copy.deepcopy(best_gen_individual)
            generations_without_improvement = 0
            
            # Calculate final mutation rate for display
            current_mutation_base = global_base_rate * adaptive_multiplier
            print(f"Gen {gen}:  NEW BEST!  Fitness = {best_overall_individual.fitness:.6f}, "
                  f"Error = {1/best_overall_individual.fitness:.6f}, "
                  f"Length = {len(best_overall_individual.chromosome)//4}, "
                  f"Global Rate = {global_base_rate:.2f}, Multiplier = {adaptive_multiplier:.2f}, "
                  f"Final Mutation Base = {current_mutation_base:.3f}")
        else:
            # STAGNATION - heat up the adaptive multiplier
            generations_without_improvement += 1
            if generations_without_improvement > STAGNATION_THRESHOLD:
                adaptive_multiplier = min(2.0, adaptive_multiplier * HEAT_UP_MULTIPLIER)  # Don't let it get insane
                generations_without_improvement = 0
                
                # Calculate final mutation rate for display
                current_mutation_base = global_base_rate * adaptive_multiplier
                print(f"Gen {gen}: Stagnation detected. "
                      f"Global Rate = {global_base_rate:.2f}, Multiplier = {adaptive_multiplier:.2f}, "
                      f"Heating up to {current_mutation_base:.3f}")
        
        # 3. Calculate the FINAL mutation base for this generation
        current_mutation_base = global_base_rate * adaptive_multiplier
        
        # Print periodic status updates
        if gen % 20 == 0:
            print(f"Gen {gen}: Error = {current_best_error:.6f}, "
                  f"Global Rate = {global_base_rate:.2f}, Multiplier = {adaptive_multiplier:.2f}, "
                  f"Final Mutation Base = {current_mutation_base:.3f}")

        # Create the next generation
        next_generation = []
        while len(next_generation) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            if random.random() < P_CROSSOVER:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            if len(child1.chromosome) > MAX_INSTRUCTIONS * 4:
                continue # Discard and try again
            if len(child2.chromosome) > MAX_INSTRUCTIONS * 4:
                continue # Discard and try again

            # Use the current, decaying mutation rate
            p_mutation_gene = current_mutation_base / len(child1.chromosome)
            mutate_with_rate(child1, p_mutation_gene)
            
            p_mutation_gene = current_mutation_base / len(child2.chromosome)
            mutate_with_rate(child2, p_mutation_gene)
            
            next_generation.append(child1)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(child2)

        population = next_generation

    with open("best_chromosome.py", "w") as f:
        f.write(f"best_chromosome = {best_overall_individual.chromosome}\n")

    print("\nOptimization finished.")
    print(f"Best chromosome saved to best_chromosome.py")
    print(f"Final best error: {1/best_overall_individual.fitness:.6f}")


if __name__ == "__main__":
    main()