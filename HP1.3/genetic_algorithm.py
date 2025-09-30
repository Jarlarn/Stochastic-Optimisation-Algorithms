import random
import math


# Initialize population:
# Uncomment the line below and implement the function
def initialize_population(population_size, number_of_genes):
    population = [
        [random.randint(0, 1) for gene_index in range(number_of_genes)]
        for chromosome_index in range(population_size)
    ]
    return population


# Decode chromosome:
# Note: the variables should each take values in the range [-a,a], where a = maximum_variable_value
# Uncomment the line below and implement the function
def decode_chromosome(chromosome, number_of_variables, maximum_variable_value):

    m = len(chromosome)
    n = number_of_variables
    k = m // n
    a = maximum_variable_value

    variables = []

    for var_index in range(number_of_variables):

        start = var_index * k
        end = start + k
        bits = chromosome[start:end]
        weighted_sum = 0.0

        for index, gene in enumerate(bits, start=1):
            weighted_sum += gene * (2**-index)

        x = -a + (2 * a / (1 - 2**-k)) * weighted_sum
        variables.append(x)

    return variables


# Evaluate indviduals:
# Note: You may hard-code the evaluation of the fitness (g(x_1,g_x2)+1)^(-1)) here
# Uncomment the line below and implement the function
def evaluate_individual(x):

    x1, x2 = float(x[0]), float(x[1])

    term1 = 1.5 - x1 + x1 * x2
    term2 = 2.25 - x1 + x1 * (x2**2)
    term3 = 2.625 - x1 + x1 * (x2**3)

    g = term1**2 + term2**2 + term3**2

    if (g + 1) == 0:
        return 0

    fitness_value = 1 / (g + 1)

    return fitness_value


# Select individuals:
# Uncomment the line below and implement the function
def tournament_select(fitness_list, tournament_probability, tournament_size):
    population_size = len(fitness_list)
    contestants = []
    for _ in range(tournament_size):
        contestant = random.randint(0, population_size - 1)
        contestants.append(contestant)

    contestants.sort(key=lambda idx: fitness_list[idx], reverse=True)

    for contestant in contestants:
        if random.random() < tournament_probability:
            return contestant

    return contestants[0]


# Carry out crossover:
# Uncomment the line below and implement the function
def cross(chromosome1, chromosome2):
    number_of_genes = len(chromosome1)
    cross_point = random.randint(1, number_of_genes - 1)

    new_chromosome1 = chromosome1[:cross_point] + chromosome2[cross_point:]
    new_chromosome2 = chromosome2[:cross_point] + chromosome1[cross_point:]

    return [new_chromosome1, new_chromosome2]


# Mutate individuals:
# Uncomment the line below and implement the function
def mutate(chromosome, mutation_probability):
    number_of_genes = len(chromosome)
    mutated_chromosome = chromosome.copy()

    for gene_index in range(number_of_genes):
        r = random.random()
        if r < mutation_probability:
            mutated_chromosome[gene_index] = 1 - chromosome[gene_index]

    return mutated_chromosome


# Genetic algorithm


def run_function_optimization(
    population_size,
    number_of_genes,
    number_of_variables,
    maximum_variable_value,
    tournament_size,
    tournament_probability,
    crossover_probability,
    mutation_probability,
    number_of_generations,
):

    # This function should return the maximum fitness and the best individual (i.e., a vector with
    # two elements (x1,x2) containing the values corresponding to the maximum fitness found.

    # Note that some parameters have different names compared to the programming introduction

    population = initialize_population(population_size, number_of_genes)

    for generation_index in range(number_of_generations):
        maximum_fitness = 0
        best_chromosome = []
        best_individual = []
        fitness_list = []
        for chromosome in population:
            individual = decode_chromosome(
                chromosome, number_of_variables, maximum_variable_value
            )
            fitness = evaluate_individual(individual)
            if fitness > maximum_fitness:
                maximum_fitness = fitness
                best_chromosome = chromosome.copy()
                best_individual = individual.copy()
            fitness_list.append(fitness)

        temp_population = []
        for i in range(0, population_size, 2):
            index_1 = tournament_select(
                fitness_list, tournament_probability, tournament_size
            )
            index_2 = tournament_select(
                fitness_list, tournament_probability, tournament_size
            )
            chromosome1 = population[index_1].copy()
            chromosome2 = population[index_2].copy()
            r = random.random()
            if r < crossover_probability:
                [new_chromosome_1, new_chromosome_2] = cross(chromosome1, chromosome2)
                temp_population.append(new_chromosome_1)
                temp_population.append(new_chromosome_2)
            else:
                temp_population.append(chromosome1)
                temp_population.append(chromosome2)

        for i in range(population_size):
            original_chromosome = temp_population[i]

            mutated_chromosome = mutate(original_chromosome, mutation_probability)
            temp_population[i] = mutated_chromosome

        temp_population[0] = best_chromosome
        population = temp_population.copy()

    return [maximum_fitness, best_individual]
