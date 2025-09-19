import random
import math
import matplotlib.pyplot as plt

# Initialize population:
def initialize_population(population_size, number_of_genes):

    population = [[random.randint(0,1) for gene_index in range(number_of_genes)] for chromosome_index in range(population_size)]
    return population

# Decode chromosome:
def decode_chromosome(chromosome,variable_range):

    number_of_genes = len(chromosome)
    n_half = int(number_of_genes/2)

    x_1 = 0
    for j in range(n_half):
        x_1 += chromosome[j]*pow(2,-j-1)
    x_1 = -variable_range + 2*variable_range*x_1 / ( 1 - pow(2,-n_half))
    
    x_2 = 0
    for j in range(n_half):
        x_2 += chromosome[j+n_half]*pow(2,-j-1)
    x_2 = -variable_range + 2*variable_range*x_2 / ( 1 - pow(2,-n_half))

    x = [x_1,x_2]
    return x
    
# Evaluate indviduals:
def evaluate_individual(x):

    x_1 = x[0]
    x_2 = x[1]

    f_numerator1 = math.exp(-x_1**2 - x_2**2)
    f_numerator2 = math.sqrt(5)*math.sin(x_2*x_1**2)
    f_numerator3 = 2*(math.cos(2*x_1 + 3*x_2)**2)

    f_denominator = 1 + x_1**2 + x_2**2
    
    f = (f_numerator1 + f_numerator2 + f_numerator3)/f_denominator

    return f

# Select individuals:
def tournament_select(fitness_list, tournament_selection_parameter):
    
    population_size = len(fitness_list)
    i_1 = random.randint(0,population_size-1)
    i_2 = random.randint(0,population_size-1)

    r = random.random()

    if r < tournament_selection_parameter:
        if (fitness_list[i_1] > fitness_list[i_2]):
            i_selected = i_1
        else:
            i_selected = i_2    
    else:
        if (fitness_list[i_1] > fitness_list[i_2]):
            i_selected = i_2
        else:
            i_selected = i_1

    return i_selected

# Carry out crossover:
def cross(chromsome1, chromosome2):

    number_of_genes = len(chromosome1)
    cross_point = random.randint(1,number_of_genes-1)
    
    new_chromosome1 = chromsome1[:cross_point] + chromosome2[cross_point:]
    new_chromosome2 =  chromosome2[:cross_point] + chromsome1[cross_point:]

    return [new_chromosome1,new_chromosome2]

# Mutate individuals:
def mutate(chromosome,mutation_probabillity):
    
    number_of_genes = len(chromosome)
    mutated_chromosome = chromosome.copy()
    
    for gene_index in range(number_of_genes):
        r = random.random()
        if r < mutation_probabillity:
            mutated_chromosome[gene_index] = 1-chromosome[gene_index]
        
        return mutated_chromosome

# Main program:

population_size = 30
number_of_genes = 40
crossover_probabillity = 0.8
mutation_probabillity = 0.025
tournament_selection_parameter = 0.75
variable_range = 3
number_of_generations = 100

maximum_fitness_list = []
population = initialize_population(population_size,number_of_genes)
for generation_index in range(number_of_generations):


    maximum_fitness = 0
    best_chromosome = []
    best_individual = []
    fitness_list = []
    for chromosome in population:
        individual = decode_chromosome(chromosome,variable_range)
        fitness = evaluate_individual(individual)
        if (fitness > maximum_fitness):
            maximum_fitness = fitness
            best_chromosome = chromosome.copy()
            best_individual = individual.copy()
        fitness_list.append(fitness)
    maximum_fitness_list.append(maximum_fitness)

    temp_population = []
    for i in range(0,population_size,2):
        index_1 = tournament_select(fitness_list,tournament_selection_parameter)
        index_2= tournament_select(fitness_list,tournament_selection_parameter)
        chromosome1 = population[index_1].copy()
        chromosome2  = population[index_2].copy()

        r = random.random()
        if (r < crossover_probabillity):
            [new_chromosome1, new_chromosome2] = cross(chromosome1,chromosome2)
            temp_population.append(new_chromosome1)
            temp_population.append(new_chromosome2)
        else:
            temp_population.append(chromosome1)
            temp_population.append(chromosome2)
    
    for i in range(0,population_size):
        original_chromosome = temp_population[i]

        mutated_chromosome = mutate(original_chromosome,mutation_probabillity)
        temp_population[i] = mutated_chromosome

    

    temp_population[0] = best_chromosome
    population = temp_population.copy()

print(f"""maximum fitness: {maximum_fitness:.7f}, 
        best individual: ({best_individual[0]:.7f},
         {best_individual[1]:.7f})""")

plt.plot(maximum_fitness_list)
plt.axis([0,100,0,3])
plt.xlabel("Generation")
plt.grid()
plt.ylabel("Fitness")
plt.show()