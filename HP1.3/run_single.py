import math
from genetic_algorithm import run_function_optimization
population_size = 100               # Do NOT change
maximum_variable_value = 5          # Do NOT change: (x_i in [-a,a], where a = maximumVariableValue)
number_of_genes = 50                # Do NOT change
number_of_variables = 2  	    # Do NOT change

tournament_size = 2                 # Changes allowed
tournament_probability = 0.75       # Changes allowed
crossover_probability = 0.8     # Changes allowed
mutation_probability = 0.02         # Changes allowed. (Note: 0.02 <=> 1/numberOfGenes)
number_of_generations = 2000        # Changes allowed.


[maximum_fitness, x_best] = run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value, tournament_size, \
                                       tournament_probability, crossover_probability, mutation_probability, number_of_generations);
#Own line
output = f"Fitness: {maximum_fitness:.4f}, x = ({x_best[0]:.8f},{x_best[1]:.8f})"
print(output)
