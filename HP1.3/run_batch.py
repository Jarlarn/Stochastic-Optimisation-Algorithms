import math
from genetic_algorithm import run_function_optimization
import matplotlib.pyplot as plt

number_of_runs = 100                # Do NOT change
population_size = 100               # Do NOT change
maximum_variable_value = 5          # Do NOT change: (x_i in [-a,a], where a = maximumVariableValue)
number_of_genes = 50                # Do NOT change
number_of_variables = 2  	    # Do NOT change
tournament_size = 2                 # Do NOT change
tournament_probability = 0.75       # Do NOT change
crossover_probability = 0.8         # Do NOT change
number_of_generations = 2000        # Do NOT change


mutation_probability_list = [0, 0.005, 0.010, 0.020, 0.05, 0.1] # Add more values in this list; see the problem sheet

# Below you should add the required code for the statistical analysis 
# (computing median fitness values, and so on), as described in the problem sheet
avg_fitness_dict = {}
for mutation_probability in mutation_probability_list:
   print("=====================================")
   print(f"mutation probability: {mutation_probability:.3f}")
   print("=====================================")
   fitness_list = []
   for run_index in range(number_of_runs):
      [maximum_fitness, x_best] = run_function_optimization(population_size, number_of_genes, number_of_variables, maximum_variable_value, tournament_size, \
                                       tournament_probability, crossover_probability, mutation_probability, number_of_generations);
      output = f"Run index: {run_index}, fitness: {maximum_fitness:.4f}, x = ({x_best[0]:.8f},{x_best[1]:.8f})"
      fitness_list.append(maximum_fitness)
      print(output)
   avg_fitness = sum(fitness_list) / len(fitness_list)
   avg_fitness_object = {mutation_probability:avg_fitness}
   avg_fitness_dict.update(avg_fitness_object)
   print(f"Average fitness for mutation probability ({mutation_probability:.3f}) = {avg_fitness}")
for mp,avg in avg_fitness_dict.items():
   print(f"Mutation probability: {mp} | Average fitness: {avg:.4f}")


xs = sorted(avg_fitness_dict)
ys = [avg_fitness_dict[x] for x in xs]
plt.figure(figsize=(7, 4))
plt.plot(xs, ys, marker='o')
plt.xlabel('mutation_probability')
plt.ylabel('avg_fitness')
plt.title('Average fitness vs mutation probability')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('avg_fitness_vs_mutation_probability.png', dpi=150)
plt.show()