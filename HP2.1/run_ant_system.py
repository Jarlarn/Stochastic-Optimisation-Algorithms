###########################
#
# Ant System (AS) for TSP
#
###########################

import math
import numpy as np

# import matplotlib.pyplot as plt
import random


###############################################################
## To do: Write the initialize_pheromone_levels function:
###############################################################


def initialize_pheromone_levels(number_of_cities, tau_0):
    pheromone_matrix = np.full((number_of_cities, number_of_cities), tau_0)
    return pheromone_matrix


###############################################################
## To do: Write the get_visibility function:
###############################################################


def get_visibility(city_locations):
    number_of_cities = len(city_locations)
    visibility_matrix = np.zeros((number_of_cities, number_of_cities))
    for i in range(number_of_cities):
        for j in range(number_of_cities):
            if i != j:
                x1, y1 = city_locations[j]
                x2, y2 = city_locations[i]
                distance = math.dist((x1, y1), (x2, y2))
                visibility_matrix[i][j] = 1.0 / distance

            else:
                visibility_matrix[i][j] = 0.0
    return visibility_matrix


#################################################################
## To do: Write the generate_path function (Note: You may wish
##       to add more functions, e.g., get_node. That is allowed).
#################################################################


def get_next_node(potential_cities):

    # Sort by probability in descending order with original indices
    s = sorted(potential_cities, key=lambda x: x[1], reverse=True)

    # Roulette wheel selection
    rand_val = random.random()
    cumulative_prob = 0
    for index, prob in s:
        cumulative_prob += prob
        if rand_val <= cumulative_prob:
            return index

    return s[0][0]  # Fallback


def generate_path(pheromone_levels, visibility, alpha, beta):
    number_of_cities = len(pheromone_levels)
    # current_node = random.randint(0, number_of_cities - 1)
    current_node = 2
    tabu_list = [current_node]

    while len(tabu_list) < number_of_cities:
        # Calculate denominator for current node to all unvisited cities
        denominator = 0
        for j in range(number_of_cities):
            if j not in tabu_list:
                denominator += (
                    pheromone_levels[current_node][j] ** alpha
                    * visibility[current_node][j] ** beta
                )
        if denominator == 0:
            break

        # Calculate probabilities ONLY for unvisited cities
        potential_cities = []
        for city in range(number_of_cities):
            if city not in tabu_list:
                numerator = (
                    pheromone_levels[current_node][city] ** alpha
                    * visibility[current_node][city] ** beta
                )
                probability = numerator / denominator
                potential_cities.append([city, probability])

        # Select next node using only unvisited cities
        next_node = get_next_node(potential_cities)
        tabu_list.append(next_node)
        current_node = next_node
    path = tabu_list
    return path


###############################################################
## To do: Write the get_path_length function:
###############################################################


def get_path_length(path, city_locations):

    path_length = 0
    if len(path) < 2:
        return path_length
    for i in range(len(path) - 1):
        x1, y1 = city_locations[path[i]]
        x2, y2 = city_locations[path[i + 1]]
        path_length += math.dist((x1, y1), (x2, y2))
    return path_length


# Add code here!

###############################################################
## To do: Write the compute_delta_pheromone_levels function:
###############################################################


def compute_delta_pheromone_levels(path_collection, path_length_collection):
    delta_pheromone_levels = [] 
    total_pheremone = 0
    for ant, path in enumerate(path_collection):
        path_length = path_length_collection[ant]
        pheremone_deposit = 1 / path_length
        for i in range(len(path)-1):
            city_from = path[i]
            next_city = path[i + 1]

    return total_pheremone


# Add code here!

###############################################################
## To do: Write the update_pheromone_levels function:
###############################################################


def update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho):
    pass


# Add code here!

##################################################
#  Plots the cities (nodes):
##################################################

# Add plot code here (can be more than one function)

#####################################
# Main program:
#####################################

###########################
# Data:
###########################
from city_data import city_locations

number_of_cities = len(city_locations)

###########################
# Parameters:
###########################
number_of_ants = 50  ## Changes allowed.
alpha = 1.0  ## Changes allowed.
beta = 5.0  ## Changes allowed.
rho = 0.5  ## Changes allowed.
tau_0 = 0.1  ## Changes allowed.

# target_path_length = 99.9999999
target_path_length = 120
#################################
# Initialization:
#################################

## To do: Add plot initialization here


pheromone_levels = initialize_pheromone_levels(number_of_cities, tau_0)
visibility = get_visibility(city_locations)

#################################
# Main loop:
#################################

iteration_index = 0
minimum_path_length = math.inf
path_length = math.inf


while minimum_path_length > target_path_length:
    iteration_index += 1
    path_collection = []
    path_length_collection = []
    for ant_index in range(number_of_ants):
        # Generate paths:
        path = generate_path(
            pheromone_levels, visibility, alpha, beta
        )  # Uncomment after writing the function
        path_length = get_path_length(
            path, city_locations
        )  # Uncomment after writing the function
        if path_length < minimum_path_length:
            minimum_path_length = path_length
            print(minimum_path_length)

            # To do: Add code for plotting here

        path_collection.append(path)
        path_length_collection.append(path_length)
    # Update pheromone levels:
    delta_pheromone_levels = compute_delta_pheromone_levels(
        path_collection, path_length_collection
    )  # Uncomment after writing the function
    # pheromone_levels = update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho) # Uncomment after writing the function

# input(f"Press return to exit")

# OWN TEST CODE
# path = generate_path(pheromone_levels, visibility, alpha, beta)

# get_path_length(path, city_locations)
# # Uncomment after writing the function
