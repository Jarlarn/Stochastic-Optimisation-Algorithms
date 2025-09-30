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
    visibility_matrix = np.zeros((number_of_cities,number_of_cities))

    for i in range(visibility_matrix):
        for j in range(visibility_matrix):
            


# Add code here!


#################################################################
## To do: Write the generate_path function (Note: You may wish
##       to add more functions, e.g., get_node. That is allowed).
#################################################################


def get_node():
    pass


def generate_path(pheromone_levels, visibility, alpha, beta):
    pass


# Add code here!

###############################################################
## To do: Write the get_path_length function:
###############################################################


def get_path_length(path, city_locations):
    pass


# Add code here!

###############################################################
## To do: Write the compute_delta_pheromone_levels function:
###############################################################


def compute_delta_pheromone_levels(path_collection, path_length_collection):
    pass


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

target_path_length = 99.9999999

#################################
# Initialization:
#################################

## To do: Add plot initialization here


pheromone_levels = initialize_pheromone_levels(number_of_cities, tau_0)
visibility = get_visibility(city_locations)
print(visibility)

#################################
# Main loop:
#################################

iteration_index = 0
minimum_path_length = math.inf
path_length = math.inf


# while minimum_path_length > target_path_length:
#     iteration_index += 1
#     path_collection = []
#     path_length_collection = []
#     for ant_index in range(number_of_ants):
#         # Generate paths:
#         # path = generate_path(pheromone_levels, visibility, alpha, beta) # Uncomment after writing the function
#         # path_length = get_path_length(path, city_locations) # Uncomment after writing the function
#         if path_length < minimum_path_length:
#             minimum_path_length = path_length
#             print(minimum_path_length)

#             # To do: Add code for plotting here

#         path_collection.append(path)
#         path_length_collection.append(path_length)
#     # Update pheromone levels:
#     # delta_pheromone_levels = compute_delta_pheromone_levels(path_collection,path_length_collection) # Uncomment after writing the function
#     # pheromone_levels = update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho) # Uncomment after writing the function

# input(f"Press return to exit")
