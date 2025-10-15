import math
import numpy as np
import matplotlib.pyplot as plt
import random
import os


def initialize_pheromone_levels(number_of_cities, tau_0):
    pheromone_matrix = np.full((number_of_cities, number_of_cities), tau_0)
    return pheromone_matrix


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


def get_next_node(potential_cities):
    s = sorted(potential_cities, key=lambda x: x[1], reverse=True)
    rand_val = random.random()
    cumulative_prob = 0
    for index, prob in s:
        cumulative_prob += prob
        if rand_val <= cumulative_prob:
            return index
    return s[0][0]


def generate_path(pheromone_levels, visibility, alpha, beta):
    number_of_cities = len(pheromone_levels)
    current_node = random.randint(0, number_of_cities - 1)
    tabu_list = [current_node]
    while len(tabu_list) < number_of_cities:
        denominator = 0
        for j in range(number_of_cities):
            if j not in tabu_list:
                denominator += (
                    pheromone_levels[current_node][j] ** alpha
                    * visibility[current_node][j] ** beta
                )
        if denominator == 0:
            break
        potential_cities = []
        for city in range(number_of_cities):
            if city not in tabu_list:
                numerator = (
                    pheromone_levels[current_node][city] ** alpha
                    * visibility[current_node][city] ** beta
                )
                probability = numerator / denominator
                potential_cities.append([city, probability])
        next_node = get_next_node(potential_cities)
        tabu_list.append(next_node)
        current_node = next_node
    path = tabu_list
    return path


def get_path_length(path, city_locations):
    if len(path) < 2:
        return 0.0
    path_length = 0.0
    for i in range(len(path) - 1):
        x1, y1 = city_locations[path[i]]
        x2, y2 = city_locations[path[i + 1]]
        path_length += math.dist((x1, y1), (x2, y2))
    x_first, y_first = city_locations[path[0]]
    x_last, y_last = city_locations[path[-1]]
    path_length += math.dist((x_last, y_last), (x_first, y_first))
    return path_length


def compute_delta_pheromone_levels(path_collection, path_length_collection):
    number_of_cities = len(path_collection[0])
    delta_pheromone_levels = np.zeros((number_of_cities, number_of_cities))
    for ant, path in enumerate(path_collection):
        L = path_length_collection[ant]
        if L == 0:
            continue
        deposit = 1.0 / L
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            delta_pheromone_levels[a][b] += deposit
        a = path[-1]
        b = path[0]
        delta_pheromone_levels[a][b] += deposit
    return delta_pheromone_levels


def update_pheromone_levels(pheromone_levels, delta_pheromone_levels, rho):
    tau_min = 1e-15
    new_pheromone_levels = (1 - rho) * pheromone_levels + delta_pheromone_levels
    np.maximum(new_pheromone_levels, tau_min, out=new_pheromone_levels)
    return new_pheromone_levels


def plot_cities(plt, city_locations):
    x = []
    y = []
    for city_index in range(len(city_locations)):
        x.append(city_locations[city_index][0])
        y.append(city_locations[city_index][1])
    plt.scatter(x, y, zorder=1, color="yellow")


def plot_path(plt, path):
    connections_x = []
    connections_y = []
    for index in path:
        location_x = city_locations[index][0]
        connections_x.append(location_x)
        location_y = city_locations[index][1]
        connections_y.append(location_y)
    start_location_x = city_locations[path[0]][0]
    start_location_y = city_locations[path[0]][1]
    connections_x.append(start_location_x)
    connections_y.append(start_location_y)
    plt.plot(connections_x, connections_y, color="lime", zorder=0)


from city_data import city_locations

number_of_cities = len(city_locations)

number_of_ants = 50
alpha = 1.0
beta = 5.0
rho = 0.5
tau_0 = 0.1
target_path_length = 99.9999999

plt.ion()
plt.figure(figsize=(10, 8))
plot_range = 20
plt.xlim(0, plot_range)
plt.ylim(0, plot_range)
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
ax.set_facecolor("xkcd:black")
plot_cities(plt, city_locations)
plt.title("Ant System TSP - Initialization")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()
plt.pause(0.1)

pheromone_levels = initialize_pheromone_levels(number_of_cities, tau_0)
visibility = get_visibility(city_locations)

iteration_index = 0
minimum_path_length = math.inf
path_length = math.inf
best_path_history = []

while minimum_path_length > target_path_length:
    iteration_index += 1
    path_collection = []
    path_length_collection = []
    for ant_index in range(number_of_ants):
        path = generate_path(pheromone_levels, visibility, alpha, beta)
        path_length = get_path_length(path, city_locations)
        if path_length < minimum_path_length:
            minimum_path_length = path_length
            print(minimum_path_length)
            plt.clf()
            plt.xlim(0, plot_range)
            plt.ylim(0, plot_range)
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")
            ax.set_facecolor("xkcd:black")
            plot_cities(plt, city_locations)
            plot_path(plt, path)
            plt.title(
                f"Ant System TSP - Best Path Length: {minimum_path_length:.2f}\nIteration: {iteration_index}"
            )
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.pause(0.1)
        path_collection.append(path)
        path_length_collection.append(path_length)
    delta_pheromone_levels = compute_delta_pheromone_levels(
        path_collection, path_length_collection
    )
    pheromone_levels = update_pheromone_levels(
        pheromone_levels, delta_pheromone_levels, rho
    )
    best_path_history.append(minimum_path_length)

plt.clf()
plt.xlim(0, plot_range)
plt.ylim(0, plot_range)
ax = plt.gca()
ax.set_aspect("equal", adjustable="box")
ax.set_facecolor("xkcd:black")
best_path_index = path_length_collection.index(min(path_length_collection))
best_path = path_collection[best_path_index]
plot_cities(plt, city_locations)
plot_path(plt, best_path)
plt.title(
    f"Final Solution - Best Path Length: {minimum_path_length:.2f}\nTotal Iterations: {iteration_index}"
)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()


with open("best_result_found.py", "w", encoding="utf-8") as f:
    f.write(f"best_path = {best_path}\n")
    f.write(f"best_path_length = {minimum_path_length:.6f}\n")

print(f"Optimization completed!")
print(f"Best path length found: {minimum_path_length:.2f}")
print(f"Total iterations: {iteration_index}")
print(f"Target was: {target_path_length}")
plt.savefig("ant_best_path.png", dpi=300, bbox_inches="tight")

input(f"Press return to exit")
