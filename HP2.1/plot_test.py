########################################
#
# Simple illustration of how to animate
# the plot for HP2.1. 
#
# Note: The animation method below is quite crude.
# If you prefer, you may use a more sophisticated approach,
# for example with FuncAnimation() (under matplotlib.animation)
#
########################################

import numpy as np
import matplotlib.pyplot as plt
import random

##################################################
#  Plots the cities (nodes):
##################################################

def plot_cities(plt, city_locations):
  # or use numpy if you prefer...
  x = []
  y = []
  for city_index in range(len(city_locations)):
    x.append(city_locations[city_index][0])
    y.append(city_locations[city_index][1])
  plt.scatter(x,y,zorder=1,color='yellow')

##################################################
#  Plots the path (connections between nodes):
##################################################
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
  plt.plot(connections_x,connections_y,color='lime',zorder=0)

######################################################
# Main program
######################################################

plot_range = 20
number_of_iterations = 50
pause_interval = 0.05

# Load data
from city_data import city_locations
number_of_cities = len(city_locations)

# Generate a random path:
path = np.random.permutation(number_of_cities) # https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html

# Prepare plot
plt.xlim(0,plot_range)
plt.ylim(0,plot_range)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_facecolor('xkcd:black')


for i in range(number_of_iterations):
  # Swap two random nodes (NOT following Ant system - just for illustration!)
  first_index = random.randint(0,len(path)-1)
  first_node = path[first_index]
  second_index = random.randint(0,len(path)-1)
  path[first_index] = path[second_index]
  path[second_index] = first_node
  
  # clear the plot, then plot nodes and connections
  plt.cla()
  plot_cities(plt,city_locations)
  plot_path(plt,path)
  plt.pause(pause_interval)

plt.show(block=False)
plt.ion()
input(f'Press return to exit')

 