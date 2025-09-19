import math
# ToDo: Uncomment the line below ("import ...") once you have implemented plot_iterations.
# You may need to install Matplotlib first: pip install matplotlib
# import matplotlib.pyplot as plt 

# ============================================
# get_polynomial_value:
# ============================================

# ToDo: Write the get_polynomial_value function

# Uncomment the line below and implement the function:
def get_polynomial_value(x, polynomial_coefficients):  
    result = 0
    for index, polynomial_coefficient in enumerate(polynomial_coefficients):
        result += polynomial_coefficient * (x**index)
    return result


# ============================================
# differentiate_polynomial:
# ============================================

# ToDo: Write the differentiate_polynomial function

# Uncomment the line below and implement the function:
def differentiate_polynomial(polynomial_coefficients, derivative_order):
    output_coefficents = []
    for index, polynomial_coefficient in enumerate(polynomial_coefficients):
        if index < derivative_order:
            continue
        output_coefficent=index * polynomial_coefficient
        output_coefficents.append(output_coefficent)
    return output_coefficents

        

    

# ============================================
# step_newton_raphson:
# ============================================

# ToDo: Write the step_newton_raphson function

# Uncomment the line below and implement the function:
# def step_newton_raphson(x, f_prime, f_double_prime):

# ============================================
# run_newton-raphson:
# ============================================
     
# ToDo: Write the run_newton-raphson function

# Uncomment the line below and implement the function:
# def run_newton_raphson(polynomial_coefficients, starting_point, tolerance, maximum_number_of_iterations):

# ============================================
# plot_iterations:
# ============================================

# ToDo: Write the plot_iterations function
#
# Here, you should use matplotlib. 
# Note: You must uncomment the second "import ..." statement above
# Then uncomment the line below and implement the function:
# def plot_iterations(polynomial_coefficients,iterations):

# ============================================
# Main loop
# ============================================

tolerance = 0.00001
maximum_number_of_iterations = 10

polynomial_coefficients = [10,-2,-1,1]
print(differentiate_polynomial(polynomial_coefficients,1))
starting_point = 2

# ToDo: Uncomment the two lines below, once you have implemented the functions above:
# iterations = run_newton_raphson(polynomial_coefficients, starting_point, tolerance, maximum_number_of_iterations)
# plot_iterations(polynomial_coefficients, iterations)