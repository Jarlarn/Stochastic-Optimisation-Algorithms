import math
import numpy as np # optional - uncomment and use if you wish

# ==============================
# run_gradient_descent function:
# ==============================

# To do: Write this function (uncomment the line below).
def run_gradient_descent(x_start, mu, eta, gradient_tolerance):
  is_norm_above_threshold = True
  current_x = np.array(x_start, dtype=float)  

  while is_norm_above_threshold:
    gradient = np.array(compute_gradient(current_x,mu))
    gradient_norm = np.linalg.norm(gradient)
    is_norm_above_threshold = gradient_norm > gradient_tolerance
    current_x = current_x - eta * gradient

  return current_x


# ==============================
# compute_gradient function:
# ==============================

# To do: Write this function (uncomment the line below).
def compute_gradient(x, mu):
    """
    Hardcoded analytical gradient for:
      f(x) = (x1 - 1)^2 + 2*(x2 - 2)^2
      g(x) = x1^2 + x2^2 - 1
      f_p(x; mu) = f(x) + mu*max(0,g(x))^2
    """
    x1, x2 = x

    df_dx1 = 2 * (x1 - 1)
    df_dx2 = 4 * (x2 - 2)

    g = x1**2 + x2**2 - 1

    if g > 0:
        df_dx1 += 4*mu * g * x1
        df_dx2 += 4*mu * g * x2

    return [df_dx1, df_dx2]


# ==============================
# Main program:
# ==============================
mu_values = [1, 10, 100, 1000]
eta = 0.0001
x_start = [1,2]
gradient_tolerance = 0.0000001

for mu in mu_values:
  x = run_gradient_descent(x_start, mu, eta, gradient_tolerance)
  output = f"x = ({x[0]:.4f}, {x[1]:.4f}), mu = {mu:.1f}"
  print(output)


