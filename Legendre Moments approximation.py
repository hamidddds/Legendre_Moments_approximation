# This script demonstrates function approximation using Legendre moments and least-squares optimization.
# It is associated with the paper:
#   "Function Approximations Valid in Both Time and Frequency Domains Using Legendre Moments"
# The code is made public for anyone to use, modify, or contribute to, as requested by the journal.
# Author: hamidddds
#
# The script approximates a function by matching its Legendre moments using a least-squares approach.
#
# Main steps:
#   1. Define the target function.
#   2. Scale the domain to [-1, 1] for Legendre polynomials.
#   3. Compute continuous Legendre moments of the target function.
#   4. Set up a least-squares system to match moments with a discrete approximation.
#   5. Solve for the best-fit discrete function values.
#   6. Plot and compare the original and approximated functions and their moments.

import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy.integrate import quad
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.special import legendre

# Define the function to approximate

def real_function(x):
    # Target function: a combination of sine and cosine
    return 5 * np.sin(4 * x) + 4 * np.cos(3 * x)

# Scale domain to [-1, 1]
def scale_domain(x, x_start, x_end):
    # Maps x from [x_start, x_end] to [-1, 1]
    return 2 * (x - x_start) / (x_end - x_start) - 1

# Unscale domain from [-1, 1] back to [x_start, x_end]
def unscale_domain(x_scaled, x_start, x_end):
    # Maps x from [-1, 1] back to [x_start, x_end]
    return (x_scaled + 1) * (x_end - x_start) / 2 + x_start

# Calculate continuous Legendre moments with normalization
def calculate_legendre_moment(func, n, x_start, x_end):
    # Computes the n-th Legendre moment of a function over [x_start, x_end]
    def integrand(x_scaled):
        x_unscaled = unscale_domain(x_scaled, x_start, x_end)
        Pn = legendre(n)
        return func(x_unscaled) * Pn(x_scaled)
    moment, _ = quad(integrand, -1, 1)
    return (2 * n + 1) / 2 * moment

# Least-squares system for moment-matching approximation
def equations(yy, x, x_start, x_end, target_moments, regularization_param):
    # Residuals: difference between target and approximated moments, plus regularization
    num_points = len(x)
    residuals = []
    for n in range(num_points):
        moment_approx = 0
        for i in range(num_points - 1):
            x_scaled_i = scale_domain(x[i], x_start, x_end)
            x_scaled_ip1 = scale_domain(x[i + 1], x_start, x_end)
            Pn = legendre(n)

            def integrand(x_scaled):
                x_unscaled = unscale_domain(x_scaled, x_start, x_end)
                # Linear interpolation between points
                return np.interp(x_unscaled, x, yy) * Pn(x_scaled)
            moment_approx += quad(integrand, x_scaled_i, x_scaled_ip1)[0]
        moment_approx *= (2 * n + 1) / 2  # Apply normalization here
        residuals.append(moment_approx - target_moments[n])
    # Optional: Regularization to stabilize solution
    regularization = regularization_param * np.linalg.norm(yy)
    residuals.append(regularization)
    return residuals

# Parameters
x_start = 0
x_end = 2*np.pi
number_of_points = 20
regularization_param = 1e-1

# Discrete sample points for approximation
x = np.linspace(x_start, x_end, number_of_points)
x_continuous = np.linspace(x_start, x_end, 500)

# Compute continuous Legendre moments as targets
target_moments = [
    calculate_legendre_moment(real_function, n, x_start, x_end)
    for n in range(number_of_points)
]

# Initial guess for the function at the sample points
yy_initial = real_function(x)

# Solve the least-squares system
result = least_squares(
    equations,
    yy_initial,
    args=(x, x_start, x_end, target_moments, regularization_param)
)
yy_solution = result.x

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_continuous, real_function(x_continuous),
         label="Original Function", color="red")
plt.plot(x, yy_solution, 'o-', label="Moment-Matching Approximation",
         linestyle="--", color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Legendre Moment-Based Function Approximation")
plt.grid(True)
plt.tight_layout()
plt.show()

# Compare moments
print("Moment Comparison (Continuous vs. Approximation):")
for n in range(number_of_points):
    moment_approx = 0
    for i in range(number_of_points - 1):
        x_scaled_i = scale_domain(x[i], x_start, x_end)
        x_scaled_ip1 = scale_domain(x[i + 1], x_start, x_end)
        Pn = legendre(n)

        def integrand(x_scaled):
            x_unscaled = unscale_domain(x_scaled, x_start, x_end)
            return np.interp(x_unscaled, x, yy_solution) * Pn(x_scaled)
        moment_approx += quad(integrand, x_scaled_i, x_scaled_ip1)[0]
    moment_approx *= (2 * n + 1) / 2  # Apply normalization
    print(
        f"Moment {n:2d}: Target = {target_moments[n]: .6f}, Approx = {moment_approx: .6f}, Diff = {moment_approx - target_moments[n]: .2e}")
