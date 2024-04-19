import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x, m):
    return np.maximum(0, 1 - (1 / x)**m)

# Generate x values
x_values = np.linspace(0.01, 4, 400)  # Adjust the range and number of points as needed

# Define values of m to plot
m_values = [1, 2, 5, 10, 20]

# Plot the function for each value of m
for m in m_values:
    y_values = f(x_values, m)
    plt.plot(x_values, y_values, label=f'm={m}', linewidth = 5)

# Add labels and legend
plt.xlabel('R0', fontsize = 30)
plt.ylabel('Probability of a major outbreak', fontsize = 30)
plt.tick_params(labelsize = 30)
plt.legend(fontsize = 30)
plt.grid(True)
plt.ylim(0, 1.2)
plt.xlim(0,4)  # Adjust the y-axis limits if needed
plt.show()
