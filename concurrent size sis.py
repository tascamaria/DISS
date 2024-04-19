import math
import numpy as np
import matplotlib.pyplot as plt

def prod(N,j):
    p = 1
    for i in range(1, j + 1, 1):
        p = p * (N-i)
    p = 1/p
    return p

def calculate_expression(N, R0, I, M):
    if I < M:
        numerator = 1 + sum((N**j / R0**j) * (prod(N,j)) for j in range(1, I))
        denominator = 1 + sum((N**j / R0**j) * (prod(N, j)) for j in range(1, M))

        result = numerator / denominator
        return result
    else:
        return 1 

# Example usage:
N_value = 140
#R0_value = 3
I_value = 1
#M_value = 4

valuesforM = np.arange(N_value)
colors = ['r', 'y', 'm', 'c', 'b','k']
valuesforR0 = [0.8,1.2,1.5,1.8, 2, 2.5]
j=0
for R0_value in valuesforR0:
    result = [calculate_expression(N_value, R0_value, 1, M_value) for M_value in valuesforM]
    point = max(0,N_value - N_value/R0_value)
    plt.plot(valuesforM, result,  label = 'R0 = {}'.format(R0_value), color = colors[j])
    plt.scatter(point, calculate_expression(N_value, R0_value, 1, int(point)), color=colors[j], marker ='D')
    plt.axhline(y = max(0,1 - 1/R0_value), linestyle = 'dashed', color = colors[j])
    j = j+1
plt.tick_params(fontsize = 30)
plt.xlabel('M', fontsize = 30)
plt.ylabel('Probability', fontsize = 30)
plt.legend(fontsize = 30)
plt.show()

valuesforI0 = [1, 2, 5, 20]
colors = ['r', 'y', 'm', 'c']
R0_value = 1.2
j = 0
point = max(0,N_value - N_value/R0_value)
valuesforM = np.arange(N_value)
for I_value in valuesforI0:
    result = [calculate_expression(N_value, R0_value, I_value, M_value) for M_value in valuesforM]
    plt.plot(valuesforM, result, label = 'I(0) = {}'.format(I_value), color = colors[j])
    plt.scatter(int(point), calculate_expression(N_value, R0_value, I_value, int(point)), color=colors[j], marker ='D')
    plt.axhline(y = 1 - (1/R0_value)**(I_value), linestyle = 'dashed', color = colors[j])
    j = j+1

plt.xlabel('M', fontsize  = 30)
plt.ylabel('Probability', fontsize = 30)
plt.legend(fontsize =30)
plt.show()