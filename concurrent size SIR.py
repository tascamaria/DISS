import numpy as np
import matplotlib.pyplot as plt
def calculate_probabilities(N, M, R0):
    # Initialize the probability matrix
    P = np.zeros((M+1, N-M+2))

    # Set boundary conditions
    P[0, :] = 0
    P[:, N-(M-1)] = 0
    P[M, :] = 1
    

    # Calculate probabilities using the recurrence relation
    for R in range(N-M, -1, -1):
        for I in range(M-1, 0, -1):
            numerator_1 =  R0* I * (N-I-R)
            denominator =  R0* I * (N-I-R) +  I * N
            term_1 = P[I+1, R] * (numerator_1 / denominator) 
            
            numerator_2 = I * N
            
            term_2 = P[I-1, R+1] * (numerator_2 / denominator) 

            P[I, R] = term_1 + term_2
    return P

# Example usage:
N_value = 100

valuesforR0 = [0.8,1.1,1.5,2]
valuesforM = np.arange(25, N_value)
colors = ['r', 'y', 'm', 'b']
j = 0
for R0 in valuesforR0:
    result = np.zeros(99)
    i = 0 
    for M_value in valuesforM:
        result[i] = calculate_probabilities(N_value, M_value, R0)[1][0]
        i= i+1
    plt.plot(valuesforM, result, color = colors[j], label = 'R0 ={}'.format(R0) )
    plt.axhline(y = max(0, 1 - 1/R0), linestyle = 'dashed', color = colors[j])
    j = j+1


plt.tick_params(labelsize = 20)
plt.title('The probability of a severe epidemic', fontsize = 20)
plt.xlabel('M', fontsize = 20)
plt.ylabel('Probability P(1,M)', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()

R0 = 2
initial = [5, 10, 25]
j = 0
for k in initial:
    result = np.zeros(99)
    i = 0 
    for M_value in valuesforM:
        result[i] = calculate_probabilities(N_value, M_value, R0)[k][0]
        i= i+1
    plt.plot(valuesforM, result, color = colors[j], label = 'I(0) ={}'.format(k) )
    plt.axhline(y = max(0, 1 - 1/R0), linestyle = 'dashed', color = colors[j])
    j = j+1

plt.tick_params(labelsize = 20)
plt.title('The probability of a severe epidemic for R0 = 2', fontsize = 20)
plt.xlabel('M', fontsize = 20)
plt.ylabel('Probability P(M,0)', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()