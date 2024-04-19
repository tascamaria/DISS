import numpy as np
import matplotlib.pyplot as plt
import math

def stochastic_sis_total_infections(beta, gamma, population, initial_infected,maxtime):
    susceptible = population - initial_infected
    infected = initial_infected
    total_infections = initial_infected
    #events = []

    current_time = 0

    while current_time<maxtime:
        infection_rate = beta * susceptible * infected / population
        recovery_rate = gamma * infected

        total_rate = infection_rate + recovery_rate

        if total_rate == 0:
            break  # No events can occur

        # Calculate time until the next event
        time_to_event = np.random.exponential(1 / total_rate)

        # Determine which event occurs
        if np.random.rand() < infection_rate / total_rate:
            # Infection event
            susceptible -= 1
            infected += 1
            total_infections +=1
            #events.append(('I', current_time + time_to_event))
        else:
            # Recovery event
            infected -= 1
            susceptible += 1
            #events.append(('R', current_time + time_to_event))

        # Update the current time
        current_time += time_to_event

    return total_infections

def run_multiple_simulations_total_infections_sis(num_simulations, beta, gamma, population, initial_infected, maxtime):
    all_simulations = []
    for j in range(num_simulations):
        all_simulations.append(stochastic_sis_total_infections(beta, gamma, population, initial_infected,maxtime))
        j = j+1
    return all_simulations

def alpha(s, R0):
    return N/(R0*s +N)

def beta(s, R0):
    return R0*(s+1)/(R0*s+N)

def delta(i):
    if i == 1:
        return 1
    else:
        return 0

def h(s, i, R0):
    if i == 0:
        return 0
    else:
        if(s == N):
            return alpha(N, R0)*h(N-1, i+1, R0) + beta(N, R0)* delta(i)*N/(R0*gamma_coeff*(N+1))
        else:
            return alpha(s, R0)*h(s-1, i+1, R0) + beta(s, R0) * h(s+1, i-1, R0)
    
N = 10
gamma_coeff = 0.1

def prob(R0):
    
    V = [0 for _ in range(N+2)]

    for s in range(N+1, 0, -1):
          V[s] = h(s, 1, R0)

    P = [gamma_coeff * V[N-z+1] for z in range(N+1)] 
    return P

def compute_h(N, alpha, beta, delta, i_max):
    h = [[0 for _ in range(i_max)] for _ in range(N)]
    print(h)
    # Base case for s = N
    for i in range(0, i_max):
        h[N - 1][i] = alpha[N - 1] * h[N - 2][i + 1] + beta[N - 1] * (delta[1][i] / (beta[N - 1] * ((N + 1) / N)))
        print(h)
    
    # Recursive case for s from 1 to N-1
    for s in range(N - 1, 0, -1):
        for i in range(0, i_max):
            h[s][i] = alpha[s] * h[s - 1][i + 1] + beta[s] * h[s + 1][i - 1]
            print(h)
    return h

# Example usage:
N = 5
alpha = [1, 2, 3, 4, 5]  # example alpha values
beta = [2, 3, 4, 5, 6]    # example beta values
delta = [[0 for _ in range(10)] for _ in range(10)]  # example delta values, assuming 10 columns for h
i_max = 10  # maximum value of i
result = compute_h(N, alpha, beta, delta, i_max)

# Printing the result for illustration
for s in range(N):
    for i in range(1, i_max):
        print(f"h_{s + 1},{i} = {result[s][i]}")
