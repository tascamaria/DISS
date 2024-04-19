import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def totalsize_det(z, R0, epsilon):
    return z + (1 - epsilon) * np.exp(-R0 * z) - 1

def totalsize_stoch(z, R0):
    return z + np.exp(-R0*z) -1

initial_guesses = np.linspace(1, 4, 50)
epsilon = 1/1000

solutions_det = [fsolve(totalsize_det, 1, args=(R0, epsilon))[0] for R0 in initial_guesses]
solutions_stoch = [fsolve(totalsize_stoch, 1, args=(R0))[0] for R0 in initial_guesses]

import math

def stochastic_sir_total_infections(beta, gamma, population, initial_infected):
    susceptible = population - initial_infected
    infected = initial_infected
    recovered = 0
    total_infections = initial_infected
    #events = []

    current_time = 0

    while infected > 0:
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
            recovered += 1
            #events.append(('R', current_time + time_to_event))

        # Update the current time
        current_time += time_to_event

    return total_infections

def run_multiple_simulations_total_infections(num_simulations, beta, gamma, population, initial_infected):
    all_simulations = []
    for j in range(num_simulations):
        all_simulations.append(stochastic_sir_total_infections(beta, gamma, population, initial_infected))
        j = j+1
    return all_simulations

def count_infection(simulations, population, z):
    infections = 0
    major_outbreaks = 0
    for j in range(len(simulations)):
        if simulations[j] >= population*z:
            major_outbreaks +=1
            infections = infections + simulations[j]
    return infections/major_outbreaks

population = 1000
initial_infected = 1
num_simulations = 10000

number = np.zeros(50)
k = 0 
for R0 in initial_guesses:
    simulations = run_multiple_simulations_total_infections(num_simulations, 0.1*R0, 0.1, population, initial_infected)
    number[k] = count_infection(simulations, population, 0.3)
    k = k+1

plt.figure(figsize=(9, 3))
plt.tick_params(labelsize = 20)
plt.plot(initial_guesses, number, color = 'y', label  = 'Average number of infections based on 10000 stochastic runs')
plt.plot(initial_guesses, 1000*np.array(solutions_stoch), color  ='g', label = 'Average number of infections based on equation 2.21')
plt.plot(initial_guesses, 1000*np.array(solutions_det), color = 'b', label  = 'Average number of infections based on equation 2.18')
plt.xlabel('R0', fontsize = 20)
plt.ylabel('Total number of infections', fontsize = 20)
plt.title('Final size of the epidemic', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()
plt.figure()

def count_infection_small(simulations):
    infections = 0
    for j in range(len(simulations)):
        infections = infections + simulations[j]
    return infections/len(simulations)

initial_guesses = np.linspace(0, 1, 100)[1:]
epsilon = 1/1000

solutions_det = [fsolve(totalsize_det, 1, args=(R0, epsilon))[0] for R0 in initial_guesses]
solutions_stoch = [fsolve(totalsize_stoch, 1, args=(R0))[0] for R0 in initial_guesses]

number = np.zeros(99)
k = 0 
for R0 in initial_guesses:
    simulations = run_multiple_simulations_total_infections(num_simulations, 0.1*R0, 0.1, population, initial_infected)
    number[k] = count_infection_small(simulations)
    k = k+1

plt.figure(figsize=(9, 3))
plt.tick_params(labelsize = 20)
plt.plot(initial_guesses, number, color = 'y', label  = 'Average number of infections based on 10000 stochastic runs')
plt.plot(initial_guesses, 1000*np.array(solutions_stoch), color  ='g', label = 'Average number of infections based on equation 2.21')
plt.plot(initial_guesses, 1000*np.array(solutions_det), color = 'b', label  = 'Average number of infections based on equation 2.18')
plt.xlabel('R0', fontsize = 20)
plt.ylabel('Total number of infections', fontsize = 20)
plt.title('Final size of the epidemic', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()
    
plt.figure()