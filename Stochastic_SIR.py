import numpy as np
import matplotlib.pyplot as plt
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

def plot_multiple_simulations_infections_only(simulations):
    plt.figure(figsize=(10, 6))
    plt.tick_params(labelsize = 30)
    plt.hist(simulations, bins = 100, color='b', edgecolor='black')
    plt.xlabel('Total number of infections', fontsize = 30)
    plt.ylabel('Frequency', fontsize = 30)
    plt.axvline( x = 30, label = 'Diving line between a major and a minor outbreak', linestyle = '-', color = 'r' )
    plt.legend(fontsize = 25)
    plt.xlim([0,100])
    plt.show()


# Parameters
beta = 0.11   # infection rate
gamma = 0.1  # recovery rate
population = 100
initial_infected = 1
num_simulations = 10000

# Run 100 simulations
simulations = run_multiple_simulations_total_infections(num_simulations, beta, gamma, population, initial_infected)

# Plot results
plot_multiple_simulations_infections_only(simulations)

# Counting outbreaks
def count_major_outbreaks(simulations, population, z):
    major_outbreaks = 0
    for j in range(len(simulations)):
        if simulations[j] >= population*z:
            major_outbreaks +=1
    return major_outbreaks

from scipy.optimize import fsolve
def totalsize_stoch(z, R0):
    return z + np.exp(-R0*z) -1


def prob_totalsize(R0,gamma, num_simulations, population):
    z = fsolve(totalsize_stoch, 1, args = (R0))[0]
    beta = R0 * gamma
    simulations = run_multiple_simulations_total_infections(num_simulations, beta, gamma, population, int(1))
    return count_major_outbreaks(simulations, population, z)/num_simulations

def fraction(R0):
    return fsolve(totalsize_stoch, 1, args = (R0))[0]

initial_guesses = np.linspace(1, 4, 100)
prob_size=[]
for R0 in initial_guesses:
    prob_size.append(prob_totalsize(R0, num_simulations, population))

plt.figure(figsize=(10, 6))
plt.plot(initial_guesses, prob_size)
plt.xlabel('R0')
plt.ylabel('Probability')
plt.title('Probability of outbreaks having more infected individuals than expected')
plt.legend()
plt.show()

def prob(R0,z):
    fact = 1
    i = 2
    while i < (z+1):
        fact = fact * (z+i)
        fact = fact / i
        i = i + 1
    return ((R0**z)*fact)/(((1+R0)**(2*z+1)))

def prob_total(R0, Z):
    sum = 0
    for i in range(int(Z)):
        sum = sum + prob(R0, i)
    return 1 - sum

def simulations_totalsize(num_simulations, beta, gamma, population):
    simulations = run_multiple_simulations_total_infections(num_simulations, beta, gamma, population, 1)
    R0 = beta/gamma
    z = fraction(R0)
    p = prob_total(R0, int(population*z))
    plt.figure(figsize=(10, 6))
    plt.hist(simulations,bins =100, color='b', edgecolor='black')
    plt.xlabel('Total number of infections')
    plt.ylabel('Frequency')
    plt.title('Stochastic SIR Model - 10000 Simulations - Comparing definitions')
    plt.text(population * z, 2000, 'Dividing line between a minor and a major outbreak', ha='center', va='center',rotation='vertical', backgroundcolor='white')
    plt.axvline( x = population * z )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(200, 4000, 'Prob(major outbreak) = {prob} using the total size metric '.format(prob = p), verticalalignment='top', bbox=props)
    plt.text(200, 3500, 'Prob(major outbreak) = {} using the standard definition'.format(1-1/R0), verticalalignment='bottom', bbox=props)
    plt.legend()
    plt.show()


plt.figure(figsize=(10,6))
# Fix gamma, the population size, the initial infected population to be 1, number of simulations to be 10000
gamma = 0.1
population = 500
num_simulations = 10000
x = np.linspace(0,4, 100)
plt.plot(x, 1- 1/x, 'r', label = 'The standard probability' )
plt.plot(x, prob_total(x, int(100 * fraction(x))), 'b', label = 'The analytic total size probability')
plt.plot(x, prob_total(x, int(100 * prob_totalsize(x, gamma, num_simulations, population))), 'y', label = 'The computational total size probability')
plt.xlabel('The basic reproduction number')
plt.ylabel('Probability')
plt.legend()
plt.show()
