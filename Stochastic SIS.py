import numpy as np
import matplotlib.pyplot as plt
import math

def stochastic_sis_total_infections(beta, gamma, population, initial_infected,maxtime):
    susceptible = population - initial_infected
    infected = initial_infected
    total_infections = initial_infected
    #events = []

    current_time = 0

    while maxtime > current_time and infected>0:
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
    for _ in range(num_simulations):
        y = stochastic_sis_total_infections(beta, gamma, population, initial_infected, maxtime)
        all_simulations.append(y)
        print(y)
    
    return all_simulations

def total_size(N, R0):
    if R0 >= 1:
        return math.sqrt(2*math.pi*N)*(1/R0)*math.exp(N*(math.log(R0) + 1/R0 - 1))
    else:
        return 1/(1-R0)

values_R0  = np.linspace(0.01,0.99,20)
N = 100
totalsize_anal=[]
totalsize_exp = []

for R0 in values_R0:
    z = total_size(N, R0)
    totalsize_anal.append(z)
    print(z)
    maxtime = 2 * z - 1
    Z = run_multiple_simulations_total_infections_sis(10000, R0 *0.1, 0.1, N, 1, maxtime)
    totalsize_exp.append(Z)
    print(Z)
    

plt.figure(figsize=(10, 6))
plt.plot(values_R0, totalsize_anal, color = 'r', label = 'Analytic solution')
plt.plot(values_R0, totalsize_exp, color = 'b', label = 'Experimental findings')
plt.xlabel('R0', fontsize = 30)
plt.ylabel('Total size', fontsize = 30)
plt.legend(fontsize = 30)
plt.tick_params(labelsize = 30)
plt.show()

rap = []
for j in range(len(values_R0)):
    rap.append(totalsize_anal[j]/totalsize_exp[j])

plt.figure(figsize=(10, 6))
plt.plot(values_R0, rap)
plt.xlabel('R0', fontsize = 30)
plt.ylabel('Ratio', fontsize = 30)
plt.ylim((1,5))
plt.tick_params(labelsize = 30)
plt.show()


def plot_multiple_simulations_infections_only_sis(simulations):
    plt.figure(figsize=(10, 6))
    plt.hist(simulations,bins =100, color='b', edgecolor='black')
    plt.xlabel('Total number of infections')
    plt.ylabel('Frequency')
    plt.title('Stochastic SIS Model - 10000 Simulations, First 100 days')
    #plt.text(100, 2000, 'Dividing line between a minor and a major outbreak', ha='center', va='center',rotation='vertical', backgroundcolor='white')
    #plt.axvline( x = 100 )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(4000, 3000, 'Prob(major outbreak) = 0,66', verticalalignment='top', bbox=props)
    plt.legend()
    plt.show()


# Counting outbreaks
def count_major_outbreaks(simulations, population, z):
    major_outbreaks = 0
    for j in range(len(simulations)):
        if simulations[j] >= population*z:
            major_outbreaks +=1
    return major_outbreaks

import collections
def prob_totalsize(N, R0):
    maxtime = 2*total_size(N, R0) - 1
    simulations = run_multiple_simulations_total_infections_sis(10000, R0 *0.1, 0.1, N,1, maxtime)
    counter = collections.Counter(simulations)
    numbers = np.zeros(max(simulations)+1)
    for j in range(max(simulations)+1):
        if counter[j]>0:
            numbers[j] = counter[j]
    
    return [y/10000 for y in numbers]

def prob_severe_epidemic(p):
    severe = np.zeros(len(p))
    severe[0] = 1- p[0]
    for i in range(1, len(p)):
        severe[i] = severe[i-1] - p[i]
    return severe

N= 100
R0 = 1.5
y = prob_totalsize(N, R0)
m = prob_severe_epidemic(y)
point = total_size(N, R0)
values_M = np.arange(120)
plt.figure(figsize=(10,6))
plt.plot(values_M, m[0:len(values_M)], color ='r', label ='Computational findings')
plt.axhline(y = 1 - 1/R0, color = 'b', linestyle = '--', label = 'Branching process estimate')
#plt.scatter(math.ceil(point), m[math.ceil(point)], marker ='D')
plt.legend(fontsize = 30)
plt.xlabel('M', fontsize = 30)
plt.ylabel('Probabilities', fontsize  = 30)
plt.tick_params(labelsize = 30)
plt.title('R0 = 1.5', fontsize = 30)
plt.show()
