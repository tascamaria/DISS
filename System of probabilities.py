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

def count_major_outbreaks(simulations, population, z):
    major_outbreaks = 0
    for j in range(len(simulations)):
        if simulations[j] > 0.3*population and simulations[j] > z*population:
            major_outbreaks +=1
    return major_outbreaks

from scipy.optimize import fsolve
def totalsize_stoch(z, R0):
    return z + np.exp(-R0*z) -1

def fraction(R0):
    return fsolve(totalsize_stoch, 1, args = (R0))[0]

def prob_totalsize(R0,gamma, num_simulations, population):
    z = fraction(R0)
    beta = R0 * gamma
    simulations = run_multiple_simulations_total_infections(num_simulations, beta, gamma, population, int(1))
    return count_major_outbreaks(simulations, population, z)/num_simulations

def alpha(s, R0):
    return N/(R0*s +N)

def beta(s, R0):
    return R0*(s+1)/(R0*s+N)

def H(s, x, R0):
    if(s == N+1):
        return x * N/ ((N+1)*R0*gamma_coeff)
    else:
        return beta(s, R0) * (x**2) / (x - alpha(s, R0)) * (H(s+1, x, R0) - V[s+1])
    


N = 19
#beta_coeff = 0.3
gamma_coeff = 1/(N-1)

R0 = 3

V = [0 for _ in range(N+2)]

for s in range(N+1, 0, -1):
    V[s] = H(s, alpha(s-1, R0), R0)
    

P = [gamma_coeff * beta(N-z, R0) / alpha(N-z, R0) * V[N-z+1] for z in range(N+1)]    
print(P)

def prob_totalsize_analytic(P,z):
    sum = 0
    i = 0
    while i <= z:
        sum = sum + P[i]
        i = i+1
    return 1 - sum

def prob_analytic_final(R0):
    
    V = [0 for _ in range(N+2)]

    for s in range(N+1, 0, -1):
          V[s] = H(s, alpha(s-1, R0), R0)

    P = [gamma_coeff * beta(N-z, R0) / alpha(N-z, R0) * V[N-z+1] for z in range(N+1)] 
    return prob_totalsize_analytic(P, int(N*fraction(R0)))


initial_guesses = np.linspace(0.001, 3, 50)
y = np.fmax(np.zeros(50),1 - 1/initial_guesses)
plt.plot(initial_guesses, [prob_totalsize(R0, gamma_coeff, 10000, N) for R0 in initial_guesses], color = 'b', label = 'Proportion of outbreaks with a bigger size than expected')
plt.plot(initial_guesses, [prob_analytic_final(R0) for R0 in initial_guesses], color = 'g', label = 'Using eq. (2.36)')
plt.plot(initial_guesses, y, color = 'c', label = 'Probability of a major outbreak estimation')
plt.legend()
plt.xlabel('R0')
plt.ylabel('Probability')
plt.ylim((-1,1))
plt.show()

