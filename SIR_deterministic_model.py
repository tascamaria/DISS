#import numpy as np
#import matplotlib.pyplot as plt

#def sir_model(beta, gamma, population, initial_infected, days):
    #susceptible = population - initial_infected
    #infected = initial_infected
    #recovered = 0

    #susceptible_list = [susceptible]
    #infected_list = [infected]
    #recovered_list = [recovered]

    #for day in range(days):
    #    new_infected = beta * susceptible * infected / population
    #    new_recovered = gamma * infected

    #    susceptible -= new_infected
    #    infected += new_infected - new_recovered
    #    recovered += new_recovered

    #    susceptible_list.append(susceptible)
    #    infected_list.append(infected)
    #    recovered_list.append(recovered)

    #return susceptible_list, infected_list, recovered_list

#def plot_sir(susceptible, infected, recovered, days):
    #plt.plot(range(days + 1), susceptible, label='Susceptible')
    #plt.plot(range(days + 1), infected, label='Infected')
    #plt.plot(range(days + 1), recovered, label='Recovered')
    #plt.xlabel('Days')
    #plt.ylabel('Population')
    #plt.title('SIR Model')
    #plt.legend()
    #plt.show()

# Parameters
#beta = 0.3   # infection rate
#gamma = 0.1  # recovery rate
#population = 1000
#initial_infected = 1
#days = 100

# Run SIR model
#susceptible, infected, recovered = sir_model(beta, gamma, population, initial_infected, days)

# Plot results
#plot_sir(susceptible, infected, recovered, days)

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 1000
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
beta, gamma = 0.08, 0.1 
# A grid of time points (in days)
t = np.linspace(0, 100, 100)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'c', alpha=0.5, lw=2, label='Recovered')
ax.set_xlabel('Time /days', fontsize = 30)
ax.set_ylabel('Population size', fontsize = 30)
ax.set_title('R0 = 0.8', fontsize = 30)
ax.yaxis.set_tick_params( labelsize = 30)
ax.xaxis.set_tick_params( labelsize = 30)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend(fontsize = 30)
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()