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

figure, axis = plt.subplots(1,2)
axis[0].tick_params(labelsize=20)
axis[0].plot(initial_guesses, solutions_det)
axis[0].set_title('Using eq. (2.18) - deterministic model', fontsize = 18)
axis[0].set_xlabel('R0', fontsize = 20)
axis[0].set_ylabel('Fraction of the infected individuals out of the total population', fontsize = 18)
axis[1].tick_params(labelsize=20)
axis[1].plot(initial_guesses, solutions_stoch)
axis[1].set_title('Using eq. (2.21) -stochastic model', fontsize = 18)
axis[1].set_xlabel('R0', fontsize = 20)
plt.legend(fontsize = 20)
plt.show()



    
