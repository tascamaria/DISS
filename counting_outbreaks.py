import numpy as np
import matplotlib.pyplot as plt


def stochastic_sir(beta, gamma, population, initial_infected, end_time):
    susceptible = population - initial_infected
    infected = initial_infected
    recovered = 0

    events = []

    current_time = 0

    while current_time < end_time:
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
            events.append(('I', current_time + time_to_event))
        else:
            # Recovery event
            infected -= 1
            recovered += 1
            events.append(('R', current_time + time_to_event))

        # Update the current time
        current_time += time_to_event

    return events

def run_multiple_simulations(num_simulations, beta, gamma, population, initial_infected, end_time):
    all_simulations = []

    for j in range(num_simulations):
        events = stochastic_sir(beta, gamma, population, initial_infected, end_time)
        all_simulations.append(events)
        j = j+1
    
    return all_simulations

def identify_outbreaks(simulations, population, end_time, outbreak_threshold=0.01):
    outbreak_periods = []

    for i, events in enumerate(simulations):
        time_series = np.zeros(int(end_time) + 1)

        for event, time in events:
            if event == 'I':
                time_series[int(time)] += 1
            else:
                time_series[int(time)] -= 1

        daily_new_infections = np.diff(time_series)
        
        # Identify potential outbreaks based on a threshold
        outbreak_indices = np.where(daily_new_infections > outbreak_threshold * population)[0]
        
        outbreak_periods.append((i + 1, outbreak_indices))

    return outbreak_periods

def plot_outbreaks(simulations, population, end_time, outbreak_threshold=0.01):
    outbreak_periods = identify_outbreaks(simulations, population, end_time, outbreak_threshold)

    plt.figure(figsize=(10, 6))

    for i, events in enumerate(simulations):
        time_series = np.zeros(int(end_time) + 1)

        for event, time in events:
            if event == 'I':
                time_series[int(time)] += 1
            else:
                time_series[int(time)] -= 1

        cumulative_time_series = np.cumsum(time_series)

        plt.plot(np.arange(int(end_time) + 1), population - cumulative_time_series, label=f'Simulation {i + 1}')

        # Highlight outbreak periods
        outbreak_indices = outbreak_periods[i][1]
        plt.scatter(outbreak_indices, population - cumulative_time_series[outbreak_indices],
                    marker='o', color='red', label='Outbreak')

    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Stochastic SIR Model - 100 Simulations (Outbreaks > {outbreak_threshold * 100}% of Population)')
    plt.legend()
    plt.show()

# ... (previous code remains unchanged)

# Parameters
beta = 0.3   # infection rate
gamma = 0.1  # recovery rate
population = 1000
initial_infected = 1
end_time = 200
num_simulations = 100

# Run 100 simulations
simulations = run_multiple_simulations(num_simulations, beta, gamma, population, initial_infected, end_time)

# Plot results with highlighted outbreaks
plot_outbreaks(simulations, population, end_time, outbreak_threshold=0.02)

def count_major_outbreaks(simulations, population, end_time, outbreak_threshold=0.01):
    major_outbreak_count = 0

    for i, events in enumerate(simulations):
        time_series = np.zeros(int(end_time) + 1)

        for event, time in events:
            if event == 'I':
                time_series[int(time)] += 1
            else:
                time_series[int(time)] -= 1

        daily_new_infections = np.diff(time_series)
        
        # Identify potential outbreaks based on a threshold
        outbreak_indices = np.where(daily_new_infections > outbreak_threshold * population)[0]
        
        if len(outbreak_indices) > 0:
            major_outbreak_count += 1

    return major_outbreak_count