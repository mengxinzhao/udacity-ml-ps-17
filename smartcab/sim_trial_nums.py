#!/usr/bin/env python
import numpy as np
import random

def percent_visited(steps, visited,states,epsilon):
    for _ in range(steps):
        if np.random.binomial(1, epsilon) == 1:
            current_state = random.randint(0, states-1)
            visited[current_state] = True
    return sum(visited)/float(states)

def epsilon_decay(trial_times):
    if (trial_times <=100*4):
        return 1.0 / (1 + np.exp(0.09 * (trial_times-100)))
    else:
        return 0.5 * np.exp(-0.03 * (trial_times - 100))            

def trial_nums_calculation(percent):
    # 384 states * 4 action possible
    states = 384*4
    coverage = 0
    trial_runs_t = 0
    for i in range(0,10):
        visited = np.zeros(states, dtype=bool)
        output = False
        trial_runs = 0
        while True:
            # on this work deadline is chosen on distance * 5. distance is minimum 5 steps away
            # The grid size (8,6) from top left to bottom down 14 steps
            n_steps = random.choice([s*5 for s in range(5,14)]) # steps in one episodes
            epsilon = epsilon_decay(trial_runs)  
            #print(epsilon)
            if epsilon <=0.02:
                print("No hope of getting the state coverage for epsilon @",epsilon, "to cover {0:.2f}% states ".format(percent*100))
                break 
            coverage = percent_visited(n_steps,visited, states,epsilon)
            #if epsilon<=0.5 and output == False:
                #print("{0:.2f}% states covered in 0.5 epsilon runs".format(coverage*100),trial_runs/10)
                #output=True
            trial_runs += 1
            #print(coverage)
            if coverage >=percent:
                break

        trial_runs_t += trial_runs
        #print(coverage,i)
    print("trial numbers needed for {0:.2f}% states to be visited:".format(coverage*100),trial_runs_t/(i+1))
    return trial_runs


trial_nums_calculation(0.80)
trial_nums_calculation(0.85)
trial_nums_calculation(0.95)
trial_nums_calculation(1.0)
