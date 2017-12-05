import random
import math
import numpy as np
from collections import OrderedDict
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5, epsilon_decay = 'linear'):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        #self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.Q = OrderedDict()
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        # track the current state/action and update them once an new action
        # is taken and reward received and state changed
        # use for one step learning back-up
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.action_index_map = dict(zip(['None', 'forward', 'left', 'right'], [0,1,2,3]))
        self.light_index_map = dict({'green': 0, 'red': 1})
        # use for n-step learning back up
        # for the test cases, the trajectory isn't super long so I keep all the steps
        # otherwise sliding window should be used to ensure only n steps are stored to save space
        self.n = 2 #  according to Sutton "methods with an intermediate value of n works the best
        self.state_action_trajectory = []
        self.rewards_history = []
        self.t = 0 # time steps
        self.trial_times = 0

        self.epsilon_decay = epsilon_decay
        if self.epsilon_decay == 'linear' :
            self.epsilon = 1.0
        else:
            print("epsilon",epsilon)
            self.epsilon_ceiling = epsilon
            self.epsilon_a = 0.03
            self.epsilon = epsilon* np.exp(-self.epsilon_a* self.trial_times)

        ## init tabular Q table
        ## only 512 states
        for  i in range(0,512+1):
            self.Q.update({i:{}})
            for action in self.valid_actions:
                self.Q[i][action] = 0.0



    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if self.epsilon_decay == 'linear':
            self.epsilon -= 0.02
        else:
            self.epsilon =  self.epsilon_ceiling*np.exp(-self.epsilon_a * self.trial_times)
        if self.epsilon < 0:
            self.epsilon = 0

        if testing == True:
            self.epsilon = 0
            self.alpha = 0

        self.last_state = None
        self.last_action = None
        self.state_action_trajectory.clear()
        self.rewards_history.clear()
        self.t = 0
        self.trial_times +=1
        return None

    # waypoint has 4 possible actions [None, 'forward', 'left', 'right']
    # Inputs = {'light': 'green' or 'red', 'oncoming': 4 directions, 'right': 4 directions, 'left': 4 directions}
    # build a map from waypoint,inputs to Q.table index
    def build_index (self, inputs,waypoint = None):

        state_index = 0
        if waypoint:
            state_index = (self.action_index_map[waypoint]) << 2

        state_index += self.light_index_map[inputs['light']]
        state_index = state_index << 1

        state_index += self.action_index_map[inputs['oncoming']]
        state_index = state_index << 2

        state_index += self.action_index_map[inputs['left']]
        state_index = state_index << 2

        state_index += self.action_index_map[inputs['right']]

        print("state: {} {}".format(waypoint,inputs))

        return state_index

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        for key, value in iter(inputs.items()):
            if value is None:
                inputs.update({key:'None'})
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        
        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.
        
        # Set 'state' as a tuple of relevant data for the agent        
        return self.build_index(inputs,waypoint)


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        maxQ = 0.0
        if state in self.Q:
            print("{}  {}".format(self.Q[self.state].items(),self.Q[self.state].values()))
            maxQ = max(self.Q[self.state].values())
            print("maxQ  {:.4f} for {} ".format(maxQ,state))
        return maxQ

    def get_expectedMaxQ(self,state):
        # Sum(π(a|St+1)Q(St+1, a))
        # assume there is only one max. In case that there are tied maxes, choose the last max
        expected_value = 0.0
        optimal_actions = []
        optimal_action = None
        max_q = -np.inf
        if state in self.Q:
            max_q = self.get_maxQ(state)
            for action, value in iter(self.Q[state].items()):
                if value == max_q:
                    optimal_actions.append(action)
            print("{} optimal action: {}".format(len(optimal_actions),optimal_actions))
            for action in iter(self.Q[state].keys()):
                if action in optimal_actions:
                    expected_value += (1/len(optimal_actions) + self.epsilon/4) * self.Q[self.state][action]
                else:
                    expected_value += (self.epsilon / 4) * self.Q[self.state][action]

        return expected_value



    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0
        if not state in self.Q:
            print("inserting new state {}",state)
            self.Q.update({state:{}})
            for action in self.valid_actions:
                self.Q[state][action]=0.0
        return

    def find_optimal_policy(self, state, peek = False ):
        """
        :param state:
        :return: return the best action in terms of q value and safety
        """
        max_q = self.get_maxQ(state)
        optimal_actions = []
        safe_actions = {}
        best_actions = []
        best_action = None
        print("action values {} in state {}".format(self.Q[state], state))

        for a, v in iter(self.Q[state].items()):
            if v == max_q:
                optimal_actions.append(a)
        print("optimal action: {}".format(optimal_actions))

        if peek == True:
            ## got the optimal action is it safe enough?
            for action in self.valid_actions:
                violation = self.env.feedback(self, state, action)
                if violation == 0:
                    safe_actions.update({action: violation})
                    print("action {} safe ".format(action))

            print("safe_actions: {}".format(safe_actions))

            # choose the highest q from the safe action

            for safe_action, violation in iter(safe_actions.items()):
                for q_action in optimal_actions:
                    if safe_action == q_action:
                        best_actions.append(q_action)

            # the action has the highest score isn't the safe action
            if len(best_actions) == 0:
                best_actions.append(min(safe_actions, key=safe_actions.get))
            print("best_action: {}".format(best_actions))
            best_action = np.random.choice(best_actions)
        else:
            best_action = np.random.choice(optimal_actions)

        print("choose optimal_action: {}".format(best_action))
        return  best_action

    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        if self.learning == False:
            ## 4 action uniformally distributed and randomly choose one
            action = np.random.choice(self.valid_actions)
        else:
            # epsilon greedy explore or when there is no state entry in Q
            if np.random.binomial(1, self.epsilon) == 1 or not bool(self.Q):
                action = np.random.choice(self.valid_actions)
                print("choose random action: {}".format(action))
            else:
                action = self.find_optimal_policy(state,peek=False)
        return action

    def expectedSarsaLearning(self,state, action, reward):
        # expected Sarsa  Q(St,At)←Q(St,At)+ alpha *[ Rt+1+gamma* E[Q(St+1,At+1)|St+1]−Q(St,At)]

        if self.last_state:
            last_value = self.Q[self.last_state][self.last_action]
        else:
            last_value = 0.0

        expected_value = self.get_expectedMaxQ(state)

        if self.last_state:
            self.Q[self.last_state][self.last_action] += self.alpha * (self.last_reward  + expected_value - last_value)
            print("Expected Sarsa updating last state  {} action {} target value  {:4f}"
                  .format(self.last_state, self.last_action, self.Q[self.last_state][self.last_action]))

        self.last_state =  state          # track the last state
        self.last_action = action
        self.last_reward = reward

        return

    def N_StepSarsaLearning(self, state, action, reward):

        update_time = int(self.t - self.n)

        if (update_time >=0 ):
            for time_step in range(update_time, update_time + self.n ):
                print("T: {:d} update time : {:d}".format(self.t, time_step))
                print("reward track: {}" .format(self.rewards_history))
                print("state action trajectory: {}".format(self.state_action_trajectory))
                returns = 0.0
                ## back track
                for t in range (time_step, min(self.t, update_time+ self.n )):
                    returns += self.rewards_history[t]
                    print("reward @ {:d} : {:.4f}".format(t,self.rewards_history[t]))
                # truncate after n steps
                if time_step + self.n < self.t:
                    last_state = self.state_action_trajectory[time_step+self.n][0]
                    last_action = self.state_action_trajectory[time_step+self.n][1]
                    print("reward @ {} : {}".format(last_state, last_action))
                    returns += self.Q[last_state][last_action]
                # update  Q(Sτ , Aτ )  <- Q(Sτ , Aτ ) + α [G − Q(Sτ , Aτ )]
                n_step_state = self.state_action_trajectory[time_step][0]
                n_step_action = self.state_action_trajectory[time_step][1]
                n_step_expected_value = self.get_expectedMaxQ(n_step_state)
                self.Q[n_step_state][n_step_action] += self.alpha * (reward + returns - n_step_expected_value)
                print("N-step Expected Sarsa updating last state  {} action {} target value  {:4f}"
                          .format(n_step_state, n_step_action, self.Q[n_step_state][n_step_action]))


        self.t += 1
        self.state_action_trajectory.append((state,action))
        self.rewards_history.append(reward)
        return

    def QLearning(self,state, action, reward):
        # Q learning Q(S, A) ← Q(S, A) + α [R + γ maxa Q(S′, a) − Q(S, A)]

        if self.last_state:
            lastQ = self.Q[self.last_state][self.last_action]
        else:
            lastQ = 0.0

        maxQ = self.get_maxQ(state)
        print("reward {} for  state {} and  action {} last Q {:.4f}".format(reward,self.last_state,self.last_action, lastQ))
        if self.last_state :
            print("before updating.", self.Q[self.last_state])
            self.Q[self.last_state][self.last_action] += self.alpha * (self.last_reward  + maxQ - lastQ)
            print("Q learning updating  state  {} action {} Q value  {:4f}"
                  .format(self.last_state, self.last_action, self.Q[self.last_state][self.last_action]))
            print("after updating.", self.Q[self.last_state])

        self.last_state =  state          # track the last state
        self.last_action = action
        self.last_reward = reward

        return

    def N_StepQLearning(self, state, action, reward):

        update_time = int(self.t - self.n)

        if (update_time >=0 ):
            for time_step in range(update_time, update_time + self.n ):
                print("T: {:d} update time : {:d}".format(self.t, time_step))
                print("reward track: {}" .format(self.rewards_history))
                print("state action trajectory: {}".format(self.state_action_trajectory))
                returns = 0.0
                ## back track
                for t in range (time_step, min(self.t, update_time+ self.n )):
                    returns += self.rewards_history[t]
                    print("reward @ {:d} : {:.4f}".format(t,self.rewards_history[t]))
                # truncate after n steps
                if time_step + self.n < self.t:
                    last_state = self.state_action_trajectory[time_step+self.n][0]
                    last_action = self.state_action_trajectory[time_step+self.n][1]
                    print("reward @ {} : {}".format(last_state, last_action))
                    returns += self.Q[last_state][last_action]
                # update  Q(Sτ , Aτ )  <- Q(Sτ , Aτ ) + α [G − Q(Sτ , Aτ )]
                n_step_state = self.state_action_trajectory[time_step][0]
                n_step_action = self.state_action_trajectory[time_step][1]
                n_step_max_value = self.get_maxQ(n_step_state)
                self.Q[n_step_state][n_step_action] += self.alpha * (returns - n_step_max_value)
                print("N-step Q Learning updating last state  {} action {} target value  {:4f}"
                          .format(n_step_state, n_step_action, self.Q[n_step_state][n_step_action]))


        self.t += 1
        self.state_action_trajectory.append((state,action))
        self.rewards_history.append(reward)
        return


    def learn(self, state, action, reward, method='QLearning'):

        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        # update the Q state of the last step.
        if method == 'Expected-Sarsa':
            self.expectedSarsaLearning(state, action, reward)
        elif method == 'NStep-Expected-Sarsa':
            self.N_StepSarsaLearning(state, action, reward)
        else:
            self.QLearning(state, action, reward)
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        print("current state: ", state)
        #self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward, method ='QLearning')   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose = True, grid_size = (8,6))

    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent,learning= True, epsilon=1.0, alpha = 0.5, epsilon_decay='exp')
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent , enforce_deadline = True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, display=True, log_metrics=True,optimized = True)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(tolerance = 0.05, n_test = 10 )

if __name__ == '__main__':
    run()
