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

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        #self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.Q = OrderedDict()
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        # track the current state/action and update them once an new action
        # is taken and reward received and state changed
        self.last_state = None
        self.last_action = None
        self.action_index_map = dict(zip(['None', 'forward', 'left', 'right'], [0,1,2,3]))
        self.light_index_map = dict({'green': 0, 'red': 1})

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
        self.epsilon -= 0.01
        if self.epsilon < 0:
            self.epsilon = 0

        if testing == True:
            self.epsilon = 0
            self.alpha = 0

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
        #print("state index  {:d}".format(state_index))

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
        #return self.build_index(inputs)

    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        maxQ = 0.0
        if state in self.Q:
            maxQ = max(self.Q[self.state].values())
            print("maxQ  {:.4f} for {} ".format(maxQ,state))
        return maxQ

    def get_expectedSarsa(self,state):
        # Sum(π(a|St+1)Q(St+1, a))

        expectedQ = 0.0
        optimal_actions = []
        max_q = -np.inf
        if state in self.Q:
            for action, value in iter(self.Q[state].items()):
                if value > max_q:
                    max_q = value
                    optimal_actions.append(action)
            for action,value in iter(self.Q[state].items()):
                if action in optimal_actions:
                    expectedQ += ((1-self.epsilon)/len(optimal_actions) + self.epsilon/4 ) * self.Q[self.state][action]
                else:
                    expectedQ += (self.epsilon / 4) * self.Q[self.state][action]

        return expectedQ



    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0
        if not state in self.Q:
            self.Q.update({state:{}})
            for action in self.valid_actions:
                self.Q[state].update({action:0.0})
        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        # Otherwise, choose an action with the highest Q-value for the current state
        # Be sure that when choosing an action with highest Q-value that you randomly select between actions that "tie".
        if self.learning == False:
            action = np.random.choice(self.valid_actions)
            ## 4 action uniformally distributed and randomly choose one
        else:
            # epsilon greedy explore or when there is no state entry in Q or the state has never been visited
            if np.random.binomial(1, self.epsilon) == 1 or not bool(self.Q):
                action = np.random.choice(self.valid_actions)
                print("choose random action: {}".format(action))
            else:
                max_q = -np.inf
                optimal_action = ''
                print("action values {} in state {}".format(self.Q[state], state))
                for action, value in iter(self.Q[state].items()):
                    if value >max_q:
                        max_q = value
                        optimal_action = action
                action = optimal_action
                print("choose optimal_action: {}".format(action))
        return action


    def learn(self, state, action, reward, method = 'Q-learning'):
        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        # update the Q state of the last step.
        if self.last_state:
            lastQ = self.Q[self.last_state][self.last_action]
        else:
            lastQ = 0.0
        if method == 'Q-learning':
            maxQ = self.get_maxQ(state)
            # Q learning Q(S, A) ← Q(S, A) + α R + γ maxa Q(S′, a) − Q(S, A)
            if self.last_state and self.last_action:
                self.Q[self.last_state][self.last_action] += self.alpha*(reward + maxQ - lastQ )
                print("Q learning updating last state  {} action {} q value  {:4f}"
                      .format(self.last_state, self.last_action,self.Q[self.last_state][self.last_action] ))
        elif method == 'Expected Sarsa':
            # expected Sarsa  Q(St,At)←Q(St,At)+ alpha * Rt+1+gamma* E[Q(St+1,At+1)|St+1]−Q(St,At)
            expectedQ = self.get_expectedSarsa(state)
            if self.last_state and self.last_action:
                self.Q[self.last_state][self.last_action] += self.alpha*(reward + expectedQ - lastQ )
                print("Expected Sarsa updating last state  {} action {} q value  {:4f}"
                      .format(self.last_state, self.last_action,self.Q[self.last_state][self.last_action] ))

        self.last_state =  state          # track the last state
        self.last_action = action
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        print("current state: ", state)
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward,'Expected Sarsa')   # Q-learn

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
    agent = env.create_agent(LearningAgent,learning= True, epsilon=1.0, alpha = 0.5)
    
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
    sim = Simulator(env, update_delay=0.01, display=True, log_metrics=True,optimized = False)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test = 10 )

if __name__ == '__main__':
    run()
