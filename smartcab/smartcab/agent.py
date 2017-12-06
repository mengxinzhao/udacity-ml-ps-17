import random
import math
import numpy as np
from collections import OrderedDict
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn import linear_model


class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5, epsilon_decay = 'linear'):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = OrderedDict()
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        # track the current state/action and update them once an new action
        # is taken and reward received and state changed
        # use for one step learning back-up
        self.verbose = True
        self.last_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.action_index_map = dict(zip(['forward', 'left', 'right',None], [0,1,2,3]))
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
            self.epsilon -= 0.05
        else:
            if self.trial_times <= 100:
                self.epsilon =  self.epsilon_ceiling/(1 + np.exp(0.05 * (self.trial_times-100)))
                self.alpha = 0.5  ##learng the rules more aggresively when exploring
            else:
                self.epsilon = 0.5 * self.epsilon_ceiling  * np.exp(-0.02 * (self.trial_times-100))
                self.alpha = 0.2  ## tune its table slowly when exploting
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

    # waypoint has 3 possible actions ['forward', 'left', 'right']
    # Inputs = {'light': 'green' or 'red', 'oncoming': 4 directions, 'right': 4 directions, 'left': 4 directions}
    # build a map from (inputs, waypoint) to Q.table index
    def build_index (self, inputs,waypoint = None):
        ##  state[1:0] = action for 'right'
        ##  state[3:2] = action for 'left'
        ##  state[5:4] = action for 'oncoming'
        ##  state[6] =  color for 'light'
        ##  state[8:7] = waypoint action

        state_index = 0

        if waypoint:
            state_index = (self.action_index_map[waypoint]) << 7

        state_index += self.light_index_map[inputs['light']]<<6

        state_index += self.action_index_map[inputs['oncoming']]<<4

        state_index += self.action_index_map[inputs['left']]<<2

        state_index += self.action_index_map[inputs['right']]

        if self.verbose:
            if waypoint:
                print("current state {} {}".format(state_index, (waypoint,inputs)))
            else:
                print("current state {} {}".format(state_index, (inputs)))

        return state_index

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        #for key, value in iter(inputs.items()):
        #    if value is None:
        #        inputs.update({key:'None'})
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


    def get_maxQ(self, state, prediction = False):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        maxQ = 0.0
        if state in self.Q:
            if self.verbose:
                print("{}  {}".format(self.Q[state].items(),self.Q[state].values()))

            if prediction and len(self.Q) > 100:    ## at least to have some entries
                for action, q in iter(self.Q[state].items()):
                    print(action, q)
                    if q == 0.0:
                        self.Q[state][action] = self.fit_and_predit_Q(state,action)
            maxQ = max(self.Q[state].values())
        else:
            if self.verbose:
                print("{} is never seen before ".format(state))
        return maxQ

    def fit_and_predit_Q(self, state, action):
        """

        :param state:
        :param action:
        :return: return predicted four Q values for a unseen state using all existing table data
        """
        # from X  and Y for train and prediction
        state_actions = []
        ## It probably would go better with decision regressor with feature matrix
        #features = []  # waypoint, light, right, left, oncoming
        #waypoint = []
        #light = []
        #right = []
        #left = []
        #oncoming = []
        #actions = []

        Q = []

        results = dict({'acc_train':0.0,'acc_test':0.0})
        predict_Q = 0.0
        for st,action_dict in iter(self.Q.items()):
            #print(st,action_dict)
            for a,q in iter(action_dict.items()):
                if q!=0.0:
                    state_actions.append(int(st)<<2 + self.action_index_map[a])
                    Q.append(q)
        #print("training X shape:", np.array(state_actions).shape)
        #print("training Y:",np.array(Q).shape)
        state_actions = np.array(state_actions).reshape(-1,1)
        Q = np.array(Q).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(state_actions,
                                                            Q,
                                                            test_size=0.2,
                                                            random_state=42)
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        y_predict_train = regr.predict(X_train)
        y_predict_test = regr.predict(X_test)

        # precision score
        # sklearn doesnt support continuous accuracy_score
        #results['acc_train'] = accuracy_score(y_train, y_predict_train)
        #results['acc_test'] = accuracy_score(y_test, y_predict_test)
        #print("accuracy on train data {:.4f} on test data {:.4f}".format(results['acc_train'],results['acc_test'] ))

        # predict real thing
        predict_Q = regr.predict(int(st)<<2 + self.action_index_map[action])## this is a numpy array need to be list
        if self.verbose:
            print("predicting :", predict_Q[0], "for action",action)
        return float(predict_Q[0])


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
                self.Q[state][action]=0.0
        return

    def find_optimal_policy(self, state, peek = False ):
        """
        :param state:
        :return: return the best action in terms of q value and safety
        """
        max_q = self.get_maxQ(state, True)
        optimal_actions = []
        safe_actions = {}
        best_actions = []
        best_action = None
        if self.verbose:
            print("action values {} in state {}".format(self.Q[state], state))

        for a, v in iter(self.Q[state].items()):
            if v == max_q:
                optimal_actions.append(a)

        if self.verbose:
            print("optimal action: {}".format(optimal_actions))

        if peek == True:
            ## get feedback from the environment
            for action in self.valid_actions:
                violation, dist_delta = self.env.feedback(self, state, action)
                if violation == 0:
                    safe_actions.update({action: dist_delta})

            # sort the safe action in ascending order of dist_delta
            # once sort it is  list with tuples
            safe_actions= sorted(safe_actions.items(),key=itemgetter(1))

            if self.verbose:
                print("safe_actions: {}".format(safe_actions))

            # choose the highest q from the safe action
            for safe_action, dist_delta in safe_actions:
                if safe_action in optimal_actions:
                    best_actions.append((safe_action, dist_delta))
                ## second choice. Not the max q but moves closer to the destination. delta is the smallest
                elif dist_delta == safe_actions[0][1]:
                    best_actions.append((safe_action, dist_delta))

            # sort
            best_action = sorted(best_actions,key=itemgetter(1))
            best_action, _ = best_actions[0]
        else:
            best_action = np.random.choice(optimal_actions)

        if self.verbose:
            print("choose best_action: {}".format(best_action))
        return best_action

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
                if self.verbose:
                    print("choose random action: {}".format(action))
            else:
                action = self.find_optimal_policy(state, peek=False)
        return action

    def QLearning(self,state, action, reward, gamma = 0.0):
        # Q learning Q(S, A) ← Q(S, A) + α [R + γ max Q(S′, a) − Q(S, A)]

        if self.last_state:
            lastQ = self.Q[self.last_state][self.last_action]
        else:
            lastQ = 0.0

        maxQ = self.get_maxQ(state)
        if self.last_state :
            self.Q[self.last_state][self.last_action] += self.alpha * (self.last_reward  + gamma *  maxQ - lastQ)
            if self.verbose:
                print("Q learning updating  state  {} action {} Q value  {:4f}"
                  .format(self.last_state, self.last_action, self.Q[self.last_state][self.last_action]))

        return

    def QLearning2(self,state, action, reward):
        # Q learning Q(S, A) ← Q(S, A) + α [R + γ max Q(S′, a) − Q(S, A)]
        # since gamma = 0.0 the Q learning simplified to Q(S, A) ← Q(S, A) + α [R  − Q(S, A)]
        maxQ = self.get_maxQ(state)

        self.Q[state][action] += self.alpha * (reward  - self.Q[state][action])
        if self.verbose:
            #print("Q learning updating  state  {} action {} Q value  {:4f}".format(state, action, self.Q[state][action]))
            print("Q learning updating  state",state," action", action, "Q value: ", self.Q[state][action])


    def learn(self, state, action, reward, method='QLearning'):

        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        #self.QLearning(state, action, reward,gamma = 0.0)
        self.QLearning2(state, action, reward )
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
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
    agent = env.create_agent(LearningAgent,learning= True, epsilon=1.0, alpha = 0.2, epsilon_decay='exp')
    
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
    sim.run(tolerance = 0.02, n_test = 20 )

if __name__ == '__main__':
    run()
