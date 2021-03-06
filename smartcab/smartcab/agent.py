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
from sklearn.tree import DecisionTreeRegressor


class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5, epsilon_decay='linear'):
        super(LearningAgent, self).__init__(env)  # Set the agent in the evironment
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning
        if self.learning:
            self.Q = OrderedDict()
        self.alpha = alpha  # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.verbose = True
        self.action_index_map = dict(zip(['forward', 'left', 'right', None], [0, 1, 2, 3]))
        self.light_index_map = dict({'green': 0, 'red': 1})
        self.t = 0  # time steps
        self.trial_times = 0
        self.epsilon_decay = epsilon_decay

        if self.epsilon_decay == 'linear' or not self.learning:
            self.epsilon = 1.0
        else:
            self.epsilon_ceiling = epsilon
        self.pred_thres = int(384 * 0.80)  # 80% the table filled

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
        if not self.learning or self.epsilon_decay == 'linear':
            self.epsilon -= 0.05
        else:
            if self.trial_times <= 100:
                # learng the rules  aggresively when exploring
                self.epsilon = self.epsilon_ceiling / (1 + np.exp(0.09 * (self.trial_times - 100)))
                self.alpha = 0.5
            else:
                # tune its table slowly when exploiting
                self.epsilon = 0.5 * self.epsilon_ceiling * np.exp(-0.03 * (self.trial_times - 100))
                # learning rate decays as agent more confident in driving on its own and stablize Q
                self.alpha = 0.2

        if self.epsilon < 0:
            self.epsilon = 0

        if testing == True:
            self.epsilon = 0
            self.alpha = 0

        self.t = 0
        self.trial_times += 1
        return None

    # waypoint has 3 possible actions ['forward', 'left', 'right']
    # Inputs = {'light': 'green' or 'red', 'oncoming': 4 directions, 'right': 4 directions, 'left': 4 directions}
    # build a map from (inputs, waypoint) to Q.table index
    def build_index(self, inputs, waypoint=None):
        ##  state[1:0] = action for 'right'
        ##  state[3:2] = action for 'left'
        ##  state[5:4] = action for 'oncoming'
        ##  state[6] =  color for 'light'
        ##  state[8:7] = waypoint action

        state_index = 0

        if waypoint:
            state_index = (self.action_index_map[waypoint]) << 7

        state_index += self.light_index_map[inputs['light']] << 6

        state_index += self.action_index_map[inputs['oncoming']] << 4

        state_index += self.action_index_map[inputs['left']] << 2

        state_index += self.action_index_map[inputs['right']]

        if self.verbose:
            if waypoint:
                print("current state {} {}".format(state_index, (waypoint, inputs)))
            else:
                print("current state {} {}".format(state_index, (inputs)))

        return state_index

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint()  # The next waypoint
        inputs = self.env.sense(self)  # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########

        # NOTE : you are not allowed to engineer features outside of the inputs available.
        # Because the aim of this project is to teach Reinforcement Learning, we have placed 
        # constraints in order for you to learn how to adjust epsilon and alpha, and thus learn about the balance between exploration and exploitation.
        # With the hand-engineered features, this learning process gets entirely negated.

        # Set 'state' as a tuple of relevant data for the agent        
        return self.build_index(inputs, waypoint)

    def get_maxQ(self, state, prediction=False):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        maxQ = 0.0
        if state in self.Q:
            if self.verbose:
                print("{}  {}".format(self.Q[state].items(), self.Q[state].values()))

            if prediction and len(self.Q) > self.pred_thres:  ## at least to have some entries
                for action, q in iter(self.Q[state].items()):
                    if q == 0.0:
                        # self.Q[state][action] = self.fit_and_predit_Linear(state,action)
                        self.Q[state][action] = self.fit_and_predit_DT(state, action)
            maxQ = max(self.Q[state].values())
        else:
            if self.verbose:
                print("{} is never seen before ".format(state))
        return maxQ

    def fit_and_predit_DT(self, state, action):

        Q_train = []
        waypoint_feature = []
        light_feature = []
        oncoming_feature = []
        left_feature = []
        right_feature = []
        action_feature = []

        for st, action_dict in iter(self.Q.items()):
            for a, q in iter(action_dict.items()):
                if q != 0.0:
                    waypoint_feature.append((st >> 7) & 0x3)
                    light_feature.append((st >> 6) & 0x1)
                    oncoming_feature.append(st >> 4 & 0x3)
                    left_feature.append((st >> 2) & 0x03)
                    right_feature.append(st & 0x3)
                    action_feature.append(self.action_index_map[a])
                    Q_train.append(q)

        features_train = np.array((np.array(waypoint_feature), np.array(light_feature), np.array(oncoming_feature), \
                                   np.array(left_feature), np.array(right_feature), np.array(action_feature))).T
        Q_train = np.array(Q_train).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(features_train, Q_train, test_size=0.2, random_state=42)

        regr = DecisionTreeRegressor(max_depth=3)

        regr.fit(X_train, y_train)

        if self.verbose:
            print("DT tree feature importance", regr.feature_importances_)

        feature_test = np.array(((state >> 7) & 0x3, (state >> 6) & 0x1, (state >> 4) & 0x3, (state >> 2) & 0x3, \
                                 state & 0x3, self.action_index_map[action])).reshape(1, 6)

        predict_Q = regr.predict(feature_test)  ## this is a numpy array need to be list

        if self.verbose:
            print("predicting :", predict_Q[0], "for action", action)
        return float(predict_Q[0])

    def fit_and_predit_Linear(self, state, action):
        """

        :param state:
        :param action:
        :return: return predicted four Q values for a unseen state using all existing table data
        """
        # from X  and Y for train and prediction
        state_actions = []
        ## It probably would go better with decision regressor with feature matrix

        Q = []

        results = dict({'acc_train': 0.0, 'acc_test': 0.0})
        predict_Q = 0.0
        for st, action_dict in iter(self.Q.items()):
            for a, q in iter(action_dict.items()):
                if q != 0.0:
                    state_actions.append(int(st) << 2 + self.action_index_map[a])
                    Q.append(q)

        state_actions = np.array(state_actions).reshape(-1, 1)
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
        # results['acc_train'] = accuracy_score(y_train, y_predict_train)
        # results['acc_test'] = accuracy_score(y_test, y_predict_test)
        # print("accuracy on train data {:.4f} on test data {:.4f}".format(results['acc_train'],results['acc_test'] ))

        # predict real thing
        predict_Q = regr.predict(int(st) << 2 + self.action_index_map[action])  ## this is a numpy array need to be list
        if self.verbose:
            print("predicting :", predict_Q[0], "for action", action)
        return float(predict_Q[0])

    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0
        if self.learning and not state in self.Q:
            self.Q.update({state: {}})
            for action in self.valid_actions:
                self.Q[state][action] = 0.0
        return

    def find_optimal_action(self, state, peek=False):
        """
        :param state:
        :return: return the best action in terms of q value and safety
        """
        max_q = self.get_maxQ(state, prediction=True)
        optimal_actions = []
        safe_actions = {}
        best_actions = []
        best_action = None

        if self.verbose:
            print("action values {} in state {}".format(self.Q[state], state))

        optimal_actions = [action for action in self.valid_actions if self.Q[state][action] == max_q]

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
            safe_actions = sorted(safe_actions.items(), key=itemgetter(1))

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
            best_action = sorted(best_actions, key=itemgetter(1))
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
        print(self.learning)
        if not self.learning:
            ## 4 action uniformally distributed and randomly choose one
            action = np.random.choice(self.valid_actions)
        else:
            # epsilon greedy explore or when there is no state entry in Q
            if np.random.binomial(1, self.epsilon) == 1 or not bool(self.Q):
                action = np.random.choice(self.valid_actions)
                if self.verbose:
                    print("choose random action: {}".format(action))
            else:
                action = self.find_optimal_action(state, peek=False)
        return action

    def learn(self, state, action, reward):

        """ The learn function is called after the agent completes an action and
            receives a reward. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')

        # Q learning Q(S, A) ← Q(S, A) + α [R + γ max Q(S′, a) − Q(S, A)]
        # since gamma = 0.0 the Q learning simplified to Q(S, A) ← Q(S, A) + α [R  − Q(S, A)]
        # print(self.learning)
        if self.learning:
            maxQ = self.get_maxQ(state)
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
            if self.verbose:
                print("Q learning updating  state", state, " action", action, "Q value: ", self.Q[state][action])

            ## planning
            ## what about other actions not chosen in this state. predict its value for future use
            for act in self.valid_actions:
                if act != action and self.Q[state][act] == 0.0 and len(
                        self.Q) > self.pred_thres:  ## 80% of the state seen
                    ## predict
                    # self.Q[state][act] = self.fit_and_predit_Linear(state, act)
                    self.Q[state][act] = self.fit_and_predit_DT(state, act)
        return

    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()  # Get current state
        self.createQ(state)  # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action)  # Receive a reward
        self.learn(state, action, reward)  # Q-learn

        return


def run(learning, decay, optim=None):
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=True, grid_size=(8, 6))

    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    #    *epsilon_decay - linear epsilon decay or sigmoid shaped epsilon decay
    agent = env.create_agent(LearningAgent, learning, epsilon=1.0, alpha=0.5, epsilon_decay=decay)

    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, display=True, log_metrics=True, optimized=optim)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(tolerance=0.02, n_test=10)


def str2bool(v):
  return v.lower() in ("True","1")

if __name__ == "__main__":
    import argparse
    import ast
    parser = argparse.ArgumentParser(description='smartcar ')
    parser.add_argument("-l", "-learning", dest="learning", action="store", default="False", help="True/1 or False/0 ")
    parser.add_argument("-o", "--optimized", dest="optimized", action="store", default="False", help="True/1 or False/0 ")
    args = parser.parse_args()

    learning = ast.literal_eval(args.learning)
    optimized = ast.literal_eval(args.optimized)
    #print(learning,optimized)
    if False == learning :
        learning = None
        optimized = None
    elif False == optimized :
        optimized = None

    if True == optimized:
        epsilon_decay = 'exp'
    else:
        epsilon_decay = 'linear'

    print("learning flag ", learning, "optimized", optimized, "epsilon_decay function",epsilon_decay )
    run(learning, epsilon_decay, optimized)
