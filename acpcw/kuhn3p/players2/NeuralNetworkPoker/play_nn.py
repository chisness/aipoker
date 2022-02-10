import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import io
import re
import socket
from player_utilities import UTILITY_DICT, STATE_DICT
import kuhn3p.betting
import kuhn3p.deck
from weights import weights
from time import sleep

if "../" not in sys.path:
    sys.path.append("../")

from ext_lib import plotting
from collections import deque, namedtuple


# CHECK = 0 # c
# CALL = 0 #  c
# BET = 1 # r
# FOLD = 1 # f

# poker Actions: 0 (check, call) 1 (bet, fold)
VALID_ACTIONS = [0, 1]


def create_state(state):
    player = int(state.group(1))
    # encoding the hand number into binary
    hand = list(map(int, list(bin(int(state.group(2))))[2:]))
    padded_hand = np.array(np.pad(hand, (0, 12 - len(hand)), 'constant'))
    # print(padded_hand, len(padded_hand))
    cards = list(map(maybe_suited_card_string_to_card, state.group(4, 5, 6)))
    player_vector = np.array(
        [np.zeros(9), np.zeros(9), np.zeros(9)], dtype=np.uint8)
    cards_vector = np.zeros((4,), dtype=int)
    node = state.group(3)
    key = STATE_DICT[node] if node else STATE_DICT['i']
    situation_vector = np.zeros((5,), dtype=int)
    player_vector[player][cards[player]] = np.uint8(1)
    player_vector[player][key - 1] = np.uint8(1)
    # player_vector[player][9:] = padded_hand
    assert(player_vector.shape == (3, 9))
    # print(player_vector.shape)
    return player_vector


class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(
                    summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 RGB frames of shape 160, 160 each
        self.X_pl = tf.placeholder(
            shape=[None, 10, 3, 9], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(
            shape=[None], dtype=tf.int32, name="actions")

        batch_size = tf.shape(self.X_pl)[0]
        X = tf.to_float(self.X_pl)

        flattened = tf.contrib.layers.flatten(X)

        # # Five fully connected layers
        fc1 = tf.contrib.layers.fully_connected(flattened, 512, tf.nn.relu)

        fc2 = tf.contrib.layers.fully_connected(fc1, 512, tf.nn.relu)

        self.predictions = tf.contrib.layers.fully_connected(
            fc2, len(VALID_ACTIONS), tf.nn.relu)

        # Get the predictions for the chosen actions only
        gather_input = tf.reshape(self.predictions, [-1])
        gather_indices = tf.range(
            batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(gather_input, gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.01, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.train.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 160, 160, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions, {self.X_pl: s})


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, state, card):
        q_values = np.array(estimator.predict(
            sess, np.expand_dims(observation, 0))[0])
        q_sum = sum(q_values)
        print('state q_values:', state, q_values, q_sum)
        if q_sum == 0.0:
            return weights['v2'][state or 'i'][str(card)]
        policy = q_values / q_sum
        return policy

    return policy_fn


import io
import re
import socket
import kuhn3p.betting
import kuhn3p.deck

assert(len(sys.argv) >= 2)

address = '127.0.0.1'


if len(sys.argv) == 3:
    address = sys.argv[1]
    port = sys.argv[2]
else:
    port = int(sys.argv[1])

print(address, port)

state_regex = re.compile(
    r"MATCHSTATE:(\d):(\d+):([^:]*):([^|]*)\|([^|]*)\|(.*)")


def maybe_suited_card_string_to_card(x):
    if (x is not None) and len(x) == 2:
        return kuhn3p.deck.string_to_card(x[0])
    else:
        return -1


def play_q_learning(sess,
                    q_estimator,
                    experiment_dir,
                    epsilon=0.0):
    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_t = sess.run(tf.train.get_global_step())

    print('total training', total_t)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    sock = socket.create_connection((address, port))
    sockin = sock.makefile(mode='rb')
    sock.send(('VERSION:2.0.0\r\n').encode())

    hand = None
    position = None

    state = deque([np.array(
        [np.zeros(9), np.zeros(9), np.zeros(9)], dtype=np.uint8) for _ in range(10)], 10)

    while True:

        line = sockin.readline().strip().decode()

        if not line:
            break

        cur_state = state_regex.match(line)

        state_tuple = cur_state.group(1, 2, 3, 4, 5, 6)

        this_position, this_hand = [int(x) for x in cur_state.group(1, 2)]
        betting = kuhn3p.betting.string_to_state(cur_state.group(3))
        cards = list(map(maybe_suited_card_string_to_card,
                         cur_state.group(4, 5, 6)))

        if (this_hand != hand):
            assert hand is None
            position = this_position
            hand = this_hand
            assert (cards[position] is not None)

        if kuhn3p.betting.is_internal(betting) and kuhn3p.betting.actor(betting) == position:

            assert not (cards[position] is None)

            cur_state = create_state(cur_state)

            state.append(cur_state)

            np_state = np.array(state)

            assert(np_state.shape == (10, 3, 9))

            action_probs = policy(
                sess, np_state, state_tuple[2], cards[position])

            # print('policy: at state', state_tuple, action_probs)

            action = np.random.choice(
                np.arange(len(action_probs)), p=action_probs)

            action = VALID_ACTIONS[action]

            response = '%s:%s\r\n' % (
                line, kuhn3p.betting.action_name(betting, action))

            sock.send(response.encode())

        elif kuhn3p.betting.is_terminal(betting):

            assert not (cards[position] is None)
            # done with this hand
            reward = UTILITY_DICT.get(cur_state.group(3))(position, cards)

            print('reward:',reward, 'card:', cards[position], 'position:', position ,'history:', cur_state.group(3))

            hand = None

    sock.close()
    sock = None
    sockin = None
    sleep(0.5)


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath(
    "./experiments/{}".format('NeuralNetAgent10x3x9'))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)

# State processor
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    play_q_learning(sess,
                    q_estimator=q_estimator,
                    experiment_dir=experiment_dir,
                    epsilon=0.0)
