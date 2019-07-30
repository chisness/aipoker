import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
import io
import re
import socket
from player_utilities import UTILITY_DICT, STATE_DICT, PAYOUT_DICT
import kuhn3p.betting
import kuhn3p.deck
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
        self.optimizer = tf.train.RMSPropOptimizer(0.001, 0.99, 0.0, 1e-6)
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

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 160, 160, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(),
             self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables(
    ) if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables(
    ) if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


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
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


import io
import re
import socket
import kuhn3p.betting
import kuhn3p.deck

assert(len(sys.argv) == 2)

address = '127.0.0.1'
port = int(sys.argv[1])

print(address, port)


state_regex = re.compile(
    r"MATCHSTATE:(\d):(\d+):([^:]*):([^|]*)\|([^|]*)\|(.*)")


def maybe_suited_card_string_to_card(x):
    if (x is not None) and len(x) == 2:
        return kuhn3p.deck.string_to_card(x[0])
    else:
        return -1


def deep_q_learning(sess,
                    q_estimator,
                    target_estimator,
                    # state_processor,
                    num_episodes,
                    experiment_dir,
                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_estimator_every=10000,
                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        q_estimator: Estimator object used for the q values
        target_estimator: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
          the reply memory.
        update_target_estimator_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes

    Returns:
        An EpisodeStats object with two numpy arrays for episode_rewards.
    """

    Transition = namedtuple(
        "Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    default_state = np.array(
            [np.zeros(9), np.zeros(9), np.zeros(9)], dtype=np.uint8)

    assert(default_state.shape == (3, 9))
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes), 
        episode_rewards=np.zeros(num_episodes))

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

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

  

    # Populate the replay memory with initial experience
    print("Populating replay memory...")
    
    i = 0

    while i < replay_memory_init_size:

        next_state = deque([default_state for _ in range(10)], 10)
        last_state = deque([default_state for _ in range(10)], 10)    
        sock = socket.create_connection((address, port))
        sockin = sock.makefile(mode='rb')
        sock.send(('VERSION:2.0.0\r\n').encode())
        hand = None
        position = None
        state = None
        reward = -1
        action = None
        done = False
        cards = []
        hand = None
        position = None
        played = False

        while True:

            line = sockin.readline().strip().decode()

            if not line:
                break

            cur_state = state_regex.match(line)

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

                next_state.append(cur_state)

                np_state = np.array(next_state)
            
                assert(np_state.shape == (10, 3, 9))

                action_probs = policy(
                    sess, np_state, epsilons[min(total_t, epsilon_decay_steps - 1)])
                action = np.random.choice(
                    np.arange(len(action_probs)), p=action_probs)

                action = VALID_ACTIONS[action]

                response = '%s:%s\r\n' % (
                    line, kuhn3p.betting.action_name(betting, action))

                sock.send(response.encode())

                if played:
                    # Save transition to replay memory
                    replay_memory.append(Transition(
                        np.array(last_state), action, -1, np.array(next_state), done))

                state = cur_state

                last_state = next_state

                played = True

            elif kuhn3p.betting.is_terminal(betting):

                assert not (cards[position] is None)
                # done with this hand
                done = True

                reward = UTILITY_DICT.get(cur_state.group(3))(position, cards)

                # reward = reward // abs(reward)

                cur_state = create_state(cur_state)

                next_state.append(cur_state)

                replay_memory.append(Transition(
                    np.array(last_state), action, reward, np.array(next_state), done))

                i += 1

            if done:
                position = None
                played = False
                state = None
                reward = 0
                action = None
                done = False
                cards = []
                position = None
                hand = None

        sock.close()
        sock = None
        sockin = None
        sleep(0.5)

        print('populated memory:', i)
        

    i_episode = 0

    while i_episode < num_episodes:

        next_state = deque([default_state for _ in range(10)], 10)
        last_state = deque([default_state for _ in range(10)], 10)

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)
        sock = socket.create_connection((address, port))
        sockin = sock.makefile(mode='rb')
        sock.send(('VERSION:2.0.0\r\n').encode())
        hand = None
        position = None
        state = None
        done = False
        action = None
        cards = []
        loss = None
        played = False

        # One step in the environment
        for t in itertools.count():

            line = sockin.readline().strip().decode()
            
            
            # safety
            if not line:
                break

            # get next state
            cur_state = state_regex.match(line)

            this_position, this_hand = [int(x) for x in cur_state.group(1, 2)]
            betting = kuhn3p.betting.string_to_state(cur_state.group(3))
            cards = list(map(maybe_suited_card_string_to_card,
                             cur_state.group(4, 5, 6)))

            if not (this_hand == hand):
                assert hand is None
                position = this_position
                hand = this_hand
                
                assert not (cards[position] is None)

            if kuhn3p.betting.is_internal(betting) and kuhn3p.betting.actor(betting) == position:

                assert not (cards[position] is None)

                cur_state = create_state(cur_state)

                next_state.append(cur_state)
                # Epsilon for this time step
                epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

                # Add epsilon to Tensorboard
                episode_summary = tf.Summary()
                episode_summary.value.add(simple_value=epsilon, tag="epsilon")
                q_estimator.summary_writer.add_summary(
                    episode_summary, total_t)


                np_state = np.array(next_state)
            
                assert(np_state.shape == (10, 3, 9))

                # Take a step
                action_probs = policy(sess, np_state, epsilon)
                action = np.random.choice(
                    np.arange(len(action_probs)), p=action_probs)

                action = VALID_ACTIONS[action]

                response = '%s:%s\r\n' % (
                    line, kuhn3p.betting.action_name(betting, action))

                sock.send(response.encode())

                if played:

                    if len(replay_memory) == replay_memory_size:
                        replay_memory.pop(0)

                    # update replay memory
                    replay_memory.append(Transition(
                        np.array(last_state), action, -1, np.array(next_state), done))

                    # every internal action is a negative reward
                    # stats.episode_rewards[i_episode] += -1

                    # Sample a minibatch from the replay memory
                    samples = random.sample(replay_memory, batch_size)
                    states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(
                        np.array, zip(*samples))

                    # Calculate q values and targets (Double DQN)
                    q_values_next = q_estimator.predict(
                        sess, next_states_batch)
                    best_actions = np.argmax(q_values_next, axis=1)
                    q_values_next_target = target_estimator.predict(
                        sess, next_states_batch)
                    targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                        discount_factor * \
                        q_values_next_target[np.arange(
                            batch_size), best_actions]

                    # Perform gradient descent update
                    states_batch = np.array(states_batch)
                    loss = q_estimator.update(
                        sess, states_batch, action_batch, targets_batch)

                played = True

                last_state = next_state

                state = cur_state

            elif kuhn3p.betting.is_terminal(betting):

                assert not (cards[position] is None)
                done = True

                reward = UTILITY_DICT.get(cur_state.group(3))(position, cards)

                # normalizing reward
                # reward = reward // abs(reward)
                # print('reward', reward)

                  # Print out which step we're on, useful for debugging.
                print("\rStep {} ({}) @ Episode {}/{}, reward: {}, loss: {}\n".format(
                    t, total_t, i_episode + 1,num_episodes, reward ,loss), end="")
                sys.stdout.flush()


                cur_state = create_state(cur_state)

                if len(replay_memory) == replay_memory_size:
                    replay_memory.pop(0)

                next_state.append(cur_state)

                # update replay memory
                replay_memory.append(Transition(
                    np.array(last_state), action, reward, np.array(next_state), done))

                # Update statistics
                stats.episode_rewards[i_episode] = reward

                # Sample a minibatch from the replay memory
                samples = random.sample(replay_memory, batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(
                    np.array, zip(*samples))

                # Calculate q values and targets (Double DQN)
                q_values_next = q_estimator.predict(sess, next_states_batch)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = target_estimator.predict(
                    sess, next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                    discount_factor * \
                    q_values_next_target[np.arange(batch_size), best_actions]

                # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = q_estimator.update(
                    sess, states_batch, action_batch, targets_batch)

                i_episode += 1

                total_t += 1

                # Add summaries to tensorboard
                episode_summary = tf.Summary()
                episode_summary.value.add(
                    simple_value=reward, tag="episode_reward")
                q_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target network.")

            if done:
                # print('hand finished, hand reward:', reward)
                hand = None
                position = None
                played = False
                state = None
                done = False
                action = None
                reward = 0
                cards = []



        sock.close()
        sock = None
        sockin = None
        
        q_estimator.summary_writer.flush()
        
        yield total_t, plotting.EpisodeStats(episode_lengths=stats.episode_lengths[:i_episode + 1], 
                                             episode_rewards=stats.episode_rewards[:i_episode + 1])


    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath(
    "./experiments/{}".format('NeuralNetAgent10x3x9'))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

# State processor
# state_processor = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    # state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=10000000,
                                    replay_memory_size=60000,
                                    replay_memory_init_size=15000,
                                    update_target_estimator_every=3000,
                                    epsilon_start=0.90,
                                    epsilon_end=0.0,
                                    epsilon_decay_steps=10000000,
                                    discount_factor=0.5,
                                    batch_size=1000):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
