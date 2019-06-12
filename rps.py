import numpy as np 
import random
import matplotlib.pyplot as plt

ROCK = 0
PAPER = 1
SCISSORS = 2
ACTIONS = [ROCK, PAPER, SCISSORS]
NUM_ACTIONS = 3


class RPSCFRPlayer:
	def __init__(self):
		self.games_played = 0
		self.regret_sum = np.zeros(NUM_ACTIONS)
		self.regret_sum_noneg = np.zeros(NUM_ACTIONS)
		self.strategy = np.zeros(NUM_ACTIONS)
		self.strategy_noneg = np.zeros(NUM_ACTIONS)
		self.strategy_sum = np.zeros(NUM_ACTIONS)
		self.strategy_sum_noneg = np.zeros(NUM_ACTIONS)
		self.winnings = 0

	def get_strategy(self):
		normalizing_sum = 0
		for a in range(NUM_ACTIONS):
			if self.regret_sum[a] > 0:
				self.strategy[a] = self.regret_sum[a]
			else:
				self.strategy[a] = 0
			normalizing_sum += self.strategy[a]

		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				self.strategy[a] /= normalizing_sum
			else:
				self.strategy[a] = 1.0/NUM_ACTIONS
			self.strategy_sum[a] += self.strategy[a]

		return self.strategy

	def get_strategy_noneg(self):
		normalizing_sum = 0
		for a in range(NUM_ACTIONS):
			if self.regret_sum_noneg[a] > 0:
				self.strategy_noneg[a] = self.regret_sum_noneg[a]
			else:
				self.strategy_noneg[a] = 0
				self.regret_sum_noneg[a] = 0
			normalizing_sum += self.strategy_noneg[a]

		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				self.strategy_noneg[a] /= normalizing_sum
			else:
				self.strategy_noneg[a] = 1.0/NUM_ACTIONS
			self.strategy_sum_noneg[a] += self.strategy_noneg[a]

		return self.strategy_noneg


	def get_average_strategy(self):
		avg_strategy = np.zeros(NUM_ACTIONS)
		normalizing_sum = 0
		
		for a in range(NUM_ACTIONS):
			normalizing_sum += self.strategy_sum[a]
		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
			else:
				avg_strategy[a] = 1.0 / num_actions
		
		return avg_strategy


	def get_average_strategy_noneg(self):
		avg_strategy = np.zeros(NUM_ACTIONS)
		normalizing_sum = 0
		
		for a in range(NUM_ACTIONS):
			normalizing_sum += self.strategy_sum_noneg[a]
		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				avg_strategy[a] = self.strategy_sum_noneg[a] / normalizing_sum
			else:
				avg_strategy[a] = 1.0 / num_actions
		
		return avg_strategy


class RPSFixedPlayer:
	def __init__(self):
		#self.strategy = [0.4, 0.3, 0.3]
		self.strategy = [1/3, 1/3, 1/3]
		self.winnings = 0

	def get_strategy(self):
		return self.strategy

	def get_average_strategy(self):
		return self.strategy


class RPS:
	def __init__(self, player1, player2):
		self.games_played = 0
		self.player1 = player1
		self.player2 = player2


	def check_result(self, p1, p2): #results from p1 perspective
		if p1 == 0:
			if p2 == 0:
				return 0
			elif p2 == 1:
				return -1
			elif p2 == 2:
				return 1

		if p1 == 2:
			if p2 == 0:
				return -1
			elif p2 == 1:
				return 1
			elif p2 == 2:
				return 0

		if p1 == 1:
			if p2 == 0:
				return 1
			elif p2 == 1:
				return 0
			elif p2 == 2:
				return -1


	def play_games(self, num_games = 1000000):
		utility = np.zeros(NUM_ACTIONS)
		current_strat_graph = []
		average_strat_graph = []
		current_strat_graph_noneg = []
		average_strat_graph_noneg = []
		ticks_graph = []
		counter = 0
		for i in range(num_games):
			p1_strategy = p1.get_strategy()
			p1_move = np.random.choice(ACTIONS, p=p1_strategy)
			#print('before', p1_strategy)

			p1_strategy_noneg = p1.get_strategy_noneg()
			p1_move_noneg = np.random.choice(ACTIONS, p=p1_strategy_noneg)

			p2_strategy =p2.get_strategy()
			p2_move = np.random.choice(ACTIONS, p=p2_strategy)

			result = self.check_result(p1_move, p2_move)
			p1.winnings += result
			p2.winnings -= result

			if counter % 100 == 0:
				#print('in counter', p1_strategy)
				current_strat_graph.append(np.array(p1.get_strategy()[:]))
				#print('curr strat', current_strat_graph)
				average_strat_graph.append(p1.get_average_strategy()[:])
				current_strat_graph_noneg.append(np.array(p1.get_strategy_noneg()[:]))
				average_strat_graph_noneg.append(p1.get_average_strategy_noneg()[:])
				ticks_graph.append(counter)

			result_noneg = self.check_result(p1_move_noneg, p2_move)

			for a in range(NUM_ACTIONS):
					utility[a] = self.check_result(a, p2_move)
					#print('util', utility[a])

			for a in range(NUM_ACTIONS):
				#print(utility[a] - utility[p1_move])
				p1.regret_sum[a] += (utility[a] - utility[p1_move])
				p1.regret_sum_noneg[a] += (utility[a] - utility[p1_move_noneg])
				#regret is other move minus our move, i.e., alternative compared to move we took

			# print('p1 strategy', p1.get_strategy())
			# print('p1 average strategy', p1.get_average_strategy())
			# print('p1 regrets', p1.regret_sum)
			# print('p1 winnings', p1.winnings)
			counter += 1

		print(current_strat_graph)
		print(average_strat_graph)
		# print(ticks_graph)
		# print(current_strat_graph)
		current_strat_graph = np.array(current_strat_graph)
		average_strat_graph = np.array(average_strat_graph)
		current_strat_graph_noneg = np.array(current_strat_graph_noneg)
		average_strat_graph_noneg = np.array(average_strat_graph_noneg)
		plt.figure()
		plt.subplot(121) #1 row, 2 columns
		#for a in range(NUM_ACTIONS):
		plt.plot(ticks_graph, current_strat_graph[:,0], label='R Current', color='green')
		plt.plot(ticks_graph, current_strat_graph[:,1], label='P Current', color='purple')
		plt.plot(ticks_graph, current_strat_graph[:,2], label='S Current', color='navy')
		plt.plot(ticks_graph, average_strat_graph[:,0], label='R Average', color='lightgreen')
		plt.plot(ticks_graph, average_strat_graph[:,1], label='P Average', color='mediumorchid')
		plt.plot(ticks_graph, average_strat_graph[:,2], label='S Average', color='royalblue')
		plt.title('Regular regret matching')
		plt.xlabel('100 iteration increments')
		plt.ylabel('Rewards')
		plt.legend(loc='upper right')
		plt.subplot(122)
		#for a in range(NUM_ACTIONS):
		plt.plot(ticks_graph, current_strat_graph_noneg[:,0], label='R Current', color='green')
		plt.plot(ticks_graph, current_strat_graph_noneg[:,1], label='P Current', color='purple')
		plt.plot(ticks_graph, current_strat_graph_noneg[:,2], label='S Current', color='navy')
		plt.plot(ticks_graph, average_strat_graph_noneg[:,0], label='R Average', color='lightgreen')
		plt.plot(ticks_graph, average_strat_graph_noneg[:,1], label='P Average', color='mediumorchid')
		plt.plot(ticks_graph, average_strat_graph_noneg[:,2], label='S Average', color='royalblue')
		plt.title('Regret matching with negative regret resetting')
		plt.xlabel('100 iteration increments')
		plt.ylabel('Rewards')
		plt.legend(loc='upper right')
		plt.show()


if __name__ == "__main__":
	p1 = RPSCFRPlayer()
	p2 = RPSFixedPlayer()
	game = RPS(p1, p2)
	game.play_games(num_games = 100000)
