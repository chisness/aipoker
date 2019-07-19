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
		self.winnings_noneg = 0

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
				avg_strategy[a] = 1.0 / NUM_ACTIONS
		
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
				avg_strategy[a] = 1.0 / NUM_ACTIONS
		
		return avg_strategy

class BRFPlayer:
	def __init__(self, cfr_player):
		self.strategy1 = [0.36, 0.32, 0.32]
		self.strategy2 = [0.32, 0.32, 0.36]
		self.cfr_player = cfr_player
		self.strategy_sum = np.zeros(NUM_ACTIONS)
		#self.strategy = [1/3, 1/3, 1/3]
		self.winnings = 0

	def get_strategy(self):
		opponent_strategy = cfr_player.get_
		if iteration < 5000:
			self.strategy_sum += self.strategy1
			return self.strategy1
		else:
			self.strategy_sum += self.strategy2
			return self.strategy2

	def get_average_strategy(self):
		avg_strategy = np.zeros(NUM_ACTIONS)
		normalizing_sum = 0
		
		for a in range(NUM_ACTIONS):
			normalizing_sum += self.strategy_sum[a]
		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
			else:
				avg_strategy[a] = 1.0 / NUM_ACTIONS
		
		return avg_strategy

class RPSSwitchPlayer:
	def __init__(self):
		self.strategy1 = [0.36, 0.32, 0.32]
		self.strategy2 = [0.32, 0.32, 0.36]
		self.strategy_sum = np.zeros(NUM_ACTIONS)
		#self.strategy = [1/3, 1/3, 1/3]
		self.winnings = 0

	def get_strategy(self, iteration):
		if iteration < 5000:
			self.strategy_sum += self.strategy1
			return self.strategy1
		else:
			self.strategy_sum += self.strategy2
			return self.strategy2

	def get_average_strategy(self):
		avg_strategy = np.zeros(NUM_ACTIONS)
		normalizing_sum = 0
		
		for a in range(NUM_ACTIONS):
			normalizing_sum += self.strategy_sum[a]
		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
			else:
				avg_strategy[a] = 1.0 / NUM_ACTIONS
		
		return avg_strategy

class RPSFixedPlayer:
	def __init__(self):
		#self.strategy = [0.9, 0.05, 0.05]
		#self.strategy = [0.4, 0.3, 0.3]
		self.strategy = [1/3, 1/3, 1/3]
		self.winnings = 0

	def get_strategy(self):
		return self.strategy

	def get_average_strategy(self):
		return self.strategy

class RPSFixedPlayer2:
	def __init__(self):
		#self.strategy = [0.9, 0.05, 0.05]
		#self.strategy = [0, 1, 0]
		self.strategy = [1/3, 1/3, 1/3]
		self.winnings = 0

	def get_strategy(self):
		return self.strategy

	def get_average_strategy(self):
		return self.strategy

class Human:
	def __init__(self):
		self.winnings = 0
		self.strategy_sum = np.zeros(NUM_ACTIONS)

	def get_strategy(self):
		move = int(input('Enter your play (ROCK 0, PAPER 1, SCISSORS 2, QUIT -1): '))
		if move in [-1, 0, 1, 2]:
			if move == -1:
				return -1
			elif move == 0:
				self.strategy_sum[move] += 1
				return [1, 0, 0]
			elif move == 1:
				self.strategy_sum[move] += 1
				return [0, 1, 0]
			elif move == 2:
				self.strategy_sum[move] += 1
				return [0, 0, 1]
		else:
			print('Invalid action, try again')
			self.get_strategy()

	def get_average_strategy(self):
		avg_strategy = np.zeros(NUM_ACTIONS)
		normalizing_sum = 0
		
		for a in range(NUM_ACTIONS):
			normalizing_sum += self.strategy_sum[a]
		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
			else:
				avg_strategy[a] = 1.0 / NUM_ACTIONS
		
		return avg_strategy

class Yoni:
	def __init__(self):
		self.winnings = 0
		self.strategy_sum = np.zeros(NUM_ACTIONS)

	def get_strategy(self, opp_2_moves_ago):
		r = random.random()
		if opp_2_moves_ago == 0:
			if r < 0.5:
				return [0, 1, 0]
				strategy_sum[1] += 1
			else:
				return [0, 0, 1]
				strategy_sum[2] += 1

		if opp_2_moves_ago == 1:
			if r < 0.5:
				return [1, 0, 0]
				strategy_sum[0] += 1
			else: 
				return [0, 0, 1]
				strategy_sum[2] += 1

		if opp_2_moves_ago == 2:
			if r < 0.5:
				return [1, 0, 0]
				strategy_sum[0] += 1
			else:
				return [0, 1, 0]
				strategy_sum[1] += 1

	def get_average_strategy(self):
		avg_strategy = np.zeros(NUM_ACTIONS)
		normalizing_sum = 0
		
		for a in range(NUM_ACTIONS):
			normalizing_sum += self.strategy_sum[a]
		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
			else:
				avg_strategy[a] = 1.0 / NUM_ACTIONS
		
		return avg_strategy

class Yoni2: #same thing or thing that beats it
	def __init__(self):
		self.winnings = 0
		self.strategy_sum = np.zeros(NUM_ACTIONS)

	def get_strategy(self, opp_2_moves_ago):
		r = random.random()
		if opp_2_moves_ago == 0:
			if r < 0.5:
				return [0, 1, 0]
				strategy_sum[1] += 1
			else:
				return [1, 0, 0]
				strategy_sum[0] += 1

		if opp_2_moves_ago == 1:
			if r < 0.5:
				return [0, 1, 0]
				strategy_sum[1] += 1
			else: 
				return [0, 0, 1]
				strategy_sum[2] += 1

		if opp_2_moves_ago == 2:
			if r < 0.5:
				return [1, 0, 0]
				strategy_sum[0] += 1
			else:
				return [0, 0, 1]
				strategy_sum[2] += 1

	def get_average_strategy(self):
		avg_strategy = np.zeros(NUM_ACTIONS)
		normalizing_sum = 0
		
		for a in range(NUM_ACTIONS):
			normalizing_sum += self.strategy_sum[a]
		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
			else:
				avg_strategy[a] = 1.0 / NUM_ACTIONS
		
		return avg_strategy

class RPSExploitPlayer:
	def __init__(self):
		#win stick with same strategy, lose rotate between R/P/S
		#25% random, 75% {win same strategy, lose rotate between R/P/S}
		self.winnings = 0
		self.strategy_sum = np.zeros(NUM_ACTIONS)

	def get_strategy(self, last_move, last_move_win, epsilon = 0.25):
		if random.random() < epsilon:
			return [1/3, 1/3, 1/3]
			for a in NUM_ACTIONS:
				strategy_sum[a] += 1/3
		else:
			if last_move_win == -1: #for player 2 since -1 is win
				if last_move == 0:
					return [1, 0, 0]
					strategy_sum[0] += 1
				elif last_move == 1:
					return [0, 1, 0]
					strategy_sum[1] += 1
				elif last_move == 2:
					return [0, 0, 1]
					strategy_sum[2] += 1
			else:
				if last_move == 0:
					return [0, 1, 0]
					strategy_sum[1] += 1
				elif last_move == 1:
					return [0, 0, 1]
					strategy_sum[2] += 1
				elif last_move == 2:
					return [1, 0, 0]
					strategy_sum[0] += 1

	def get_average_strategy(self):
		avg_strategy = np.zeros(NUM_ACTIONS)
		normalizing_sum = 0
		
		for a in range(NUM_ACTIONS):
			normalizing_sum += self.strategy_sum[a]
		for a in range(NUM_ACTIONS):
			if normalizing_sum > 0:
				avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
			else:
				avg_strategy[a] = 1.0 / NUM_ACTIONS
		
		return avg_strategy


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
		current_strat_graph2 = []
		average_strat_graph2 = []
		current_strat_graph_noneg = []
		average_strat_graph_noneg = []
		ticks_graph = []
		avg_rewards = []
		rewards1 = []
		rewards2 = []
		counter = 0
		last_action_p2 = 0
		p2_move = random.randrange(3)
		result = 0
		last_moves = np.zeros(2)
		for i in range(num_games):
			p1_strategy = p1.get_strategy()
			p1_move = np.random.choice(ACTIONS, p=p1_strategy)
			#print('before', p1_strategy)

			last_moves = [last_moves[1], p1_move]

			# p1_strategy_noneg = p1.get_strategy_noneg()
			# p1_move_noneg = np.random.choice(ACTIONS, p=p1_strategy_noneg)

			#p2_strategy = p2.get_strategy()
			#p2_strategy = p2.get_strategy(p2_move, result)#, epsilon = 0.25)
			#p2_strategy = p2.get_strategy()#p2.get_strategy(counter)#last_moves[1])#(counter)#last_moves[1])
			p2_strategy = p2.get_strategy()#(counter)#last_moves[1])
			if p2_strategy == -1:
				print('Quit game')
				break
			else:
				p2_move = np.random.choice(ACTIONS, p=p2_strategy)

				result = self.check_result(p1_move, p2_move)
				#result_nn = self.check_result(p1_move_noneg, p2_move)
				print('Computer move is: {}'.format(p1_move))
				if result == 0:
					print('Tie game')
				elif result == 1:
					print('Computer wins')
				elif result == -1:
					print('Human player wins')
				p1.winnings += result
				#p1.winnings_noneg += result_nn
				p2.winnings -= result
				rewards1.append(p1.winnings)
				#rewards2.append(p1.winnings_noneg)
				print(p1.winnings)

				if counter % 1 == 0 and counter > -1:
					#print(p1.regret_sum)
					current_strat_graph.append(np.array(p1.get_strategy()[:]))
					average_strat_graph.append(p1.get_average_strategy()[:])
					#current_strat_graph_noneg.append(np.array(p1.get_strategy_noneg()[:]))
					#average_strat_graph_noneg.append(p1.get_average_strategy_noneg()[:])
					current_strat_graph2.append(np.array(p1.get_strategy()[:]))
					average_strat_graph2.append(p2.get_average_strategy()[:])
					#avg_rewards.append(p1.winnings/counter)
					ticks_graph.append(counter)

				#result_noneg = self.check_result(p1_move_noneg, p2_move)

				for a in range(NUM_ACTIONS):
						utility[a] = self.check_result(a, p2_move)
						#print('util', utility[a])

				for a in range(NUM_ACTIONS):
					#print(utility[a] - utility[p1_move])
					p1.regret_sum[a] += (utility[a] - utility[p1_move])
					#p1.regret_sum_noneg[a] += (utility[a] - utility[p1_move_noneg])
					#p2.regret_sum[a] += (utility[a] - utility[p2_move])
					#regret is other move minus our move, i.e., alternative compared to move we took

				# print('p1 strategy', p1.get_strategy())
				# print('p1 average strategy', p1.get_average_strategy())
				# print('p1 regrets', p1.regret_sum)
				# print('p1 winnings', p1.winnings)
				counter += 1

		#print(current_strat_graph)
		#print(average_strat_graph)
		# print(ticks_graph)
		# print(current_strat_graph)
		current_strat_graph = np.array(current_strat_graph)
		average_strat_graph = np.array(average_strat_graph)
		average_strat_graph2 = np.array(average_strat_graph2)
		#current_strat_graph_noneg = np.array(current_strat_graph_noneg)
		#average_strat_graph_noneg = np.array(average_strat_graph_noneg)
		rewards1 = np.array(rewards1)
		rewards2 = np.array(rewards2)
		#avg_rewards1 = avg_rewards1 / counter
		#avg_rewards2 = avg_rewards2 / counter

		print(rewards1)

		fig = plt.figure()
		ax1 = fig.add_subplot(221)
		ax2 = ax1.twinx()
		#plt.figure()
		#plt.subplot(131) #1 row, 2 columns, 1 index
		#for a in range(NUM_ACTIONS):
		ax1.plot(ticks_graph, current_strat_graph[:,0], label='R Current P1', color='green')
		ax1.plot(ticks_graph, current_strat_graph[:,1], label='P Current P1', color='purple')
		ax1.plot(ticks_graph, current_strat_graph[:,2], label='S Current P1', color='navy')
		ax1.plot(ticks_graph, average_strat_graph[:,0], label='R Average P1', color='lightgreen')
		ax1.plot(ticks_graph, average_strat_graph[:,1], label='P Average P1', color='mediumorchid')
		ax1.plot(ticks_graph, average_strat_graph[:,2], label='S Average P1', color='royalblue')
		ax2.plot(ticks_graph, rewards1, label = 'Rewards', color='black')
		ax1.set_title('Regret matching')
		ax1.set_xlabel('Games')
		ax1.set_ylabel('Strategy percentage')
		ax2.set_ylabel('Rewards')
		ax1.legend(loc='upper right')
		ax2.legend(loc='lower right')
		# ax1 = fig.add_subplot(222)
		# ax2 = ax1.twinx()
		# ax1.plot(ticks_graph, current_strat_graph_noneg[:,0], label='R Current P1', color='green')
		# ax1.plot(ticks_graph, current_strat_graph_noneg[:,1], label='P Current P1', color='purple')
		# ax1.plot(ticks_graph, current_strat_graph_noneg[:,2], label='S Current P1', color='navy')
		# ax1.plot(ticks_graph, average_strat_graph_noneg[:,0], label='R Average P1', color='lightgreen')
		# ax1.plot(ticks_graph, average_strat_graph_noneg[:,1], label='P Average P1', color='mediumorchid')
		# ax1.plot(ticks_graph, average_strat_graph_noneg[:,2], label='S Average P1', color='royalblue')
		# ax2.plot(ticks_graph, rewards2, label = 'Rewards', color='black')
		# ax1.set_title('Regret matching with negative regret resetting')
		# ax1.set_xlabel('100 iteration increments')
		# ax1.set_ylabel('Strategy percentage')
		# ax2.set_ylabel('Rewards')
		# ax1.legend(loc='upper right')
		# ax2.legend(loc='lower right')
		plt.subplot(223)
		plt.plot(ticks_graph, average_strat_graph[:,0], label='R Average P1', color='green')
		plt.plot(ticks_graph, average_strat_graph[:,1], label='P Average P1', color='purple')
		plt.plot(ticks_graph, average_strat_graph[:,2], label='S Average P1', color='navy')
		plt.plot(ticks_graph, average_strat_graph2[:,0], label='R Average P2', color='lightgreen')
		plt.plot(ticks_graph, average_strat_graph2[:,1], label='P Average P2', color='mediumorchid')
		plt.plot(ticks_graph, average_strat_graph2[:,2], label='S Average P2', color='royalblue')
		plt.title('Regret matching P1 vs P2 average strategies')
		plt.xlabel('Games')
		plt.ylabel('Strategy percentage')
		plt.legend(loc='upper right')
		plt.show()


if __name__ == "__main__":
	p1 = RPSCFRPlayer()
	#p2 = Human()
	#p2 = RPSSwitchPlayer()
	#p2 = RPSCFRPlayer()
	#p1 = RPSFixedPlayer2()
	p2 = RPSFixedPlayer()
	#p2 = RPSExploitPlayer()
	#p2 = Yoni()
	game = RPS(p1, p2)
	game.play_games(num_games = 10000)

	#vs opponent
	#fixed opponent changes strategy
