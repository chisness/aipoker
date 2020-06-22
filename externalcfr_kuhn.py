import numpy as np
import random
import matplotlib.pyplot as plt

#modifications:
# CFR+: regret sums cannot go negative
#get expected utility of each action
#importance also based on how often opponent plays to this point

class Node:
	def __init__(self, num_actions):
		self.regret_sum = np.zeros(num_actions)
		self.util_sum = np.zeros(num_actions)
		self.util_counter = np.zeros(num_actions)
		self.strategy = np.zeros(num_actions)
		self.strategy_sum = np.zeros(num_actions)
		self.num_actions = num_actions
		self.strategy_curr = []
		self.strategy_avg = []
		self.counter = 0

	def get_strategy(self):
		normalizing_sum = 0
		for a in range(self.num_actions):
			if self.regret_sum[a] > 0:
				self.strategy[a] = self.regret_sum[a]
			else:
				self.strategy[a] = 0
			normalizing_sum += self.strategy[a]

		for a in range(self.num_actions):
			if normalizing_sum > 0:
				self.strategy[a] /= normalizing_sum
			else:
				self.strategy[a] = 1.0/self.num_actions

		return self.strategy

	def get_average_strategy(self):
		avg_strategy = np.zeros(self.num_actions)
		normalizing_sum = 0
		
		for a in range(self.num_actions):
			normalizing_sum += self.strategy_sum[a]
		for a in range(self.num_actions):
			if normalizing_sum > 0:
				avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
			else:
				avg_strategy[a] = 1.0 / self.num_actions
		
		return avg_strategy

class KuhnCFR:
	def __init__(self, iterations, decksize):
		self.nbets = 2
		self.iterations = iterations
		self.decksize = decksize
		self.cards = np.arange(decksize)
		self.bet_options = 2
		self.nodes = {}

	def cfr_iterations_external(self, k = 10):
		util = np.zeros(2)
		for t in range(1, self.iterations + 1): 
			for i in range(2):
				# for k2 in range(k):
				random.shuffle(self.cards)
				util[i] += self.external_cfr(self.cards[:2], [], 2, 0, i, t)
		# print('Average game value: {}'.format(util[0]/(self.iterations * k)))
		util_diffs = {}
		count_nodes = 0
		for i in sorted(self.nodes):
			#print(i)
		# 	infoset = str(i[0]) + str(i[1:])
		# 	print('card[history]', '[strategy]')
			util_avg1 = self.nodes[i].util_sum/self.nodes[i].util_counter
			print(i, self.nodes[i].get_average_strategy(), util_avg1)
			util_max = -10000
			util_min = 10000
			for a in range(self.bet_options):
				if util_avg1[a] > util_max:
					util_max = util_avg1[a]
				if util_avg1[a] < util_min:
					util_min = util_avg1[a]
			util_diffs[count_nodes] = util_max - util_min 
			count_nodes += 1
		# 	for a in range(self.bet_options):
		# 		print(f'regret_sum for action {a}: {self.nodes[infoset].regret_sum[a]}')
		print('\n\n\n')
		#while True:
		print('Selecting poker scenario...')
		#print('len', len(self.nodes))
		print(util_diffs)
		util_diffs = {k: v for k, v in sorted(util_diffs.items(), key=lambda item: item[1])} #sorted
		#print(util_diffs)
		#n = random.choice(sorted(self.nodes)) #Random, but we want to do this based on importance
		#for i in sorted(util_diffs):
		util_array = []
		count_wl = 0
		count_max = len(util_diffs)
		for key in util_diffs:
			util_array.append(key)
		#while count_wl < count_max: 
			#print(util_diffs)
			#print(len(util_diffs))
			# for i in reversed(util_array):
			# 	print(i)
			# print(util_array)
		for node_index in reversed(util_array):
		#for node index in reversed(range())
			#print(node_index)
			n = sorted(self.nodes)[node_index]
			print('Your cards and action history: ', n)
			user_estimates = np.zeros(self.bet_options)
			for i in range(self.bet_options):
				print(f'What percent would you take action {i}?')
				user_estimates[i] = input()
			utils = {}
			for i in range(self.bet_options):
				print(f'Your play for action {i}: {user_estimates[i]}')
				utils[a] = self.nodes[n].util_sum/self.nodes[n].util_counter
				print(f'Game theory optimal play for action {i}: {self.nodes[n].get_average_strategy()[i]} has expected utility {utils[a]}')
			fig, ax1 = plt.subplots()
			#ax1 = fig.add_subplot(221)
			l = range(self.nodes[n].counter)
			#print('l', l)
			#print('current strategy', self.nodes[n].strategy_curr)
			#print('avg strategy', self.nodes[n].strategy_avg)
			self.nodes[n].strategy_curr = np.array(self.nodes[n].strategy_curr)
			self.nodes[n].strategy_avg = np.array(self.nodes[n].strategy_avg)
			#print(self.nodes[n].strategy_curr[:,0])
			#print(self.nodes[n].strategy_curr[:,1])
			#print(self.nodes[n].strategy_avg[:,0])
			#print(self.nodes[n].strategy_avg[:,1])
			ax1.plot(l, self.nodes[n].strategy_curr[:,0], label='Pass Current', color='green')
			ax1.plot(l, self.nodes[n].strategy_curr[:,1], label='Bet Current', color='purple')
			ax1.plot(l, self.nodes[n].strategy_avg[:,0], label='Pass Average', color='lightgreen')
			ax1.plot(l, self.nodes[n].strategy_avg[:,1], label='Bet Average', color='mediumorchid')
			ax1.set_title('Strategies')
			ax1.set_xlabel('Times Node Touched')
			ax1.set_ylabel('Strategy percentage')
			ax1.legend(loc='lower right')
			plt.show()
			print('')
			print('Press q to quit or any key to continue: ')
			q_option = input()
			if str(q_option) == 'q':
				break
			else:
				count_wl += 1

	def external_cfr(self, cards, history, pot, nodes_touched, traversing_player, t):
		print('THIS IS ITERATION', t)
		plays = len(history)
		acting_player = plays % 2
		opponent_player = 1 - acting_player
		if plays >= 2:
			if history[-1] == 0 and history[-2] == 1: #bet fold
				if acting_player == traversing_player:
					return 1
				else:
					return -1
			if (history[-1] == 0 and history[-2] == 0) or (history[-1] == 1 and history[-2] == 1): #check check or bet call, go to showdown
				if acting_player == traversing_player:
					if cards[acting_player] > cards[opponent_player]:
						return pot/2 #profit
					else:
						return -pot/2
				else:
					if cards[acting_player] > cards[opponent_player]:
						return -pot/2
					else:
						return pot/2

		#infoset = cards[acting_player], history
		infoset = str(cards[acting_player]) + str(history)
		if infoset not in self.nodes:
			self.nodes[infoset] = Node(self.bet_options)

		self.nodes[infoset].counter += 1
		self.nodes[infoset].strategy_curr.append(np.array(self.nodes[infoset].get_strategy()[:]))
		self.nodes[infoset].strategy_avg.append(self.nodes[infoset].get_average_strategy()[:])

		# bets = -torch.ones(self.nbets)
		# for (i, b) in enumerate(history):
		# 	# bets should be a list of the proportion each bet is of the pot size
		# 	bets[i] = b / (sum(history[:i]) + 2)

		nodes_touched += 1

		if acting_player == traversing_player:
			util = np.zeros(self.bet_options) #2 actions
			node_util = 0
			#advantages = self.val_nets[acting_player].forward(torch.tensor([[cards[acting_player]]], dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
			strategy = self.nodes[infoset].get_strategy()
			for a in range(self.bet_options):
				next_history = history + [a]
				pot += a
				util[a] = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
				node_util += strategy[a] * util[a]

			action_advantages = np.zeros(self.bet_options)
			stone = np.zeros(self.bet_options)
			for a in range(self.bet_options):
				regret = util[a] - node_util
				self.nodes[infoset].regret_sum[a] += regret
				stone[a] = regret
				self.nodes[infoset].util_sum[a] += util[a]
				self.nodes[infoset].util_counter[a] += 1
			print('****************')
			print('CARDS: ', cards[acting_player])
			print('BETS: ', history)
			print('SUM ADVANTAGES (REGRETS): ', self.nodes[infoset].regret_sum)
			print('INSTANT ADVANTAGES (REGRETS): ', stone)
			print('SUM UTIL: ', self.nodes[infoset].util_sum)
			print('INSTANT UTIL: ', util)
			#self.m_v[traversing_player].append((infoset, t, action_advantages))
			return node_util

		else: #acting_player != traversing_player
			#advantages = self.val_nets[acting_player].forward(torch.tensor([[cards[acting_player]]],  dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
			strategy = self.nodes[infoset].get_strategy()
			#self.m_pi.append((infoset, t, strategy, acting_player))
			util = 0
			if random.random() < strategy[0]:
				next_history = history + [0]
			else: 
				next_history = history + [1]
				pot += 1
			util = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
			for a in range(self.bet_options):
				self.nodes[infoset].strategy_sum[a] += strategy[a]
				#self.nodes[infoset].strategy_hist[a].append(strategy[a])
			return util

if __name__ == "__main__":
	k = KuhnCFR(10000, 3)
	k.cfr_iterations_external()