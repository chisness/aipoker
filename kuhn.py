import numpy as np
import random
from itertools import permutations
import matplotlib.pyplot as plt
import collections 
import torch
import torch.nn as nn
import torch.nn.functional as F

# class Poker:
# 	def __init__(self, dealer_player = 1, players = 2):
# 		self.dealer_player = dealer_player
# 		self.pot = players * ante * game.ante
# 		if self.dealer_player == 1:
# 			self.player1cards = self.cards[1]
# 			self.player2cards = self.cards[0]
# 		else:
# 			self.player1cards = self.cards[0]
# 			self.player2cards = self.cards[1]
# 		if self.game == 'l':
# 			self.community_card = cards[2]
# 		print('Dealer player: ', self.dealer_player)
# 		print('Player 1 cards: ', self.player1cards)
# 		print('Player 2 cards: ', self.player2cards)

#class LimitLeduc(Poker):

class Kuhn:
	def __init__(self, players, dealer_player = 0, ante = 1, shuffled_deck = [1,2,3], buckets = 0):
		self.dealer_player = dealer_player
		self.num_players = len(players)
		self.pot = self.num_players * ante
		self.history = []
		self.cards = shuffled_deck
		self.betsize = 1
		self.buckets = 0
		if self.num_players == 2:
			self.player0cards = self.cards[1 - dealer_player]
			self.player1cards = self.cards[dealer_player]
		self.players = []
		for i in range(len(players)):
			self.players.append(players[i])
		print('Player 0 card: {}'.format(self.player0cards))
		print('Player 1 card: {}'.format(self.player1cards))

	def game(self):
		print('Action history: {}'.format(self.history))
		plays = len(self.history)
		acting_player = (plays + self.dealer_player)%2
		if len(self.history) >= 1:
			if self.history[-1] == 'f': #folded
				print('folded)')
				profit = (self.pot - self.betsize) / 2
				self.players[acting_player].profit += profit
				self.players[1-acting_player].profit -= profit
				print('End of game! Player {} wins pot of {} (profits of {})\n'.format(self.acting_player, self.pot, profit))
				return profit

		if len(self.history) >= 2:
			if self.history[-2] == self.history[-1]: #check check or bet call, go to showdown
				profit = self.pot/2
				winner = self.evaluate_hands()
				self.players[winner].profit += profit
				self.players[1-winner].profit -= profit
				print('End of game! Player {} wins pot of {} (profits of {})\n'.format(winner, self.pot, profit))
				if winner == acting_player:
					return profit
				else:
					return -profit

		#still going in round
		print('\nPot size: {}'.format(self.pot))
		print('Player {} turn to act'.format(acting_player))
		bet = self.players[acting_player].select_move(self.valid_bets())
		if bet != 'f':
			self.pot += bet
		self.history.append(bet)
		print('Action: {}'.format(bet))
		self.game() #continue


	def evaluate_hands(self):
		if self.player0cards > self.player1cards:
			return 0
		else:
			return 1

	def valid_bets(self):
		if self.history == []:
			return [0, self.betsize] #check or bet
		elif self.history[-1] == self.betsize:
			return ['f', self.betsize] #fold or call
		elif self.history[-1] == 0:
			return [0, self.betsize] #check or bet



class Node:
	def __init__(self, num_actions):
		self.regret_sum = np.zeros(num_actions)
		self.strategy_br = np.zeros(num_actions)
		self.strategy = np.zeros(num_actions)
		self.strategy_sum = np.zeros(num_actions)
		self.num_actions = num_actions

	def get_strategy(self, realization_weight = False):
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
			if realization_weight:
				self.strategy_sum[a] += realization_weight * self.strategy[a]

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

class DeepCFRNet(nn.Module):
	def __init__(self, ncardtypes, nbets, nactions, dim = 128):
		self.card1 = nn.Linear(dim * ncardtypes, dim)
		self.card2 = nn.Linear(dim, dim)
		self.card3 = nn.Linear(dim, dim)

		self.bet1 = nn.Linear(nbets * 2, dim)
		self.bet2 = nn.Linear(dim, dim)

		self.comb1 = nn.Linear(2 * dim, dim)
		self.comb2 = nn.Linear(dim, dim)
		self.comb3 = nn.Linear(dim, dim)

		self.action_head = nn.Linear(dim, nactions)

	def forward(self, cards, bets):
		#cards leduc is 1 preflop, 1 flop
		#bets 

		x = F.relu(self.card1(cards))
		x = F.relu(self.card2(x))
		x = F.relu(self.card3(x))

		bet_occurred = bets.ge(0) #bets that are >= 0
		bet_feats = torch.cat([bet_size, bet_occurred.float()], dim=1)
		y = F.relu(self.bet1(bet_feats))
		y = F.relu(self.bet2(y) + y)

		z = torch.cat([x, y], dim = 1)
		z = F.relu(self.comb1(z))
		z = F.relu(self.comb2(z) + z)
		z = F.relu(self.comb3(z) + z)

		z = normalize(z)
		return self.action_head(z)


class KuhnCFR:
	def __init__(self, iterations, decksize, buckets):
		self.iterations = iterations
		self.decksize = decksize
		self.cards = np.arange(decksize)
		self.nodes = {}
		self.bet_options = 2
		self.buckets = buckets
		self.counter = 0
		self.exploit = collections.defaultdict(float)
		self.theta_0 = 0
		self.theta_1 = 0
		self.m_v0 = []
		self.m_v1 = []
		self.m_pi = []

	def cfr_iterations_chance(self):
		util = 0
		opens = []
		opens_st = []
		kb = []
		kb_st = []
		b = []
		b_st = []
		k = []
		k_st = []

		for i in range(self.iterations):
			random.shuffle(self.cards)
			util += self.do_cfr(self.cards[:2], '', 1, 1, 2, 0)
		print('Average game value: {}'.format(util/self.iterations))
		for i in sorted(self.nodes):
			avg_st = self.nodes[i].get_average_strategy()
			print(i, avg_st)

			if len(i) == 1:
				opens.append(i)
				opens_st.append(avg_st[0])
			if len(i) == 3:
				kb.append(i)
				kb_st.append(avg_st[0])
			if i[1:] == 'b':
				b.append(i)
				b_st.append(avg_st[0])
			if i[1:] == 'k':
				k.append(i)
				k_st.append(avg_st[0])

		opens_st = np.array(opens_st)
		kb_st = np.array(kb_st)
		b_st = np.array(b_st)
		k_st = np.array(k_st)

		ind = np.arange(len(opens))

		#plt.figure()
		plt.subplot(221)
		plt.bar(ind, 1 - opens_st, label = 'Bet')
		plt.bar(ind, opens_st, bottom=(1-opens_st), label = 'Pass')
		plt.xticks(ind, opens)
		plt.legend()
		plt.xlabel('Information set')
		plt.ylabel('Strategy percent')
		plt.title('Kuhn Strategy for Player 1 starting action with {} iterations'.format(self.iterations))

		plt.subplot(224)
		plt.bar(ind, 1 - kb_st, label = 'Bet')
		plt.bar(ind, kb_st, bottom=(1-kb_st), label = 'Pass')
		plt.xticks(ind, kb)
		plt.legend()
		plt.xlabel('Information set')
		plt.ylabel('Strategy percent')
		plt.title('Kuhn Strategy for Player 1 after check/bet action with {} iterations'.format(self.iterations))

		plt.subplot(223)
		plt.bar(ind, 1 - b_st, label = 'Bet')
		plt.bar(ind, b_st, bottom=(1-b_st), label = 'Pass')
		plt.xticks(ind, b)
		plt.legend()
		plt.xlabel('Information set')
		plt.ylabel('Strategy percent')
		plt.title('Kuhn Strategy for Player 2 after bet action with {} iterations'.format(self.iterations))

		plt.subplot(222)
		plt.bar(ind, 1 - k_st, label = 'Bet')
		plt.bar(ind, k_st, bottom=(1-k_st), label = 'Pass')
		plt.xticks(ind, k)
		plt.legend()
		plt.xlabel('Information set')
		plt.ylabel('Strategy percent')
		plt.title('Kuhn Strategy for Player 2 after check with {} iterations'.format(self.iterations))
		plt.show()

		expl_i = []
		expl = []
		for i in self.exploit:
			expl_i.append(i)
			expl.append(self.exploit[i])
		plt.plot(expl_i, expl)
		plt.show()

	def cfr_iterations_chance_fake(self):
		util = 0
		for i in range(self.iterations):
			util += self.do_cfr([1,2], '', 1, 1, 2, 1)
		print('Average game value: {}'.format(util/self.iterations))
		for i in sorted(self.nodes):
			print(i, self.nodes[i].get_average_strategy())


	def cfr_iterations_vanilla(self):
		util = 0
		for i in range(self.iterations):
			for perm_card in list(permutations(self.cards)):
				util += self.do_cfr(perm_card[:2], '', 1, 1, 2, 0)
		print('Average game value: {}'.format(util/self.iterations))
		for i in sorted(self.nodes):
			print(i, self.nodes[i].get_average_strategy())

	def cfr_iterations_external(self):
		util = np.zeros(2)
		for i in range(self.iterations): #start with the chance node sampling
			random.shuffle(self.cards)
			for i in range(2):
				util[i] += self.external_cfr(self.cards[:2], '', 2, 0, i)
		print('Average game value: {}'.format(util[0]/self.iterations))
		for i in sorted(self.nodes):
			print(i, self.nodes[i].get_average_strategy())

	def cfr_iterations_deep(self):
		util = np.zeros(2)
		for t in range(self.iterations): #start with the chance node sampling
			random.shuffle(self.cards)
			for i in range(2):
				for k in range(10):
					if i == 0:
						util[i] += self.deep_cfr(self.cards[:2], '', 2, 0, i, self.theta_0, self.theta_1, self.m_v0, self.m_pi, t)
					else: 
						util[i] += self.deep_cfr(self.cards[:2], '', 2, 0, i, self.theta_0, self.theta_1, self.m_v1, self.m_pi, t)
				#train theta_p from scratch on loss
				#iterate over every infoset and every action
		#train theta_pi on loss
		#return theta_pi
		print('Average game value: {}'.format(util[0]/self.iterations))
		# for i in sorted(self.nodes):
		# 	print(i, self.nodes[i].get_average_strategy())

	def brf(self, player_card, history, player_iteration, opp_reach, buckets):
		plays = len(history)
		acting_player = plays % 2
		expected_payoff = 0

		if plays >= 2: #can be terminal
			opponent_dist = np.zeros(len(opp_reach))
			opponent_dist_total = 0
			#print('opp reach', opp_reach)
			if history[-1] == 'f' or history[-1] == 'c' or (history[-1] == history[-2] == 'k'):
				for i in range(len(opp_reach)):
					opponent_dist_total += opp_reach[i] #compute sum of dist. for normalizing
				for i in range(len(opp_reach)):
					opponent_dist[i] = opp_reach[i] / opponent_dist_total
					payoff = 0
					is_player_card_higher = player_card > i
					if history[-1] == 'f': #bet fold
						if acting_player == player_iteration:
							payoff = 1
						else:
							payoff = -1
					elif history[-1] == 'c': #bet call
						if is_player_card_higher:
							payoff = 2
						else:
							payoff = -2
					elif (history[-1] == history[-2] == 'k'): #check check
						if is_player_card_higher:
							payoff = 1
						else:
							payoff = -1
					expected_payoff += opponent_dist[i] * payoff
				return expected_payoff

		d = np.zeros(2) #opponent action distribution
		d = [0, 0]

		new_opp_reach = np.zeros(len(opp_reach))
		for i in range(len(opp_reach)):
			new_opp_reach[i] = opp_reach[i]

		v = -100000
		util = np.zeros(2)
		util = [0, 0]
		w = np.zeros(2)
		w = [0, 0]

		#infoset = history

		for a in range(2):
			if acting_player != player_iteration:
				for i in range(len(opp_reach)):
					if buckets > 0:
						bucket1 = 0
						for j in range(buckets):
							if j < len(opp_reach)/buckets * (j + 1):
								bucket1 = j
								break
						infoset = str(bucket1) + history
					else:
						infoset = str(i) + history 
					if infoset not in self.nodes:
						self.nodes[infoset] = Node(2)
					strategy = self.nodes[infoset].get_average_strategy()#get_strategy_br()
					new_opp_reach[i] = opp_reach[i] * strategy[a] #update reach prob
					w[a] += new_opp_reach[i] #sum weights over all poss. of new reach

			if a == 0:
				if len(history) != 0:
					if history[-1] == 'b':
						next_history = history + 'f'
					elif history[-1] == 'k':
						next_history = history + 'k'
				else:
					next_history = history + 'k'
			elif a == 1:
				if len(history) != 0:
					if history[-1] == 'b':
						next_history = history + 'c'
					elif history[-1] == 'k':
						next_history = history + 'b'
				else:
					next_history = history + 'b'
			#print('w', w)
			#print('history', history)
			#print('next history', next_history)
			util[a] = self.brf(player_card, next_history, player_iteration, new_opp_reach, buckets)
			#print('util a', util[a])
			if (acting_player == player_iteration and util[a] > v):
				v = util[a] #this action better than previously best action
		
		if acting_player != player_iteration:
			#D_(-i) = Normalize(w) , d is action distribution that = normalized w
			d[0] = w[0] / (w[0] + w[1])
			d[1] = w[1] / (w[0] + w[1])
			v = d[0] * util[0] + d[1] * util[1]

		return v

	def do_cfr(self, cards, history, p0, p1, pot, nodes_touched):
		plays = len(history)
		acting_player = plays % 2
		opponent_player = 1 - acting_player
		self.counter += 1

		if plays >= 2:
			if history[-1] == 'f': #bet fold
				return 1
			if (history[-1] == history[-2] == 'k') or (history[-1] == 'c'): #check check or bet call, go to showdown
				if cards[acting_player] > cards[opponent_player]:
					return pot/2 #profit
				else:
					return -pot/2

		#print('infoset cards: {}'.format(cards[acting_player]))
		#print('infoset history: {}'.format(history))
		num_actions = 2

		
		if self.buckets > 0:
			bucket = int(cards[acting_player] * self.buckets/self.decksize)
			infoset = str(bucket) + history

		else:
			infoset = str(cards[acting_player]) + history

		if infoset not in self.nodes:
			self.nodes[infoset] = Node(num_actions)

		print('==INFOSET==', infoset)
		print('REGRET_SUM FOR ACTION P', self.nodes[infoset].regret_sum[0])
		print('REGRET_SUM FOR ACTION B', self.nodes[infoset].regret_sum[1])
		print('STRATEGY SUM FOR ACTION P', self.nodes[infoset].strategy_sum[0])
		print('STRATEGY SUM FOR ACTION B', self.nodes[infoset].strategy_sum[1])

		nodes_touched += 1

		if self.counter % 1000 == 0:
			br = np.zeros(2)
			opp_reach = np.zeros(self.decksize)
			for p in [0, 1]:
				for c1 in range(self.decksize):
					for c2 in range(self.decksize):
						if c1 == c2:
							opp_reach[c2] = 0
						else:
							opp_reach[c2] = 1.0/(self.decksize - 1.0)
					br[p] += self.brf(c1, '', p, opp_reach, self.buckets)
			print('Iteration number: ', self.counter)
			print('Best response player 0: ', br[0])
			print('Best response player 1: ', br[1])
			print('Exploitability: ', (br[0] + br[1]) / 2)
			self.exploit[nodes_touched] = (br[0] + br[1]) / 2

		if acting_player == 0:
			realization_weight = p0
		else:
			realization_weight = p1
		print('realization weight p0', p0)
		print('realization weight p1', p1)
		strategy = self.nodes[infoset].get_strategy(realization_weight)
		print('strategy', strategy)
		util = np.zeros(self.bet_options) #2 actions
		node_util = 0

		for a in range(num_actions):
			#print('a is: {}'.format(a))
			if a == 0:
				if len(history) != 0:
					if history[-1] == 'b':
						next_history = history + 'f'
					elif history[-1] == 'k':
						next_history = history + 'k'
				else:
					next_history = history + 'k'
			elif a == 1:
				if len(history) != 0:
					if history[-1] == 'b':
						next_history = history + 'c'
					elif history[-1] == 'k':
						next_history = history + 'b'
				else:
					next_history = history + 'b'
				pot += 1
			if acting_player == 0:
				util[a] = -self.do_cfr(cards, next_history, p0*strategy[a], p1, pot, nodes_touched)
			elif acting_player == 1: 
				 util[a] = -self.do_cfr(cards, next_history, p0, p1*strategy[a], pot, nodes_touched)
			print('util of {}'.format(a), util[a])
			node_util += strategy[a] * util[a]

		print('node util', node_util)
		for a in range(2):
			regret = util[a] - node_util
			print('regret of {}'.format(a), regret)
			if acting_player == 0:
				self.nodes[infoset].regret_sum[a] += p1 * regret
				print('regret sum increase for action {} for player 0 += p1 * regret'.format(a), p1 * regret)
			elif acting_player == 1:
				self.nodes[infoset].regret_sum[a] += p0 * regret
				print('regret sum increase for action {} for player 1 += p0 * regret'.format(a), p0 * regret)

		return node_util

	def external_cfr(self, cards, history, pot, nodes_touched, traversing_player):
		plays = len(history)
		acting_player = plays % 2
		opponent_player = 1 - acting_player
		self.counter += 1

		if plays >= 2:
			if history[-1] == 'f': #bet fold
				if acting_player == traversing_player:
					return 1
				else:
					return -1
			if (history[-1] == history[-2] == 'k') or (history[-1] == 'c'): #check check or bet call, go to showdown
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

		#print('infoset cards: {}'.format(cards[acting_player]))
		#print('infoset history: {}'.format(history))
		num_actions = 2

		
		if self.buckets > 0:
			bucket = int(cards[acting_player] * self.buckets/self.decksize)
			infoset = str(bucket) + history

		else:
			infoset = str(cards[acting_player]) + history

		if infoset not in self.nodes:
			self.nodes[infoset] = Node(num_actions)

		nodes_touched += 1

		if self.counter % 1000 == 0:
			br = np.zeros(2)
			opp_reach = np.zeros(self.decksize)
			for p in [0, 1]:
				for c1 in range(self.decksize):
					for c2 in range(self.decksize):
						if c1 == c2:
							opp_reach[c2] = 0
						else:
							opp_reach[c2] = 1.0/(self.decksize - 1.0)
					br[p] += self.brf(c1, '', p, opp_reach, self.buckets)
			print('Iteration number: ', self.counter)
			print('Best response player 0: ', br[0])
			print('Best response player 1: ', br[1])
			print('Exploitability: ', (br[0] + br[1]) / 2)
			self.exploit[nodes_touched] = (br[0] + br[1]) / 2


		strategy = self.nodes[infoset].get_strategy()

		if acting_player == traversing_player:
			util = np.zeros(self.bet_options) #2 actions
			node_util = 0

			for a in range(num_actions):
				#print('a is: {}'.format(a))
				if a == 0:
					if len(history) != 0:
						if history[-1] == 'b':
							next_history = history + 'f'
						elif history[-1] == 'k':
							next_history = history + 'k'
					else:
						next_history = history + 'k'
				elif a == 1:
					if len(history) != 0:
						if history[-1] == 'b':
							next_history = history + 'c'
						elif history[-1] == 'k':
							next_history = history + 'b'
					else:
						next_history = history + 'b'
					pot += 1
				util[a] = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player)
				node_util += strategy[a] * util[a]

			for a in range(num_actions):
				regret = util[a] - node_util
				self.nodes[infoset].regret_sum[a] += regret
			return node_util

		else: #acting_player != traversing_player
			util = 0
			if random.random() < strategy[0]:
				if len(history) != 0:
					if history[-1] == 'b':
						next_history = history + 'f'
					elif history[-1] == 'k':
						next_history = history + 'k'
				else:
					next_history = history + 'k'
			else: 
				if len(history) != 0:
					if history[-1] == 'b':
						next_history = history + 'c'
					elif history[-1] == 'k':
						next_history = history + 'b'
				else:
					next_history = history + 'b'
				pot += 1
			util = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player)
			for a in range(num_actions):
				self.nodes[infoset].strategy_sum[a] += strategy[a]
			return util

	def deep_cfr(self, cards, history, pot, nodes_touched, traversing_player, theta_0, theta_1, m_v, m_pi, t):
		plays = len(history)
		acting_player = plays % 2
		opponent_player = 1 - acting_player
		self.counter += 1

		if plays >= 2:
			if history[-1] == 'f': #bet fold
				if acting_player == traversing_player:
					return 1
				else:
					return -1
			if (history[-1] == history[-2] == 'k') or (history[-1] == 'c'): #check check or bet call, go to showdown
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

		#print('infoset cards: {}'.format(cards[acting_player]))
		#print('infoset history: {}'.format(history))
		num_actions = 2

		
		if self.buckets > 0:
			bucket = int(cards[acting_player] * self.buckets/self.decksize)
			infoset = str(bucket) + history

		else:
			infoset = str(cards[acting_player]) + history

		if infoset not in self.nodes:
			self.nodes[infoset] = Node(num_actions)

		nodes_touched += 1

		#strategy = self.nodes[infoset].get_strategy()
		#should come from predicted advantages using regret matching with network for the acting player

		if acting_player == traversing_player:
			util = np.zeros(self.bet_options) #2 actions
			node_util = 0

			for a in range(num_actions):
				#print('a is: {}'.format(a))
				if a == 0:
					if len(history) != 0:
						if history[-1] == 'b':
							next_history = history + 'f'
						elif history[-1] == 'k':
							next_history = history + 'k'
					else:
						next_history = history + 'k'
				elif a == 1:
					if len(history) != 0:
						if history[-1] == 'b':
							next_history = history + 'c'
						elif history[-1] == 'k':
							next_history = history + 'b'
					else:
						next_history = history + 'b'
					pot += 1
				util[a] = self.deep_cfr(cards, next_history, pot, nodes_touched, traversing_player, theta_0, theta_1, m_v, m_pi, t)
				node_util += strategy[a] * util[a]

			action_advantages = np.zeros(num_actions)
			for a in range(num_actions):
				action_advantages[a] = util[a] - node_util
			m_v.append([infoset, t, action_advantages])
			return node_util

		else: #acting_player != traversing_player
			action_probs = np.zeros(num_actions)
			for a in range(num_actions):
				#self.nodes[infoset].strategy_sum[a] += strategy[a]
				action_probs[a] = strategy[a]
				#insert infoset, t, strategies into strategy memory
			m_pi.append([infoset, t, action_probs])
			util = 0
			if random.random() < strategy[0]:
				if len(history) != 0:
					if history[-1] == 'b':
						next_history = history + 'f'
					elif history[-1] == 'k':
						next_history = history + 'k'
				else:
					next_history = history + 'k'
			else: 
				if len(history) != 0:
					if history[-1] == 'b':
						next_history = history + 'c'
					elif history[-1] == 'k':
						next_history = history + 'b'
				else:
					next_history = history + 'b'
				pot += 1
			return self.deep_cfr(cards, next_history, pot, nodes_touched, traversing_player, theta_0, theta_1, m_v, m_pi, t)


class Kuhn3CFR:
	def __init__(self, iterations, decksize):
		self.iterations = iterations
		self.decksize = decksize
		self.cards = np.arange(decksize)
		self.nodes = {}
		self.bet_options = 2
		self.counter = 0

	def cfr_iterations_external(self):
		util = np.zeros(3)
		for i in range(self.iterations): #start with the chance node sampling
			random.shuffle(self.cards)
			for i in range(3):
				util[i] += self.external_cfr(self.cards[:3], '', 3, 0, i)
		print('Average value 0: {}'.format(util[0]/self.iterations))
		print('Average value 1: {}'.format(util[1]/self.iterations))
		print('Average value 2: {}'.format(util[2]/self.iterations))
		for i in sorted(self.nodes):
			print(i, self.nodes[i].get_average_strategy())

	# def cfr_iterations_chance(self):
	# 	util = 0
	# 	opens = []
	# 	opens_st = []
	# 	kb = []
	# 	kb_st = []
	# 	b = []
	# 	b_st = []
	# 	k = []
	# 	k_st = []

	# 	for i in range(self.iterations):
	# 		random.shuffle(self.cards)
	# 		util += self.do_cfr(self.cards[:3], '', 1, 1, 1, 2, 0)
	# 	print('Average game value: {}'.format(util/self.iterations))
	# 	for i in sorted(self.nodes):
	# 		avg_st = self.nodes[i].get_average_strategy()
	# 		print(i, avg_st)

	# 		if len(i) == 1:
	# 			opens.append(i)
	# 			opens_st.append(avg_st[0])
	# 		if len(i) == 3:
	# 			kb.append(i)
	# 			kb_st.append(avg_st[0])
	# 		if i[1:] == 'b':
	# 			b.append(i)
	# 			b_st.append(avg_st[0])
	# 		if i[1:] == 'k':
	# 			k.append(i)
	# 			k_st.append(avg_st[0])

	# 	opens_st = np.array(opens_st)
	# 	kb_st = np.array(kb_st)
	# 	b_st = np.array(b_st)
	# 	k_st = np.array(k_st)

	# 	ind = np.arange(len(opens))

	# 	#plt.figure()
	# 	plt.subplot(221)
	# 	plt.bar(ind, 1 - opens_st, label = 'Bet')
	# 	plt.bar(ind, opens_st, bottom=(1-opens_st), label = 'Pass')
	# 	plt.xticks(ind, opens)
	# 	plt.legend()
	# 	plt.xlabel('Information set')
	# 	plt.ylabel('Strategy percent')
	# 	plt.title('Kuhn Strategy for Player 1 starting action with {} iterations'.format(self.iterations))

	# 	plt.subplot(224)
	# 	plt.bar(ind, 1 - kb_st, label = 'Bet')
	# 	plt.bar(ind, kb_st, bottom=(1-kb_st), label = 'Pass')
	# 	plt.xticks(ind, kb)
	# 	plt.legend()
	# 	plt.xlabel('Information set')
	# 	plt.ylabel('Strategy percent')
	# 	plt.title('Kuhn Strategy for Player 1 after check/bet action with {} iterations'.format(self.iterations))

	# 	plt.subplot(223)
	# 	plt.bar(ind, 1 - b_st, label = 'Bet')
	# 	plt.bar(ind, b_st, bottom=(1-b_st), label = 'Pass')
	# 	plt.xticks(ind, b)
	# 	plt.legend()
	# 	plt.xlabel('Information set')
	# 	plt.ylabel('Strategy percent')
	# 	plt.title('Kuhn Strategy for Player 2 after bet action with {} iterations'.format(self.iterations))

	# 	plt.subplot(222)
	# 	plt.bar(ind, 1 - k_st, label = 'Bet')
	# 	plt.bar(ind, k_st, bottom=(1-k_st), label = 'Pass')
	# 	plt.xticks(ind, k)
	# 	plt.legend()
	# 	plt.xlabel('Information set')
	# 	plt.ylabel('Strategy percent')
	# 	plt.title('Kuhn Strategy for Player 2 after check with {} iterations'.format(self.iterations))
	# 	plt.show()

	# 	expl_i = []
	# 	expl = []
	# 	for i in self.exploit:
	# 		expl_i.append(i)
	# 		expl.append(self.exploit[i])
	# 	plt.plot(expl_i, expl)
	# 	plt.show()


	def external_cfr(self, cards, history, pot, nodes_touched, traversing_player):
		plays = len(history)
		acting_player = plays % 3
		self.counter += 1

		if plays >= 3:
			
			ct = cards[traversing_player]
			
			ca = cards[acting_player]
			if acting_player == 0:
				prev1 = 2
				prev2 = 1
			elif acting_player == 1:
				prev1 = 0
				prev2 = 2
			elif acting_player == 2:
				prev1 = 1
				prev2 = 0
			if traversing_player == 0:
				other1 = 2
				other2 = 1
			elif traversing_player == 1:
				other1 = 0
				other2 = 2
			elif traversing_player == 2:
				other1 = 1
				other2 = 0

			if history[-3:] == 'kkk': #all check
				print('history', history)
				print('traversing player', traversing_player)
				print('traversing player cards', ct)
				print('other1 cards', cards[other1])
				print('other2 cards', cards[other2])
				if ct > cards[other1] and ct > cards[other2]:
					print('return', 2)
					return 2
				else:
					print('return', 1)
					return -1

			elif history[-3:] == 'bff': #acting player bet, others folded
				print('history', history)
				print('traversing player', traversing_player)
				print('traversing player cards', ct)
				print('other1 cards', cards[other1])
				print('other2 cards', cards[other2])
				if traversing_player == acting_player:
					print('return', 2)
					return 2
				else:
					print('return', -1)
					return -1

			elif history[-3:] == 'bcf': #acting player bet, 1 call, 1 fold
				print('history', history)
				print('traversing player', traversing_player)
				print('traversing player cards', ct)
				print('other1 cards', cards[other1])
				print('other2 cards', cards[other2])
				if traversing_player == acting_player:
					if ct > cards[prev2]:
						print('return', 3)
						return 3
					else:
						print('return', -2)
						return -2
				elif traversing_player == prev2:
					if ct > cards[acting_player]:
						print('return', 3)
						return 3
					else:
						print('return', -2)
						return -2
				elif traversing_player == prev1:
					print('return', -1)
					return -1

			elif history[-3:] == 'bfc':
				print('history', history)
				print('traversing player', traversing_player)
				print('traversing player cards', ct)
				print('other1 cards', cards[other1])
				print('other2 cards', cards[other2])
				if traversing_player == acting_player:
					if ct > cards[prev1]:
						print('return', 3)
						return 3
					else:
						print('return', -2)
						return -2
				elif traversing_player == prev1:
					if ct > cards[acting_player]:
						print('return', 3)
						return 3
					else:
						print('return', -2)
						return -2
				elif traversing_player == prev2:
					print('return', -1)
					return -1

			elif history[-3:] == 'bcc':
				print('history', history)
				print('traversing player', traversing_player)
				print('traversing player cards', ct)
				print('other1 cards', cards[other1])
				print('other2 cards', cards[other2])
				if ct > cards[other1] and ct > cards[other2]:
					print('return', 4)
					return 4
				else:
					print('return', -2)
					return -2

		num_actions = 2

		infoset = str(cards[acting_player]) + history
		if infoset not in self.nodes:
			self.nodes[infoset] = Node(num_actions)

		nodes_touched += 1


		strategy = self.nodes[infoset].get_strategy()

		if acting_player == traversing_player:
			util = np.zeros(self.bet_options) #2 actions
			node_util = 0

			for a in range(num_actions):
				if a == 0:
					if 'b' in history:
						next_history = history + 'f'
					else:
						next_history = history + 'k'
				elif a == 1:
					if 'b' in history:
						next_history = history + 'c'
					else:
						next_history = history + 'b'
					pot += 1
				util[a] = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player)
				node_util += strategy[a] * util[a]

			for a in range(num_actions):
				regret = util[a] - node_util
				self.nodes[infoset].regret_sum[a] += regret
			return node_util

		else: #acting_player != traversing_player
			util = 0
			if random.random() < strategy[0]:
					if 'b' in history:
						next_history = history + 'f'
					else:
						next_history = history + 'k'
			else: 
					if 'b' in history:
						next_history = history + 'c'
					else:
						next_history = history + 'b'
					pot += 1
			util = self.external_cfr(cards, next_history, pot, nodes_touched, traversing_player)
			for a in range(num_actions):
				self.nodes[infoset].strategy_sum[a] += strategy[a]
			return util

	# def do_cfr(self, cards, history, p0, p1, p2, pot, nodes_touched):
	# 	plays = len(history)
	# 	acting_player = plays % 3
	# 	opponent_player = 1 - acting_player
	# 	self.counter += 1

	# 	if plays >= 3:
	# 		if history[-1] == 'f': #bet fold
	# 			return 1
	# 		if (history[-1] == history[-2] == 'k') or (history[-1] == 'c'): #check check or bet call, go to showdown
	# 			if cards[acting_player] > cards[opponent_player]:
	# 				return pot/2 #profit
	# 			else:
	# 				return -pot/2

	# 	num_actions = 2
	# 	infoset = str(cards[acting_player]) + history

	# 	if infoset not in self.nodes:
	# 		self.nodes[infoset] = Node(num_actions)

	# 	if acting_player == 0:
	# 		realization_weight = p0
	# 	elif acting_player == 1:
	# 		realization_weight = p1
	# 	else:
	# 		realization_weight = p2

	# 	strategy = self.nodes[infoset].get_strategy(realization_weight)
	# 	util = np.zeros(self.bet_options) #2 actions
	# 	node_util = 0

	# 	for a in range(num_actions):
	# 		#print('a is: {}'.format(a))
	# 		if a == 0:
	# 			if 'b' in history:
	# 				next_history = history + 'f'
	# 			else:
	# 				next_history = history + 'k'
	# 		elif a == 1:
	# 			if 'b' in history:
	# 				next_history = history + 'c'
	# 			else:
	# 				next_history = history + 'b'
	# 			pot += 1
	# 		if acting_player == 0:
	# 			util[a] = -self.do_cfr(cards, next_history, p0*strategy[a], p1, p2, pot, nodes_touched)
	# 		elif acting_player == 1: 
	# 			util[a] = -self.do_cfr(cards, next_history, p0, p1*strategy[a], p2, pot, nodes_touched)
	# 		elif acting_player == 2:
	# 			util[a] = -self.do_cfr(cards, next_history, p0, p1, p2*strategy[a], pot, nodes_touched)
	# 		node_util += strategy[a] * util[a]

	# 	for a in range(num_actions):
	# 		regret = util[a] - node_util
	# 		if acting_player == 0:
	# 			self.nodes[infoset].regret_sum[a] += p1 * p2 * regret
	# 		elif acting_player == 1:
	# 			self.nodes[infoset].regret_sum[a] += p0 * p2 * regret
	# 		elif acting_player == 2:
	# 			self.nodes[infoset].regret_sum[a] += p0 * p1 * regret

	# 	return node_util

class LimitLeducCFR:
	def __init__(self, iterations, decksize):
		self.iterations = iterations
		self.decksize = decksize
		self.cards = sorted(np.concatenate((np.arange(decksize),np.arange(decksize))))
		self.nodes = {}

	def cfr_iterations_chance(self):
		util = 0
		for i in range(self.iterations):
			random.shuffle(self.cards)
			util += self.do_cfr(self.cards[:2], '', 1, 1, 2, 1)
		print('Average game value: {}'.format(util/self.iterations))
		for i in sorted(self.nodes):
			print(i, self.nodes[i].get_average_strategy())

	def cfr_iterations_vanilla(self):
		util = 0
		for i in range(self.iterations):
			for perm_card in list(permutations(self.cards)):
				util += self.do_cfr(perm_card[:2], '', 1, 1, 2, 1)
		print('Average game value: {}'.format(util/self.iterations))
		for i in sorted(self.nodes):
			print(i, self.nodes[i].get_average_strategy())

	def valid_bets(self, history, round):
# 		bet_options = ['f', 'k', 'c', 'b', 'r'] #fold check call bet raise
		if round == 2:
			if 'kk' in history:
				history = history.split('kk',1)[1]
			else:
				history = history.split('c',1)[1]
		if len(history) >=1:
			if history[-1] == 'b' or history[-1] == 'r':
				if history [-3:] == 'rrr': #maximum 4 bets in a round
					return ['f', 'c'] #can only fold or call after a bet and 3 raises 
				else:
					return ['f', 'c', 'r'] #can fold or call or raise when facing a bet or raise
			if history[-1] == 'k':
				return ['k', 'b']

		if history == '':
			return ['k', 'b'] #check or bet after check or when first to act

	def do_cfr(self, cards, history, p0, p1, pot, round):
		plays = len(history)
		acting_player = plays % 2
		opponent_player = 1 - acting_player
		betsize = 2*round


		if plays >= 2:
			if history[-1] == 'f': #bet fold
				return (pot-betsize)/2
			if (history[-1] == history[-2] == 'k'): #check check or bet call, go to showdown
				if cards[acting_player] > cards[opponent_player]:
					return pot/2 #profit
				else:
					return -pot/2
			if (round == 1 and history[-1] == 'c'):
				round = 2
				betsize = 2*round
			elif (round == 2 and history[-1] == 'c'): #bet call in last round
				if cards[acting_player] > cards[opponent_player]:
					return pot/2
				else:
					return -pot/2

		actions = self.valid_bets(history, round)
		num_actions = len(actions)
		#bet_options = ['f', 'k', 'c', 'b', 'r']

		#print('infoset cards: {}'.format(cards[acting_player]))
		#print('infoset history: {}'.format(history))
		infoset = str(cards[acting_player]) + history

		if infoset not in self.nodes:
			self.nodes[infoset] = Node(num_actions) #check, call, bet, fold, raise
		#after bet: raise, call, fold
		#after check: bet, check
		#after raise: raise, call, fold
		#after fold: hand over
		#after call: hand over

		if acting_player == 0:
			realization_weight = p0
		else:
			realization_weight = p1
		strategy = self.nodes[infoset].get_strategy(realization_weight)

		util = np.zeros(num_actions) #2 actions
		node_util = 0

		for i, a in enumerate(actions):
			# bet_options = ['f', 'k', 'c', 'b', 'r'] #fold check call bet raise
			#print('a is: {}'.format(a))
			#a_num = bet_options.index(a)
			next_history = history + a

			if a == 'c' or a == 'b':
				pot += betsize

			if a == 'r':
				pot += 2*betsize

			if acting_player == 0:
				util[i] = -self.do_cfr(cards, next_history, p0*strategy[i], p1, pot, round)
			elif acting_player == 1: 
				 util[i] = -self.do_cfr(cards, next_history, p0, p1*strategy[i], pot, round)
			node_util += strategy[i] * util[i]

		for i, a in enumerate(actions):
			regret = util[i] - node_util
			if acting_player == 0:
				self.nodes[infoset].regret_sum[i] += p1 * regret
			elif acting_player == 1:
				self.nodes[infoset].regret_sum[i] += p0 * regret

		return node_util


class NLLeducCFR:
	def __init__(self, iterations, decksize):
		self.iterations = iterations
		self.decksize = decksize
		self.cards = sorted(np.concatenate((np.arange(decksize),np.arange(decksize))))
		self.nodes = {}

	def cfr_iterations_chance(self):
		util = 0
		for i in range(self.iterations):
			random.shuffle(self.cards)
			util += self.do_cfr(self.cards[:2], '', 1, 1, 2, 1, 19, 19)
		print('Average game value: {}'.format(util/self.iterations))
		for i in sorted(self.nodes):
			print(i, self.nodes[i].get_average_strategy())

	def cfr_iterations_vanilla(self):
		util = 0
		for i in range(self.iterations):
			for perm_card in list(permutations(self.cards)):
				util += self.do_cfr(perm_card[:2], '', 1, 1, 2, 1, 19, 19)
		print('Average game value: {}'.format(util/self.iterations))
		for i in sorted(self.nodes):
			print(i, self.nodes[i].get_average_strategy())

	def valid_bets(self, history, round, acting_stack):
# 		bet_options = ['f', 'k', 'c', 'b', 'r'] #fold check call bet raise
		if round == 2:
			if ['k', 'k'] in history:
				history = history.split('kk',1)[1]
			else:
				history = history.split('c',1)[1]
		if len(history) >=1:
			if history[-1] == 'b' or history[-1] == 'r':
				if history [-3:] == 'rrr': #maximum 4 bets in a round
					return ['f', 'c'] #can only fold or call after a bet and 3 raises 
				else:
					return ['f', 'c', 'r'] #can fold or call or raise when facing a bet or raise
			if history[-1] == 'k':
				return ['k', 'b']

		if history == '':
			poss_bets = arange(acting_stack)
			#bets_for_array = str(i) + 'b' for i in poss_bets
			return np.concatenate(('k',  bets_for_array)) #check or bet after check or when first to act

	def do_cfr(self, cards, history, p0, p1, pot, round, p0stack, p1stack):
		plays = len(history)
		if round == 1:
			acting_player = plays % 2
			opponent_player = 1 - acting_player
		elif round == 2:
			acting_player = 1 - (plays % 2)
			opponent_player = 1 - acting_player


		if plays >= 2:
			if history[-2] > 0 and history[-1] == 0: #bet fold
				return (pot-history[-2])/2
			if (history[-1] == history[-2] == 0): #check check, go to showdown
				if cards[acting_player] > cards[opponent_player]:
					return pot/2 #profit
				else:
					return -pot/2
			if (round == 1 and history[-1] == 'c'): #call bet in 1st round
				round = 2
			elif (round == 2 and history[-1] == 'c'): #bet call in last round
				if cards[acting_player] > cards[opponent_player]:
					return pot/2
				else:
					return -pot/2

		if acting_player == 0:
			actions = self.valid_bets(history, round, p0stack)
		elif acting_player == 1:
			actions = self.valid_bets(history, round, p1stack)
		num_actions = len(actions)
		infoset = np.concatenate((cards[acting_player], history))

		if infoset not in self.nodes:
			self.nodes[infoset] = Node(num_actions) #check, call, bet, fold, raise
		#after bet: raise, call, fold
		#after check: bet, check
		#after raise: raise, call, fold
		#after fold: hand over
		#after call: hand over

		if acting_player == 0:
			realization_weight = p0
		else:
			realization_weight = p1
		strategy = self.nodes[infoset].get_strategy(realization_weight)

		util = np.zeros(num_actions) #2 actions
		node_util = 0

		for i, a in enumerate(actions):
			# bet_options = ['f', 'k', 'c', 'b', 'r'] #fold check call bet raise
			#print('a is: {}'.format(a))
			#a_num = bet_options.index(a)
			next_history = np.concatenate((history, a))
			pot += a

			if acting_player == 0:
				util[i] = -self.do_cfr(cards, next_history, p0*strategy[i], p1, pot, round, p0stack, p1stack)
			elif acting_player == 1: 
				 util[i] = -self.do_cfr(cards, next_history, p0, p1*strategy[i], pot, round, p0stack, p1stack)
			node_util += strategy[i] * util[i]

		for i, _ in enumerate(actions):
			regret = util[i] - node_util
			if acting_player == 0:
				self.nodes[infoset].regret_sum[i] += p1 * regret
			elif acting_player == 1:
				self.nodes[infoset].regret_sum[i] += p0 * regret

		return node_util


# class LimitLeduc:
# 	def __init__(self, dealer_player = 0, players = [Human('p1'), Human('p2')], ante = 1):
# 		#super().__init__()
# 		self.dealer_player = dealer_player
# 		self.num_players = len(players) #should be 2 or 3
# 		self.pot = num_players * ante 
# 		self.round = 1
# 		self.r1history = []
# 		self.r2history = []
# 		self.cards = [1,1,2,2,3,3]
# 		random.shuffle(self.cards)
# 		if self.num_players == 2:
# 			self.player0cards = self.cards[1 - dealer_player]
# 			self.player1cards = self.cards[dealer_player]

# 	def round_1():
# 		betsize = 2
# 		print(r1history)
# 		plays = len(history1)
# 		self.acting_player = (plays+dealer_player)%2 #first hand should be the non-dealer player
# 		self.pot = (num_players * ante) + betsize*r1history.count('b') + betsize*r1history.count('c') + 2*betsize*r1history.count('r')

# 		if r1history[-1:] == 'f':
# 			profit = (self.pot - betsize) / 2
# 			players[acting_player].profit += profit
# 			players[1-acting_player].profit -= profit
# 			print('End of game! Player {} wins pot of {} (profits {}\n'.format(self.acting_player,self.pot,profit))

# 		if r1history[-2:] == [k,k] or r1history[-1:] == [c]:
# 			round_2()

# 		print('\nPot size: {}'.format(self.pot))
# 		print('Player {} turn to act'.format(self.acting_player))
# 		bet = players[acting_player].select_move(valid_bets())
# 		if 
# 		self.pot += #or make this auto calculate from the history?
# 		self.r1history.append(bet)
# 		print('Action: {}'.format(bet))

# 		round_2()


# 	def round_2():
# 		betsize = 4
# 		self.round = 2
# 		print(r1history, r2history)
# 		plays = len(history2)
# 		self.acting_player = dealer_player 
# 		self.flop_card = cards[num_players]
# 		print('\n Dealing flop card: {}.format(self.flop_card)')

# 		if r2history[-1:] == [f]:
# 			profit = (self.pot - betsize) / 2
# 			players[acting_player].profit += profit
# 			players[1-acting_player].profit -= profit
# 			print('End of hand! Player {} wins pot of {} (profits {})\n'.format(self.acting_player, self.pot, profit))

# 		if r2history[-2:] == [k,k] or r2history[-1:] == [c]:
# 			winner = evaluate_hands()
# 			if winner == -1:
# 				print('Tie game! Pot size {} and both players had {}\n'.format(self.pot, self.player0cards.append(self.flop_card)))
# 			else:
# 				profit = pot / 2 
# 				print('End of hand! Player {} wins pot of {} (profits {})\n'.format(winner, self.pot, profit))
# 				players[winner].profit += profit
# 				players[1-winner].profit -= profit
		
# 		print('\nPot size: {}'.format(self.pot))
# 		print('Player {} turn to act'.format(self.acting_player))
# 		bet = players[acting_player].select_move(valid_bets())
# 		print('Action: {}'.format(bet))
# 		self.r2history.append(bet)
# 		round_2()


# 	def evaluate_hands():
# 		#returns 0 for player 0 winning, 1 for player 1 winning, -1 for draw
# 		if self.player0cards == self.flop_card:
# 			return 0
# 		elif:
# 			self.player1cards == self.flop_card:
# 			return 1
# 		elif:
# 			self.player0cards > self.player1cards:
# 			return 0
# 		elif:
# 			self.player1cards > self.player0cards:
# 			return 1
# 		else:
# 			return -1

# 	def make_bet():
# 		players[self.acting_player].select_move(valid_bets())

# 	def valid_bets():
# 		bet_options = ['f', 'k', 'c', 'b', 'r'] #fold check call bet raise

# 		if self.round == 1:
# 			history = self.r1history
# 		else:
# 			history = self.r2history

# 		if history[-1:] == 'b' or history[-1:] == 'r':
# 			betoptions[3] = False #Bet invalid -- Can only bet at the start of a round or after a check
# 			betoptions[1] = False #Check invalid -- Can only check at the start of a round or after a check

# 		if history[-1:] != 'b':
# 			betoptions[4] = False #Raise invalid -- Can only raise after a bet (and only one raise per round)

# 		if history[-1:] == 'k' or history[-1:] == '': #Previous action check or none
# 			betoptions[0] = False #Fold invalid -- Can only fold after a bet or raise
# 			betoptions[2] = False #Call invalid -- Can only call after a bet or raise

# 		return bet_options

# class Kuhn(Poker):
# 	def __init__(self, decksize, ):
# 		super().__init__()
# 		self.cards = 
# 		self.history1 = []
# 		self
# 		pdeck = np.arange(decksize)
# 		random.shuffle(pdeck)
# 		self.deck = deck
# 		self.rounds = 1
# 		self.ante = 1
# 		self.max_bets = 1
# 		self.r1_betsize = 1

	# def round_1():
	# 	betsize = self.r1_betsize



	# def round_1():
	# 	betsize = self.game.r1_betsize
	# 	if self.gametype == 'k':
			

	# 	if self.gametype == 'l':

	# def round_2():
	# 	betsize = self.game.r2_betsize
	# 	if self.gametype == 'l':


	# def valid_bets(self, game):
	# 	pass


	# def evaluate(self, game):
	# 	if self.game == 'k':
	# 		if self.player1cards > self.player2cards:
	# 			return 1
	# 		else:
	# 			return 2
	# 	if self.game == 'l':
	# 		if self.player1cards == self.community_card:
	# 			return 1
	# 		elif self.player2cards = self.community_card:
	# 			return 2
	# 		elif self.player1cards > self.player2cards:
	# 			return 1
	# 		elif self.player2cards > self.player1cards:
	# 			return 0
	# 		else:
	# 			return -1 #Tie game


# Kuhn2 = namedTuple('Kuhn', ['deck', 'rounds', 'ante', 'max_bets', 'r1_betsize'])
# newKuhn = Kuhn2(random.shuffle(np.arange(decksize)), 1, 1, 1, 1)
# print(newKuhn.deck)


# class BaseGame:
# 	def __init__(self, gameType):
# 		self.gameType = gameType
# 		self.createDeck()
# 		self.rounds = 
# 		self.ante =
# 		self.max_bets =

# class Kuhn:
# 	def __init__(self, decksize):
# 		self.deck = random.shuffle(np.arange(decksize))
# 		self.rounds = 1
# 		self.ante = 1
# 		self.max_bets = 1
# 		self.r1_betsize = 1

# class LimitLeduc:
# 	def __init__(self):
# 		self.deck = random.shuffle([1,1,2,2,3,3])
# 		self.rounds = 2
# 		self.ante = 1
# 		self.max_bets = 4
# 		self.r1_betsize = 2
# 		self.r2_betsize = 4

# class NLLeduc:
# 	def __init__(self, ante, stack_size):
# 		self.deck = random.shuffle([1,1,2,2,3,3])
# 		self.rounds = 2
# 		self.ante = ante
# 		self.max_bets = 4
# 		self.stack_size = stack_size

# class Node: 
# 	def __init__(self, regretSum, strategy, strategySum, num_actions):
# 		self.regretSum = np.zeros(num_actions)
# 		self.strategy = np.zeros(num_actions)
# 		self.strategySum = np.zeros(num_actions)


# 	def getStrategy():

# 	def getAverageStrategy():


		# if self.num_players == 3:
		# 	self.player0cards = self.cards[(dealer_player+1)%3]
		# 	self.player1cards = self.cards[(dealer_player+2)%3]
		# 	self.player2cards = self.cards[dealer_player]

class Player:
	def __init__(self):
		self.profit = 0

# class RandomAgent(Player):
# 	def select_move(self, game):
# 		return random.choice(game.valid_bets())

class Human(Player):
	def select_move(self, betoptions):
		print(betoptions)
		move = int(input('Enter your bet: '))
		if move in betoptions:
			return move
		else:
			print('Invalid action, try again')
			self.select_move(betoptions)
		# if betoptions[0]: print('FOLD: f')
  #   	if betoptions[1]: print('CHECK: k')
  #   	if betoptions[2]: print('CALL {}: c'.format(betsize))
  #   	if betoptions[3]: print('BET {}: b'.format(betsize))
  #   	if betoptions[4]: print('RAISE {}: r'.format(betsize*2))
		# input(move)
		# if move in game.valid_bets():
		# 	return move
		# else:
		# 	print('Invalid action, try again')
		# 	select_move(game)



# class CFRAgent: 
# 	def __init__(self, cfr_type):
# 		self.cfr_type = cfr_type

# class DSAgent:
# 	def __init__(self):

# class BRF:
# 	def __init__(self, tree):

if __name__ == "__main__":
	# p0 = Human()
	# p1 = Human()
	# cards = [1,2,3]
	# random.shuffle(cards)
	# hands = 5

	# for i in range(hands):
	# 	print('STARTING HAND NUMBER: {}'.format(i))
	# 	p = Kuhn(players = [p0, p1], dealer_player = i%2, ante = 1, shuffled_deck = cards)
	# 	p.game()
	# 	print('PLAYER 0 PROFIT: {}'.format(p0.profit))
	# 	print('PLAYER 1 PROFIT: {}'.format(p1.profit))
	k = KuhnCFR(100000, 3, 0)
	#k.cfr_iterations_vanilla()

	#k = LimitLeducCFR(100000, 3)
	k.cfr_iterations_chance()
	#k.cfr_iterations_external()

	#k = Kuhn3CFR(100000, 4)
	#k.cfr_iterations_external()
