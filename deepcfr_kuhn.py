import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.0001
LR = 1e-2
loss = torch.nn.MSELoss()

#tensorboard
#initialize to 0?
#log things like loss and layers to visualize in tensorboard 

# class Node:
# 	def __init__(self, num_actions):
# 		self.regret_sum = np.zeros(num_actions)
# 		self.strategy = np.zeros(num_actions)
# 		self.strategy_sum = np.zeros(num_actions)
# 		self.num_actions = num_actions

# 	def get_strategy(self):
# 		normalizing_sum = 0
# 		for a in range(self.num_actions):
# 			if self.regret_sum[a] > 0:
# 				self.strategy[a] = self.regret_sum[a]
# 			else:
# 				self.strategy[a] = 0
# 			normalizing_sum += self.strategy[a]

# 		for a in range(self.num_actions):
# 			if normalizing_sum > 0:
# 				self.strategy[a] /= normalizing_sum
# 			else:
# 				self.strategy[a] = 1.0/self.num_actions

# 		return self.strategy

class GiantLookupTable(nn.Module):
	def __init__(self, ncardtypes, nbets, nactions, dim = 128):
		super(GiantLookupTable, self).__init__()
		#take input, if in memory, return latest value
		#self.memory

	def forward(self, cards, bets):
		for (n, mem) in enumerate(reversed(self.memory)):
			infoset, timestep, regrets = mem
			cards_mem, history_mem = infoset
			bets_mem = -torch.ones_like(bets)
			for (a, b) in enumerate(history_mem):
				bets_mem[0][a] = b / (sum(history_mem[:a]) + 2)
			if (torch.equal(bets, bets_mem)) and (torch.equal(cards, torch.tensor([[cards_mem]], dtype=torch.float))):
				return torch.tensor(regrets)
		return torch.tensor([0, 0])


	def train_cfr_value(self, curr_memory, nbets, batch_size):
		self.memory = curr_memory

	#save memory to parameter, look up latest value if previously encountered
		
class DeepCFRNet(nn.Module):
	def __init__(self, ncardtypes, nbets, nactions, dim = 128):
		super(DeepCFRNet, self).__init__()
		self.card1 = nn.Linear(3, dim)
		self.card2 = nn.Linear(dim, dim)
		self.card3 = nn.Linear(dim, dim)

		self.bet1 = nn.Linear(nbets * 2, dim)
		self.bet2 = nn.Linear(dim, dim)

		self.comb1 = nn.Linear(2 * dim, dim)
		self.comb2 = nn.Linear(dim, dim)
		self.comb3 = nn.Linear(dim, dim)

		self.action_head = nn.Linear(dim, nactions)

	def train_cfr_value(self, curr_memory, nbets, batch_size):
		# print("VALUE NETWORK TRAINING FOR PLAYER {}".format(i))
		curr_optim = torch.optim.Adam(self.parameters(), lr = LR)
		# curr_optim = [torch.optim.Adam(self.val_nets[0].parameters(), lr = LR), torch.optim.Adam(self.val_nets[1].parameters(), lr = LR)]
		for s in range(8): #sgd iterations
			batch_loss_history = []
			print('sgd iteration: {}'.format(s))
			for (n, mem) in enumerate(curr_memory):
				infoset, timestep, regrets = mem
				cards, history = infoset
				bets = -torch.ones(nbets)
				for (a, b) in enumerate(history):
					bets[a] = b / (sum(history[:a]) + 2)
				valnet_out = self.forward(torch.tensor([[cards]], dtype=torch.float), bets.unsqueeze(0))
				print('****************')
				print('CARDS: ', cards)
				print('BETS: ', bets)
				print('VALNET ADVANTAGES: ', valnet_out)
				print('MEMORY ADVANTAGES: ', regrets)
				batch_loss_history.append(timestep * loss(valnet_out, torch.tensor([regrets]).float()))
				if n % batch_size == 0 and n > 0:
					batch_loss_history_mean = torch.stack(batch_loss_history).mean()
					curr_optim.zero_grad()
					batch_loss_history_mean.backward()
					curr_optim.step()
					batch_loss_history = []

	def forward(self, cards, bets):
		#card branch
		x = F.relu(self.card1(cards))

		#bet branch
		bet_size = bets.clamp(0, 1e6)
		bet_occurred = bets.ge(0) #bets that are >= 0
		bet_feats = torch.cat([bet_size, bet_occurred.float()], dim=1)
		y = F.relu(self.bet1(bet_feats))

		#combined trunk
		z = torch.cat([x, y], dim = 1)
		z = F.relu(self.comb1(z))

		z_mean = z.mean()
		z_std = z.std()
		z = (z - z_mean) / (z_std + EPS)
		return self.action_head(z)

class KuhnCFR:
	def __init__(self, iterations, decksize):
		self.nbets = 2
		self.iterations = iterations
		self.decksize = decksize
		self.cards = np.arange(decksize)
		self.bet_options = 2
		self.nodes = {}

		self.m_v = [[], []]
		self.m_pi = []

		self.batch_size = 20
		self.val_nets = [lambda cards, bets: torch.tensor([0, 0])]*2

		self.strategynet = DeepCFRNet(decksize, self.nbets, self.bet_options, dim=8)
		self.strategynet_optim = torch.optim.Adam(self.strategynet.parameters(), lr = LR)

	def get_strategy(self, adv):
		normalizing_sum = 0
		strategy = np.zeros(self.bet_options)
		adv = torch.squeeze(adv)
		for a in range(self.bet_options):
			if adv[a] > 0:
				strategy[a] = adv[a]
			else:
				strategy[a] = 0
			normalizing_sum += strategy[a]

		for a in range(self.bet_options):
			if normalizing_sum > 0:
				strategy[a] /= normalizing_sum
			else:
				strategy[a] = 1.0/self.bet_options

		return strategy

	def cfr_iterations_deep(self, k = 20):
		util = np.zeros(2)
		for t in range(1, self.iterations + 1): 
			for i in range(2):
				for _ in range(k):
					random.shuffle(self.cards)
					util[i] += self.deep_cfr(self.cards[:2], [], 2, 0, i, t)

				self.val_nets[i] = DeepCFRNet(self.decksize, self.nbets, self.bet_options, dim=8)
				curr_valnet = self.val_nets[i]
				curr_memory = self.m_v[i]				
				
				curr_valnet.train_cfr_value(curr_memory, self.nbets, self.batch_size)
		
		for p in range(10):
			batch_loss_history = []
			print('ITERATION OF POLICY NETWORK: ', p)
			#iterate through memory, loss over strategynet output given cards and bets from memory compared to strategy from memory
			for (n, mem) in enumerate(self.m_pi):
				infoset, timestep, action_probs, _ = mem
				cards, history = infoset
				bets = -torch.ones(self.nbets)
				for (a, b) in enumerate(history):
					bets[a] = b / (sum(history[:a]) + 2)
				strategynet_out = self.strategynet.forward(torch.tensor([[cards]], dtype=torch.float), bets.float().unsqueeze(0))
				batch_loss_history.append(timestep * loss(strategynet_out, torch.tensor([action_probs]).float()))
				if n % self.batch_size == 0 and n>0:
					batch_loss_history = torch.stack(batch_loss_history).mean()
					self.strategynet_optim.zero_grad()
					batch_loss_history.backward()
					self.strategynet_optim.step()
					batch_loss_history = []
		
		bets = -torch.ones(self.nbets)
		print('Average game value: {}'.format(util[0]/(self.iterations*k)))
		a1 = self.strategynet.forward(torch.tensor([[0]], dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
		a2 = self.strategynet.forward(torch.tensor([[1]], dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
		a3 = self.strategynet.forward(torch.tensor([[2]], dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
		bets = -torch.ones(self.nbets)
		bets[0] = 0.5
		a4 = self.strategynet.forward(torch.tensor([[2]], dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
		a5 = self.strategynet.forward(torch.tensor([[0]], dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
		print('card 0 opening action: ', a1)
		print('card 1 opening action: ', a2)
		print('card 2 opening action: ', a3)
		print('card 2 facing bet action: ', a4)
		print('card 0 facing bet action: ', a5)
		#return self.m_pi

	def deep_cfr(self, cards, history, pot, nodes_touched, traversing_player, t):
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

		infoset = cards[acting_player], history

		bets = -torch.ones(self.nbets)
		for (i, b) in enumerate(history):
			# bets should be a list of the proportion each bet is of the pot size
			bets[i] = b / (sum(history[:i]) + 2)

		nodes_touched += 1

		# infoset_str = str(cards[acting_player]) + str(history)
		# if infoset_str not in self.nodes:
		# 	self.nodes[infoset_str] = Node(self.bet_options)
		# strategy = self.nodes[infoset_str].get_strategy()

		if acting_player == traversing_player:
			util = np.zeros(self.bet_options) #2 actions
			node_util = 0
			advantages = self.val_nets[acting_player](torch.tensor([[cards[acting_player]]], dtype=torch.float), bets.unsqueeze(0))
			strategy = self.get_strategy(advantages)
			for a in range(self.bet_options):
				next_history = history + [a]
				pot += a
				util[a] = self.deep_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
				node_util += strategy[a] * util[a]

			action_advantages = np.zeros(self.bet_options)
			for a in range(self.bet_options):
				action_advantages[a] = util[a] - node_util
				#regret = util[a] - node_util
				#self.nodes[infoset_str].regret_sum[a] += regret

			self.m_v[traversing_player].append((infoset, t, action_advantages))
			return node_util

		else: #acting_player != traversing_player
			advantages = self.val_nets[acting_player](torch.tensor([[cards[acting_player]]],  dtype=torch.float), bets.unsqueeze(0))
			strategy = self.get_strategy(advantages)
			self.m_pi.append((infoset, t, strategy, acting_player))
			# util = 0
			if random.random() < strategy[0]:
				next_history = history + [0]
			else: 
				next_history = history + [1]
				pot += 1
			return self.deep_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
			# util = self.deep_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
			# for a in range(self.bet_options):
			# 	self.nodes[infoset_str].strategy_sum[a] += strategy[a]
			# return util


if __name__ == "__main__":
	k = KuhnCFR(300, 3)
	k.cfr_iterations_deep()