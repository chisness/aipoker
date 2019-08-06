import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.0001
LR = 1e-2

class DeepCFRNet(nn.Module):
	def __init__(self, ncardtypes, nbets, nactions, dim = 128):
		super(DeepCFRNet, self).__init__()
		self.card1 = nn.Linear(1, dim)
		self.card2 = nn.Linear(dim, dim)
		self.card3 = nn.Linear(dim, dim)

		self.bet1 = nn.Linear(nbets * 2, dim)
		self.bet2 = nn.Linear(dim, dim)

		self.comb1 = nn.Linear(2 * dim, dim)
		self.comb2 = nn.Linear(dim, dim)
		self.comb3 = nn.Linear(dim, dim)

		self.action_head = nn.Linear(dim, nactions)

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

		self.m_v = [[], []]
		self.m_pi = []

		self.batch_size = 10
		self.val_nets = [DeepCFRNet(decksize, self.nbets, self.bet_options, dim=8), DeepCFRNet(decksize, self.nbets, self.bet_options, dim=8)]
		self.val_net_optims = [torch.optim.Adam(self.val_nets[0].parameters(), lr = LR), torch.optim.Adam(self.val_nets[1].parameters(), lr = LR)]

		self.strategynet = DeepCFRNet(decksize, self.nbets, self.bet_options, dim=8)
		self.strategynet_optim = torch.optim.Adam(self.strategynet.parameters(), lr = LR)

	def get_strategy(self, adv):
		normalizing_sum = 0
		strategy = np.zeros(self.bet_options)
		adv = adv[0]
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

	def cfr_iterations_deep(self, k = 10):
		util = np.zeros(2)
		for t in range(1, self.iterations + 1): 
			for i in range(2):
				for k2 in range(k):
					random.shuffle(self.cards)
					util[i] += self.deep_cfr(self.cards[:2], [], 2, 0, i, t)

				self.val_nets[i] = DeepCFRNet(self.decksize, self.nbets, self.bet_options, dim=8)
				curr_valnet = self.val_nets[i]
				curr_memory = self.m_v[i]
				curr_optim = self.val_net_optims[i]
				loss = torch.nn.MSELoss()
				
				print("VALUE NETWORK TRAINING FOR PLAYER {}".format(i))
				for s in range(4): #sgd iterations
					batch_loss_history = []
					print('sgd iteration: {}'.format(s))
					for (n, mem) in enumerate(curr_memory):
						infoset, timestep, regrets = mem
						cards, history = infoset
						bets = -torch.ones(self.nbets)
						for (a, b) in enumerate(history):
							bets[a] = b / (sum(history[:a]) + 2)
						valnet_out = curr_valnet.forward(torch.tensor([[cards]], dtype=torch.float), bets.unsqueeze(0))
						print('****************')
						print('CARDS: ', cards)
						print('BETS: ', bets)
						print('VALNET ADVANTAGES: ', valnet_out)
						print('MEMORY ADVANTAGES: ', regrets)
						batch_loss_history.append(loss(valnet_out, torch.tensor([regrets]).float()))
						if n % self.batch_size == 0 and n > 0:
							batch_loss_history_mean = torch.stack(batch_loss_history).mean()
							curr_optim.zero_grad()
							batch_loss_history_mean.backward()
							curr_optim.step()
							batch_loss_history = []
		
		for p in range(10):
			batch_loss_history = []
			print('ITERATION OF POLICY NETWORK: ', p)
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

		if acting_player == traversing_player:
			util = np.zeros(self.bet_options) #2 actions
			node_util = 0
			advantages = self.val_nets[acting_player].forward(torch.tensor([[cards[acting_player]]], dtype=torch.float), bets.unsqueeze(0))
			strategy = self.get_strategy(advantages)
			for a in range(self.bet_options):
				next_history = history + [a]
				pot += a
				util[a] = self.deep_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
				node_util += strategy[a] * util[a]

			action_advantages = np.zeros(self.bet_options)
			for a in range(self.bet_options):
				action_advantages[a] = util[a] - node_util
			self.m_v[traversing_player].append((infoset, t, action_advantages))
			return node_util

		else: #acting_player != traversing_player
			advantages = self.val_nets[acting_player].forward(torch.tensor([[cards[acting_player]]],  dtype=torch.float), bets.unsqueeze(0))
			strategy = self.get_strategy(advantages)
			self.m_pi.append((infoset, t, strategy, acting_player))
			util = 0
			if random.random() < strategy[0]:
				next_history = history + [0]
			else: 
				next_history = history + [1]
				pot += 1
			return self.deep_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)


if __name__ == "__main__":
	k = KuhnCFR(1000, 3)
	k.cfr_iterations_deep()