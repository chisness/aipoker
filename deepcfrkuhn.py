import numpy as np
import random
from itertools import permutations
import matplotlib.pyplot as plt
import collections 
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 0.0001
LR = 1e-2

class Node:
	def __init__(self, num_actions):
		self.regret_sum = np.zeros(num_actions)
		self.strategy = np.zeros(num_actions)
		self.num_actions = num_actions

# class CardEmbedding(nn.Module):
# 	def __init__(self, dim):
# 		super(CardEmbedding, self).__init__()
# 		self.rank = nn.Embedding(13, dim)
# 		self.suit = nn.Embedding(4, dim)
# 		self.card = nn.Embedding(52, dim)

# 	def forward(self, input):
# 		B, num_cards = input.shape
# 		x = input.view(-1)

# 		valid = x.ge(0).float() #-1 means no card
# 		x = x.clamp(min=0)

# 		embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
# 		embs = embs * valid.unsqueeze(1) #zero out ' no card' embeddings

# 		#sum across the cards in the hole/board
# 		return embs.view(B, num_cards, -1).sum(1)


class CardEmbedding(nn.Module):
	def __init__(self, dim):
		super(CardEmbedding, self).__init__()
		self.card = nn.Embedding(3, dim)

	def forward(self, input):
		B, num_cards = input.shape
		x = input.view(-1)

		valid = x.ge(0).float() #-1 means no card
		x = x.clamp(min=0)

		embs = self.card(x)
		embs = embs * valid.unsqueeze(1) #zero out ' no card' embeddings

		#sum across the cards in the hole/board
		return embs.view(B, num_cards, -1).sum(1)

class DeepCFRNet(nn.Module):
	def __init__(self, ncardtypes, nbets, nactions, dim = 128):
		super(DeepCFRNet, self).__init__()

		#self.card_embeddings = nn.ModuleList([CardEmbedding(dim) for _ in range(ncardtypes)])

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
		#bets N x nbet_feats

		#card branch (embed hole, flop, and optionally turn/river)
		# card_embs = []
		# for embedding, card_group in zip(self.card_embeddings, cards):
		# 	card_embs.append(embedding(card_group))
		# card_embs = torch.cat(card_embs, dim=1)

		#x = F.relu(self.card1(card_embs))
		#print('PRE RELU', cards.shape)
		#print('cards', cards)
		x = F.relu(self.card1(cards))
		#x = F.relu(self.card2(x))
		#x = F.relu(self.card3(x))

		#bet branch
		bet_size = bets.clamp(0, 1e6)
		bet_occurred = bets.ge(0) #bets that are >= 0
		#print('bet size', bet_size)
		#print('bet occ', bet_occurred)
		bet_feats = torch.cat([bet_size, bet_occurred.float()], dim=1)
		#print('bet feats', bet_feats)
		y = F.relu(self.bet1(bet_feats))
		#y = F.relu(self.bet2(y) + y)

		#combined trunk
		# print('x', x)
		# print('y', y)
		
		z = torch.cat([x, y], dim = 1)
		# print('z', z)
		z = F.relu(self.comb1(z))
		# print('z relu', z)
		#z = F.relu(self.comb2(z) + z)
		#z = F.relu(self.comb3(z) + z)
		#print(z)
		z_mean = z.mean()
		z_std = z.std()
		z = (z - z_mean) / (z_std + EPS)
		# print('z norm', z)
		return self.action_head(z)

class KuhnCFR:
	def __init__(self, iterations, decksize, buckets):
		self.nbets = 2
		self.iterations = iterations
		self.decksize = decksize
		self.cards = np.arange(decksize)
		self.nodes = {}
		self.bet_options = 2
		self.buckets = buckets
		self.counter = 0
		self.exploit = collections.defaultdict(float)

		self.m_v = [[], []]
		self.m_pi = []
		
		self.s1 = []
		self.s2 = []
		self.s3 = []

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
		losses = []
		for t in range(1, self.iterations + 1): 
			worst_hand_opp_bet = []
			middle_hand_act_first = []
			best_hand_opp_bet = []
			# worst_hand_opp_bet_valnetwork0 = []
			# middle_hand_act_first_valnetwork0 = []
			# best_hand_opp_bet_valnetwork0 = []
			
			# worst_hand_opp_bet_valnetwork0.append(self.val_nets[0].forward(torch.tensor([[cards]], dtype=torch.float), torch.tensor(bets).float().unsqueeze(0)))
			print('THIS IS T IN THE ITERATIONS LOOP: ', t)
			if t>0 and t%10 == 0:
				for (infoset, _, action_probs, _) in self.m_pi:
					card, bet_history = infoset
					if len(bet_history) == 1 and bet_history[0] == 1 and card == 0:
						worst_hand_opp_bet.append(action_probs[0])
					elif len(bet_history) == 1 and bet_history[0] == 1 and card == 2:
						best_hand_opp_bet.append(action_probs[1])
					elif len(bet_history) == 0 and card == 1:
						middle_hand_act_first.append(action_probs[0])

				print('LOSSES', losses)

				plt.subplot(211)
				plt.scatter(np.arange(len(worst_hand_opp_bet)), worst_hand_opp_bet, label = 'worst hand opp bet')
				plt.legend()

				plt.subplot(221)
				plt.scatter(np.arange(len(best_hand_opp_bet)), best_hand_opp_bet, label = 'best hand opp bet')
				plt.legend()

				plt.subplot(222)
				plt.scatter(np.arange(len(middle_hand_act_first)), middle_hand_act_first, label = 'middle hand act first')
				plt.legend()

				plt.subplot(212)
				plt.plot(np.arange(len(losses)), losses, label = 'losses')
				plt.legend()
				plt.show()
			for i in range(2):
				for k2 in range(k):
					random.shuffle(self.cards)
					# print('THIS IS T IN THE ITERATIONS LOOP AFTER K: ', t)
					# print('THIS IS K*T IN THE ITERATIONS LOOP: ', k2*t)
					util[i] += self.deep_cfr(self.cards[:2], [], 2, 0, i, t)

				self.val_nets[i] = DeepCFRNet(self.decksize, self.nbets, self.bet_options, dim=8)
				curr_valnet = self.val_nets[i]
				curr_memory = self.m_v[i]
				curr_optim = self.val_net_optims[i]
				loss = torch.nn.MSELoss()

				# print("------------------------------------------")
				# print("Iteration: {}, Player: {}".format(t, i))
				# print("Value memory size: {}".format(len(curr_memory)))
				# print("m_v0 size: {}".format(len(self.m_v[0])))
				# print("m_v1 size: {}".format(len(self.m_v[1])))
				# print("Strategy memory size: {}".format(len(self.m_pi)))
				# for (infoset, timestep, action_probs) in self.m_pi:
				# 	print("**************************")
				# 	print("\CFR Iteration: {}".format(timestep))
				# 	print("\tInfoset: {}".format(infoset))
				# 	print("\tAction probs: {}".format(action_probs))	
				
				# for (infoset, timestep, regrets) in curr_memory:
				# 	print("**************************")
				# 	print("\CFR Iteration: {}".format(timestep))
				# 	print("\tInfoset: {}".format(infoset))
				# 	print("\tRegrets: {}".format(regrets))	
				
#				for i in range(10000):
				print("VALUE NETWORK TRAINING FOR PLAYER {}".format(i))
				for s in range(10): #sgd iterations
					batch_loss_history = []
					#print(len(curr_memory))
					print('sgd iteration: {}'.format(s))
					for (n, mem) in enumerate(curr_memory):
						infoset, timestep, regrets = mem
						cards, history = infoset
						bets = -torch.ones(self.nbets)
						for (a, b) in enumerate(history):
							bets[a] = b / (sum(history[:a]) + 2)
						#print('training bets', bets)
						valnet_out = curr_valnet.forward(torch.tensor([[cards]], dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
						# print('training valnet out', valnet_out)
						# print('training regrets', regrets)
						# print('**********************')
						# print('timestep: ', timestep)
						# print('infoset: ', infoset)
						# print('valnet_out: ', valnet_out)
						# print('regrets: ', regrets)

						stone = loss(valnet_out, torch.tensor([regrets]).float()) #* timestep
						# print('appended to batch loss history: ', stone)
						# if np.isnan(stone.detach()):
						# 	print("Stone is nan")
						# 	print("Valnet out: {}".format(valnet_out))
						# 	print("Regrets: {}".format(regrets))
						# 	exit(1)

						# print('Batch loss appending to history: ', stone)
						batch_loss_history.append(stone)
						if n % self.batch_size == 0 and n > 0:
							#print(n)
							#print('batch loss history', batch_loss_history)
							#print('batch loss history length: ', len(batch_loss_history))
							batch_loss_history_mean = torch.stack(batch_loss_history).mean()
							# print('BATCH LOSS HISTORY MEAN', batch_loss_history_mean.item())
							losses.append(batch_loss_history_mean.item())
							# print("Update number: {}, Loss = {}".format(n/self.batch_size, batch_loss_history))
							curr_optim.zero_grad()
							batch_loss_history_mean.backward()
							curr_optim.step()
							batch_loss_history = []
		print('m_pi')

		# save mpi to a file
		with open('m_pi_history.txt', 'w+') as f:
			for (infoset, timestep, action_probs, player) in self.m_pi:
				f.write("{}, {}\n{}\n{}\n\n".format(player, timestep, infoset, action_probs))
		
		for p in range(10):
			batch_loss_history = []
			#print(len(self.m_pi))
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
					print('BATCH LOSS HISTORY POLICY: ', batch_loss_history)
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
			if (history[-1] == history[-2] == 0) or (history[-1] == history[-2] == 1): #check check or bet call, go to showdown
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
		infoset = cards[acting_player], history
		infoset_str = str(cards[acting_player]) + ''.join(str(history))

		if infoset_str not in self.nodes:
			self.nodes[infoset_str] = Node(num_actions)

		bets = -torch.ones(self.nbets)
		for (i, b) in enumerate(history):
			# bets should be a list of the proportion each bet is of the pot size
			bets[i] = b / (sum(history[:i]) + 2)

		nodes_touched += 1

		if acting_player == traversing_player:
			util = np.zeros(self.bet_options) #2 actions
			node_util = 0
			advantages = self.val_nets[acting_player].forward(torch.tensor([[cards[acting_player]]], dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
			
			
			strategy = self.get_strategy(advantages)
			# print('TRAVERSING PLAYER')
			# print('acting player cards: ', cards[acting_player])
			# print('bet history: ', bets)
			# print('advantages: ', advantages)
			# print('strategy: ', strategy)
			for a in range(num_actions):
				#print('a is: {}'.format(a))
				next_history = history + [a]
				pot += a
				util[a] = self.deep_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)
				node_util += strategy[a] * util[a]

			action_advantages = np.zeros(num_actions)
			for a in range(num_actions):
				action_advantages[a] = util[a] - node_util
			#print("in deep cfr", t)
			print('************************')
			print('infoset of situation: ', infoset)
			print('action advantages from valnet: ', advantages)
			print('action advantages when acting player == traversing player: ', action_advantages)
			self.m_v[traversing_player].append((infoset, t, action_advantages))
			return node_util

		else: #acting_player != traversing_player
			advantages = self.val_nets[acting_player].forward(torch.tensor([[cards[acting_player]]],  dtype=torch.float), torch.tensor(bets).float().unsqueeze(0))
			strategy = self.get_strategy(advantages)
			# print('NON-TRAVERSING PLAYER')
			# print('acting player cards: ', cards[acting_player])
			# print('bet history: ', bets)
			# print('advantages: ', advantages)
			# print('strategy: ', strategy)
			action_probs = np.zeros(num_actions)
			for a in range(num_actions):
				#self.nodes[infoset].strategy_sum[a] += strategy[a]
				action_probs[a] = strategy[a]
				#insert infoset, t, strategies into strategy memory
			self.m_pi.append((infoset, t, action_probs, acting_player))
			# print('ITERATION: ', t)
			# print('M_PI: ', self.m_pi)
			util = 0
			if random.random() < strategy[0]:
				next_history = history + [0]
			else: 
				next_history = history + [1]
				pot += 1
			return self.deep_cfr(cards, next_history, pot, nodes_touched, traversing_player, t)


if __name__ == "__main__":
	k = KuhnCFR(100, 3, 0)
	k.cfr_iterations_deep()