import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CardEmbedding(nn.Module):
	def __init__(self, dim):
		super(CardEmbedding, self).__init__()
		self.rank = nn.Embedding(13, dim)
		self.suit = nn.Embedding(4, dim)
		self.card = nn.Embedding(52, dim)

	def forward(self, input):
		print('HELLO FORWARD')
		print('INPUT', input)
		print('INPUT TYPE', type(input))
		print('INPUT SHAPE', input.shape)
		input = torch.tensor(input)
		B, num_cards = input.shape
		x = input.view(-1)
		print('x', x)

		valid = x.ge(0).float() #-1 means no card
		x = x.clamp(min=0)

		embs = self.card(x) + self.rank(x // 4) + self.suit(x % 4)
		embs = embs * valid.unsqueeze(1) #zero out ' no card' embeddings

		#sum across the cards in the hole/board
		return embs.view(B, num_cards, -1).sum(1)


class DeepCFRNet(nn.Module):
	def __init__(self, ncardtypes, nbets, nactions, dim = 128):
		super(DeepCFRNet, self).__init__()

		self.card_embeddings = nn.ModuleList([CardEmbedding(dim) for _ in range(ncardtypes)])
		print(self.card_embeddings[5].rank.weight)
		#print(self.card_embeddings[19].rank.weight)

		card_embs = []
		# r1 = np.array([5,10])
		# r2 = np.array([3, 7, 19])
		# r3 = np.array([r1, r2])
		# B = 1
		r1 = np.array([[ 5., 10.]])
		r2 = np.array([[3, 7, 19]])
		# cards = np.zeros((B, len(r3)))
		cards = [r1, r2]
		for embedding, card_group in zip(self.card_embeddings, cards):
			print(embedding.rank.weight)
			print(card_group)
			#print(type(embedding), embedding(card_group))
			card_embs.append(embedding(card_group))
		card_embs = torch.cat(card_embs, dim = 1)

		print(card_embs)
		#print(card_embs.shape)

		self.card1 = nn.Linear(1, dim)
		self.card2 = nn.Linear(dim, dim)
		self.card3 = nn.Linear(dim, dim)

		self.bet1 = nn.Linear(nbets * 2, dim)
		self.bet2 = nn.Linear(dim, dim)

		self.comb1 = nn.Linear(2 * dim, dim)
		self.comb2 = nn.Linear(dim, dim)
		self.comb3 = nn.Linear(dim, dim)

		self.action_head = nn.Linear(dim, nactions)


if __name__ == "__main__":
	k = DeepCFRNet(52, 6, 10)
