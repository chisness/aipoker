import numpy as np

class Poker:
	def __init__(self, gameType):
		self.gameType = gameType
		self.cards = []
		self.currentPlayer = 1
		self.playerOneCards = []
		self.playerTwoCards = []
		self.setupGame()
		self.playGame()
		nolimit
		limit
		limit bet sizes

	def setupGame():

	def playGame():
		if self.gameType=='l':

class Node: 
	def __init__(self, regretSum, strategy, strategySum, num_actions):
		self.regretSum = np.zeros(num_actions)
		self.strategy = np.zeros(num_actions)
		self.strategySum = np.zeros(num_actions)


	def getStrategy():

	def getAverageStrategy():


class CFR:
	def __init__(self, cfr_type):
		self.cfr_type = cfr_type

class DeepStack:
	def __init__(self):


class Player:
	def __init__(self, n):
		self.

class Board:
	def __init__(self):


class Cards:
	def __init__(self, game):
		self.decksize = 

class BRF:
	def __init__(self, tree):

class Evaluate:
	def __init__(self, cards):
		self.cards = cards

	if len(cards[0]==1):
		if cards[0]>cards[1]:
			return 0
		else:
			return 1

	else:
		if cards[0][0] == cards[0][1]:
			return 0
		elif:
			cards[1][0] == cards[1][1]:
			return 1
		else:
			if cards[0][0] > cards[]


class Kuhn:
	def __init__(self, decksize):
		self.cards = np.arange(decksize)
		self.rounds = 1
		self.ante = 1
		self.max_bets = 1
		self.r1_betsize = 1

class Leduc:
	def __init__(self):
		self.cards = [1,1,2,2,3,3]
		self.rounds = 2
		self.ante = 1
		self.max_bets = 4
		self.r1_betsize = 2
		self.r2_betsize = 4


if __name__ == "__main__":

			