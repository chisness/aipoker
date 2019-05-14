import numpy as np
import random

#TO DO
#Leduc with random agent and player agent
#Keep track of profit
#Add Kuhn
#Check setup
#Add CFR Chance Sampling
#Add other CFR variations
#Add NL Leduc
#Think about more interesting small games
#Add BRF
#Add DS
#Add support for 3 players

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

class LimitLeduc
	def __init__(self, dealer_player = 0, players = [Human('p1'), Human('p2')], ante = 1):
		#super().__init__()
		self.dealer_player = dealer_player
		self.num_players = len(players) #should be 2 or 3
		self.pot = num_players * ante 
		self.round = 1
		self.r1history = []
		self.r2history = []
		self.cards = [1,1,2,2,3,3]
		random.shuffle(self.cards)
		if self.num_players == 2:
			self.player0cards = self.cards[1 - dealer_player]
			self.player1cards = self.cards[dealer_player]

	def round_1():
		betsize = 2
		print(r1history)
		plays = len(history1)
		self.acting_player = (plays+dealer_player)%2 #first hand should be the non-dealer player
		self.pot = (num_players * ante) + betsize*r1history.count('b') + betsize*r1history.count('c') + 2*betsize*r1history.count('r')

		if r1history[-1:] == 'f':
			profit = (self.pot - betsize) / 2
			players[acting_player].profit += profit
			players[1-acting_player].profit -= profit
			print('End of game! Player {} wins pot of {} (profits {}\n'.format(self.acting_player,self.pot,profit))

		if r1history[-2:] == [k,k] or r1history[-1:] == [c]:
			round_2()

		print('\nPot size: {}'.format(self.pot))
		print('Player {} turn to act'.format(self.acting_player))
		bet = players[acting_player].select_move(valid_bets())
		self.pot += #or make this auto calculate from the history?
		self.r1history.append(bet)
		print('Action: {}'.format(bet))

		round_2()


	def round_2():
		betsize = 4
		self.round = 2
		print(r1history, r2history)
		plays = len(history2)
		self.acting_player = dealer_player 
		self.flop_card = cards[num_players]
		print('\n Dealing flop card: {}.format(self.flop_card)')

		if r2history[-1:] == [f]:
			profit = (self.pot - betsize) / 2
			players[acting_player].profit += profit
			players[1-acting_player].profit -= profit
			print('End of hand! Player {} wins pot of {} (profits {})\n'.format(self.acting_player, self.pot, profit))

		if r2history[-2:] == [k,k] or r2history[-1:] == [c]:
			winner = evaluate_hands()
			if winner == -1:
				print('Tie game! Pot size {} and both players had {}\n'.format(self.pot, self.player0cards.append(self.flop_card)))
			else:
				profit = pot / 2 
				print('End of hand! Player {} wins pot of {} (profits {})\n'.format(winner, self.pot, profit))
				players[winner].profit += profit
				players[1-winner].profit -= profit
		
		print('\nPot size: {}'.format(self.pot))
		print('Player {} turn to act'.format(self.acting_player))
		bet = players[acting_player].select_move(valid_bets())
		print('Action: {}'.format(bet))
		self.r2history.append(bet)
		round_2()


	def evaluate_hands():
		if self.player0cards == self.flop_card:
			return 0
		elif:
			self.player1cards = self.flop_card:
			return 1
		elif:
			self.player0cards > self.player1cards:
			return 0
		elif:
			self.player1cards > self.player0cards:
			return 1
		else:
			return -1

	def make_bet():
		players[self.acting_player].select_move(valid_bets())

	def valid_bets():
		bet_options = ['f', 'k', 'c', 'b', 'r'] #fold check call bet raise

		if self.round == 1:
			history = self.r1history
		else:
			history = self.r2history

		if history[-1:] == 'b' or history[-1:] == 'r':
			betoptions[3] = False #Bet invalid -- Can only bet at the start of a round or after a check
			betoptions[1] = False #Check invalid -- Can only check at the start of a round or after a check

		if history[-1:] != 'b':
			betoptions[4] = False #Raise invalid -- Can only raise after a bet (and only one raise per round)

		if history[-1:] == 'k' or history[-1:] == '': #Previous action check or none
			betoptions[0] = False #Fold invalid -- Can only fold after a bet or raise
			betoptions[2] = False #Call invalid -- Can only call after a bet or raise

		return bet_options

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
	def __init__(self, name):
		self.profit = 0
		self.name = name

# class RandomAgent(Player):
# 	def select_move(self, game):
# 		return random.choice(game.valid_bets())

class Human(Player):
	def select_move(betoptions):
		print('Enter your bet')
		if betoptions[0]: print('FOLD: f')
    	if betoptions[1]: print('CHECK: k')
    	if betoptions[2]: print('CALL {}: c'.format(betsize))
    	if betoptions[3]: print('BET {}: b'.format(betsize))
    	if betoptions[4]: print('RAISE {}: r'.format(betsize*2))
		input(move)
		if move in game.valid_bets():
			return move
		else:
			print('Invalid action, try again')
			select_move(game)



# class CFRAgent: 
# 	def __init__(self, cfr_type):
# 		self.cfr_type = cfr_type

# class DSAgent:
# 	def __init__(self):

# class BRF:
# 	def __init__(self, tree):

if __name__ == "__main__":
	p1 = Human('p1')
	p2 = Human('p2')
	for i in range(hands):
		p = LimitLeduc(dealer_player = i%2, players = [p1, p2])
		p.round_1()