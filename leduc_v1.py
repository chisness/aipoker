import numpy as np
import random

#TO DO
#Leduc with random agent and player agent
#Maximum bets per round
#Add CFR Chance Sampling
#Check setup
#Add Kuhn and make Poker class
#Add other CFR variations
#Add NL Leduc
#===================
#Get working with AAAI poker thing
#Get working with 2 players on different computers
#Think about more interesting small games
#Add BRF
#Add DS
#Add support for 3 players

class LimitLeduc:
	def __init__(self, players, dealer_player = 0, ante = 1):
		#super().__init__()
		self.dealer_player = dealer_player
		self.num_players = len(players) #should be 2 or 3
		self.pot = self.num_players * ante 
		self.round = 1
		self.r1history = []
		self.r2history = []
		self.cards = [1,1,2,2,3,3]
		random.shuffle(self.cards)
		if self.num_players == 2:
			self.player0cards = self.cards[1 - dealer_player]
			self.player1cards = self.cards[dealer_player]

	def play_hand(self, iterations):
		for i in range(iterations):
			self.dealer_player = iterations%2
			round_1()


	def round_1(self):
		betsize = 2
		print(self.r1history)
		plays = len(self.r1history)
		self.acting_player = (plays+self.dealer_player)%2 #first hand should be the non-dealer player

		if r1history[-1] == 'f':
			profit = (self.pot - betsize) / 2
			self.players[acting_player].chips += profit
			self.players[1-acting_player].chips -= profit
			print('End of game! Player {} wins pot of {} (profits {})\n'.format(self.acting_player, self.pot, profit))

		if self.r1history[-2:] == ['k', 'k'] or self.r1history[-1] == 'c':
			round_2()

		print('\nPot size: {}'.format(self.pot))
		print('Player {} turn to act'.format(self.acting_player))
		bet = self.players[acting_player].select_move(valid_bets(), betsize)
		print('Action: {}'.format(bet))

		if bet == 'c' or bet =='b':
			self.pot += betsize
		elif bet == 'r':
			self.pot += 2*betsize
		self.r1history.append(bet)

		round_1()

	def round_2(self):
		betsize = 4
		self.round = 2
		print(self.r1history, self.r2history)
		plays = len(self.r2history)
		self.acting_player = 1 - ((plays+dealer_player)%2)
		self.flop_card = self.cards[num_players]
		print('\n Dealing flop card: {}.format(self.flop_card)')

		if r2history[-1] == 'f':
			profit = (self.pot - betsize) / 2
			self.players[acting_player].profit += profit
			self.players[1-acting_player].profit -= profit
			print('End of hand! Player {} wins pot of {} (profits {})\n'.format(self.acting_player, self.pot, profit))

		if r2history[-2:] == ['k', 'k'] or r2history[-1] == 'c':
			winner = evaluate_hands()
			if winner == -1:
				print('Tie game! Pot size {} and both players had {}\n'.format(self.pot, self.player0cards.append(self.flop_card)))
			else:
				profit = pot / 2 
				print('End of hand! Player {} wins pot of {} (profits {})\n'.format(winner, self.pot, profit))
				self.players[winner].chips += profit
				self.players[1-winner].chips -= profit
		
		print('\nPot size: {}'.format(self.pot))
		print('Player {} turn to act'.format(self.acting_player))
		bet = self.players[acting_player].select_move(valid_bets(), betsize)
		print('Action: {}'.format(bet))
		self.r2history.append(bet)
		round_2()

	def evaluate_hands(self):
		if self.player0cards == self.flop_card:
			return 0
		elif self.player1cards == self.flop_card:
			return 1
		elif self.player0cards > self.player1cards:
			return 0
		elif self.player1cards > self.player0cards:
			return 1
		else:
			return -1

	def make_bet(self):
		players[self.acting_player].select_move(valid_bets())

	def valid_bets(self):
		bet_options = ['f', 'k', 'c', 'b', 'r'] #fold check call bet raise

		if self.round == 1:
			history = self.r1history
		else:
			history = self.r2history

		if history[-1] == 'b' or history[-1] == 'r':
			betoptions[3] = False #Bet invalid -- Can only bet at the start of a round or after a check
			betoptions[1] = False #Check invalid -- Can only check at the start of a round or after a check

		if history[-1] != 'b':
			betoptions[4] = False #Raise invalid -- Can only raise after a bet (and only one raise per round)

		if history[-1] == 'k' or history[-1] == '': #Previous action check or none
			betoptions[0] = False #Fold invalid -- Can only fold after a bet or raise
			betoptions[2] = False #Call invalid -- Can only call after a bet or raise

		return bet_options


class Player:
	def __init__(self, name, chips = 100):
		self.chips = chips
		self.name = name
		self.hands = 0
		self.profit = 0

class Human(Player):
	# def __init__(self, name, chips = 100):
	# 	pass
	def select_move(betoptions, betsize):
		print('Enter your bet')
		if betoptions[0]: print('FOLD: f')
		if betoptions[1]: print('CHECK: k')
		if betoptions[2]: print('CALL {}: c'.format(betsize))
		if betoptions[3]: print('BET {}: b'.format(betsize))
		if betoptions[4]: print('RAISE {}: r'.format(betsize*2))
		input(move)
		if move in betoptions:
			return move
		else:
			print('Invalid action, try again')
			select_move(betoptions, betsize)

class RandomAgent(Player):
	def select_move(betoptions, betsize):
		agent_options = []
		for i in betoptions:
			if betoptions[i] != False:
				agent_options.append(betoptions[i])
		return random.choice(options)

if __name__ == "__main__":
	p0 = Human('p0')
	p1 = Human('p1')
	players = [p0, p1]
	p = LimitLeduc(players = players, dealer_player = 0, ante = 1)
	p.play_hand(10)


