import random
from kuhn3p import betting, deck, Player

class EFS(Player):

	def act(self, state, card, bets):
		if (betting.can_bet(state)):
			if (card < deck.ACE):
				if (card == deck.KING):
					if (random.Random() < 2/3):
						return betting.BET
					else:
						return betting.CHECK
					
				if (card == deck.QUEEN):
					if (random.Random() < 1/3):
						return betting.BET
					else:
						return betting.CHECK
								
				#only bet on Jack if both players have folded
				if (card == deck.JACK):
					folds = bets.count('f')
				if folds > 1:
					return betting.BET
					
			#always bet on an ACE
			return betting.BET
		        
		else:
			if (card < deck.KING):
	    	    if (random.Random() < 1/3):
			    	return betting.CALL
				else:
			   		return betting.FOLD
			else:
				#call on an Ace or King
				return betting.CALL