import random
from kuhn3p import betting, deck, Player

class Bluffer(Player):
	def __init__(self, bluff, rng=random.Random()):
		assert bluff >= 0 and bluff <= 1

		self.bluff = bluff
		self.rng   = rng 

	def act(self, state, card):
		if betting.can_bet(state):
			if card < deck.ACE:
				if self.rng.random() < self.bluff:
					return betting.BET
				else:
					return betting.CHECK
			else:
				return betting.BET
		else:
			if card == deck.ACE:
				return betting.CALL
			else:
				return betting.FOLD

	def __str__(self):
		return 'Bluffer(bluff=%f)' % (self.bluff)
