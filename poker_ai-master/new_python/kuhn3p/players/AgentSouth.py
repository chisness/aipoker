#created by Jorge Cura and Carlos Lores

import random
from kuhn3p import betting, deck, Player

class AgentSouth(Player):
	def __init__(self, bluff, rng=random.Random()):
		assert bluff >= 0 and bluff <= 1

		self.bluff = bluff
		self.rng   = rng 

	def act(self, state, card):
		if betting.can_bet(state):
			if card == deck.ACE:
				    return betting.BET

			#randomly bet or check
			elif card == deck.KING:
				if self.rng.random() < self.bluff:
					return betting.BET
				else:
					return betting.CHECK

			elif card == deck.QUEEN:
					return betting.CHECK
			elif card == deck.JACK:
					return betting.CHECK

		else:
			if card == deck.ACE:
				return betting.CALL
			#randomly call or fold
			elif card == deck.KING:
				if self.rng.random() < self.bluff:
					return betting.CALL
				else:
					return betting.FOLD
			elif card == deck.QUEEN:
					return betting.FOLD
			elif card == deck.JACK:
					return betting.FOLD


	def __str__(self):
		return 'Agent(bluff=%f)' % (self.bluff)
