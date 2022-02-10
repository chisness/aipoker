import random
from kuhn3p import betting, deck, Player

class Nash(Player):
	def __init__(self, rng=random.Random()):
                self.bluff = bluff
		self.rng   = rng 

        # Nash equilibrium
	def act(self, state, card):
		if betting.can_bet(state):
                        if card == deck.ACE:
                                return betting.BET
                        elif card == deck.JACK:
                                return betting.CALL
			else:
				if self.rng.random() > self.bluff:
					return betting.BET
				else:
					return betting.CHECK
		else:
			if card == deck.ACE:
				return betting.CALL
			elif card == deck.JACK:
				return betting.FOLD
			else:
                                if self.rng.random() > self.bluff:
                                        return betting.CALL
                                else:
                                        return betting.FOLD

	def __str__(self):
		return 'Always on ACE(bluff=%f' % self.bluff
