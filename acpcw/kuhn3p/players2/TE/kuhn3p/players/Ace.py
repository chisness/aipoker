import random
from kuhn3p import betting, deck, Player

# A study on reinforcement Learning:
# http://studentnet.cs.manchester.ac.uk/resources/library/3rd-year-projects/2015/yifei.wang-6.pdf

class Ace(Player):
        
	def __init__(self, bluff=0.75, rng=random.Random()):
                assert  0 <= bluff  <= 1
                
                self.bluff = bluff
		self.rng   = rng

        # Always bet on Ace. Always fold on Jack
	def act(self, state, card):
                action = 0
                
		if betting.can_bet(state):
                        if card == deck.ACE:
                                action = betting.BET
                        elif card == deck.JACK:
                                action = betting.CALL
			else:
				if self.rng.random() > self.bluff:
					action = betting.BET
				else:
					action = betting.CHECK
		else:
			if card == deck.ACE:
				action = betting.CALL
			elif card == deck.JACK:
				action = betting.FOLD
			else:
                                if self.rng.random() > self.bluff:
                                        action = betting.CALL
                                else:
                                        action = betting.FOLD
                return action
                

	def __str__(self):
		return 'Always on ACE(bluff=%f' % self.bluff
