import random
from kuhn3p import betting, deck, Player

class Kevin(Player):
	def __init__(self):
		self.counter = 0
		
	def act(self, state, card):
		self.counter = self.counter + 1
		if betting.can_bet(state):
    		    if card == deck.ACE:
		            return betting.BET
                elif card == deck.KING:
                    if(random.Random() < 1/2):
                        return betting.BET
                    else:
                        return betting.CHECK
                elif card == deck.QUEEN:
                    if(random.Random() < 1/6):
		                return betting.BET
                    else:
                        return betting.CHECK
		        
		else:
                    if card == deck.ACE:
	    		return betting.CALL
                    elif card == deck.KING:
                        if(self.counter % 2 == 0):
                            return betting.CALL
                        else:
                            return betting.FOLD
                    elif card == deck.QUEEN:
                        if(random.Random() < 1/25):
		    	    return betting.CALL
                        else:
                            return betting.FOLD
		    else:
			return betting.FOLD


	def __str__(self):
		return 'Kevin The Rubber Duck'
