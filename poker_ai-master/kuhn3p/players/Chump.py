import random
from kuhn3p import betting, Player

class Chump(Player):
	def __init__(self, bet, call, fold, rng=random.Random()):
		assert bet >= 0
		assert call >= 0
		assert fold >= 0
		assert call + bet > 0
		assert call + fold > 0

		self.p1  = call / (0.0 + call + bet)
		self.p2  = call / (0.0 + call + fold)
		self.rng = rng 
		self.player = -1

	def start_hand(self, position, card):
		self.player = position

	def act(self, state, card, node = None):
		if betting.can_bet(state):
			p = self.p1
		else:
			p = self.p2

		dart = self.rng.random()
		if betting.can_bet(state):
			# print('chump state checked:', state, self.player )		
			return betting.CHECK
		else:
			# print('chump state call:', state, self.player )					
			return betting.CALL

	def __str__(self):
		return 'Chump(bet=%f,fold=%f)' % (1 - self.p1, 1 - self.p2)
