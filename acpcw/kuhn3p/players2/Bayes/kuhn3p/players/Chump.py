import random
import kuhn3p

class Chump(kuhn3p.Player):
	def __init__(self, bet, call, fold, rng=random.Random()):
		assert bet >= 0
		assert call >= 0
		assert fold >= 0
		assert call + bet > 0
		assert call + fold > 0

		self.p1  = call / (0.0 + call + bet)
		self.p2  = call / (0.0 + call + fold)
		self.rng = rng 

	def act(self, state, card):
		if kuhn3p.betting.can_bet(state):
			p = self.p1
		else:
			p = self.p2

		dart = self.rng.random()
		if dart < p:
			return 0
		else:
			return 1

	def __str__(self):
		return 'Chump(bet=%f,fold=%f)' % (1 - self.p1, 1 - self.p2)
