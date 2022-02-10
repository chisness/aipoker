import random
from kuhn3p import betting, deck, Player

class Shark(Player):
	def __init__(self):
		self.actionProbabilities = []

		# Randomly pick between a complex and a simple nash strategy profile.
		# Such a strategy may make it tougher for other players to learn from repeated play.
		# Probability of picking simple strategy = 1/4
		# Probability of picking complex strategy = 1 - 1/4
		if random.randrange(3) >= 1:
			self.buildNashProfile()
		else:
			self.buildNashProfile("Simple")

	def act(self, state, card):
		p = 0
		if betting.facing_bet_call(state):
			p = self.actionProbabilities[betting.actor(state)][card][3]
		elif betting.facing_bet_fold(state):
			p = self.actionProbabilities[betting.actor(state)][card][2]
		elif betting.facing_bet(state):
			p = self.actionProbabilities[betting.actor(state)][card][1]
		else:
			p = self.actionProbabilities[betting.actor(state)][card][0]

		if p > 1 - p:
			if betting.can_bet(state):
				return betting.BET
			else:
				return betting.CALL

		else:
			if betting.can_bet(state):
				return betting.CHECK
			else:
				return betting.FOLD
			

	def buildNashProfile(self, type="Complex"):
		for p in range(3):
			self.actionProbabilities.append([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])	# for each player, store arrays of cards, each with aggressive action probabilites for each situation
		for p in range(3):
			if p == 0:
				for k in range(4):
					self.actionProbabilities[p][deck.JACK][k] = 0
					self.actionProbabilities[p][deck.QUEEN][k] = 0
					self.actionProbabilities[p][deck.KING][k] = 0.5 if k == 2 else 0
					self.actionProbabilities[p][deck.ACE][k] = 1 if k != 0 else 0

			if p == 1:
				for k in range(4):
					self.actionProbabilities[p][deck.JACK][k] = round(random.uniform(0, 0.25), 2) if k == 0 else 0

					b11 = self.actionProbabilities[p][deck.JACK][0]
					if k == 0:
						if type == "Simple":
							self.actionProbabilities[p][deck.QUEEN][k] = round(random.uniform(0, 0.25), 2) 
							self.actionProbabilities[p][deck.JACK][k] = round(random.uniform(0, self.actionProbabilities[p][deck.QUEEN][k]))
							b11 = self.actionProbabilities[p][deck.JACK][0]
						else:
							self.actionProbabilities[p][deck.QUEEN][k] = round(random.uniform(0, b11), 2) if b11 <= (1/6) else round(random.uniform(0, 0.5 - 2 * b11), 2) 
					b21 = self.actionProbabilities[p][deck.QUEEN][0]
					if k == 1:
						if type == "Simple":
							self.actionProbabilities[p][deck.KING][k] = round(random.uniform(0, (2 + 3 * b11 + 4 * b21)/4))
						else:
							self.actionProbabilities[p][deck.KING][k] = round(random.uniform(0, (2 + 4 * b11 + 3 * b21)/4))
					if k == 2:
						self.actionProbabilities[p][deck.KING][k] = (1 + b11 + 2 * b21) / 2 
						if type == "Simple":
							self.actionProbabilities[p][deck.QUEEN][k] = 0
						else:
							self.actionProbabilities[p][deck.QUEEN][k] = round(random.uniform(0, (b11 - b21) / 2 * (1 - b21)))
					else:
						self.actionProbabilities[p][deck.QUEEN][k] = 0
						self.actionProbabilities[p][deck.KING][k] = 0 

					self.actionProbabilities[p][deck.ACE][k] = 1 if k != 0 else (2 * b11 + 2 * b21)


			if p == 2:
				for k in range(4):
					if type == "Simple":
						self.actionProbabilities[p][deck.JACK][k] = 0
						self.actionProbabilities[p][deck.QUEEN][k] = 0 if k != 0 else 0.5
					else:
						self.actionProbabilities[p][deck.JACK][k] = 0.5 if k == 0 else 0
						self.actionProbabilities[p][deck.QUEEN][k] = 0

					self.actionProbabilities[p][deck.ACE][k] = 1
					
					if k == 2:
						b11 = self.actionProbabilities[1][deck.JACK][0]
						b21 = self.actionProbabilities[1][deck.QUEEN][0]
						b32 = self.actionProbabilities[1][deck.KING][1]
						if type == "Simple":
							self.actionProbabilities[p][deck.KING][k] = round(random.uniform(0.5 - b32, 0.5 - b32 + (3 * b11 + 4 * b21) / 4))
						else:
							self.actionProbabilities[p][deck.KING][k] = round(random.uniform(0.5 - b32, 0.5 - b32 + (4 * b11 + 3 * b21) / 4))
					elif k == 3:
						self.actionProbabilities[p][deck.KING][k] = round(random.uniform(0, 1.0))
					else:
						self.actionProbabilities[p][deck.KING][k] = 0 

	def __str__(self):
		return "The Shark"	
