from kuhn3p import deck, dealer, players
from random import Random

rng         = Random()
rng.seed(31337)  # each seed corresponds to a different set of hands

num_hands   = 50
the_players = [ players.Bluffer(0.2), players.Custom(), 
    players.Ace(0.1),  ]

total = [0, 0, 0]
for hand in range(num_hands):
	first          = hand % 3
	second         = (first + 1) % 3
	third          = (second + 1) % 3

	this_players   = [the_players[first], the_players[second], the_players[third]]

	(state, delta) = dealer.play_hand(this_players, deck.shuffled(rng))
	for i in range(3):
		total[(first + i)%3] += delta[i]

for i in range(3):
	print the_players[i], total[i]
