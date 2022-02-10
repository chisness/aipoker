from kuhn3p import deck, dealer, players, brf
from random import Random
from collections import defaultdict
from itertools import permutations

opp1_counter = defaultdict(float)
opp2_counter = defaultdict(float)
#format is 'our_card 0/1/2/3' + 'our_position 0/1/2' + 'other opponent card 0/1/2/3/n' + hand_history use b/p' + 'showdown card 0/1/2/3/n' + 'opponent_action use b/p'
#values are tuple of (pass, bet) counters
#add other player showdown card
#do we need our position (also implies other player's position)?

rng         = Random()
#rng.seed(313322)  # each seed corresponds to a different set of hands

# all_players = [players.Bluffer(0.7), players.Chump(0.1, 0.1, 0.0), players.Bluffer(0.5)], players.Kevin(), players.AgentSouth, players.Bayes, players.EFS, players.Kevin,
# players.Shark, players.TE, players.Thief, players.Ultimate, players.Nash]

history_counter = {}

#all_players = [1, 2, 3, 4, 5]
# all_players = [players.Bluffer(0.7), players.Chump(0.1, 0.1, 0.0), players.Bluffer(0.5), players.Bluffer(0.5), players.Bluffer(0.3)]

# Agent South
# Bayes
# EFS
# Electric Caviar Racers
# Juan
# Kevin
# KuhnShark
# NeuralNetPoker (ignore)
# TE
# Thief
# UltimateAIKuhn


# Nash 

#duplicate matches with winrates and standard errors
#just pass in string

all_players = [players.Bluffer(0.7), players.Chump(0.1, 0.1, 0.0), players.AgentSouth()]

matches = permutations(all_players, 3) 
agent_matches = []
other_matches = []
for i in matches:
	if all_players[0] in i:
		agent_matches.append(i)
	else:
		other_matches.append(i)
print(len(agent_matches))

num_hands = 10
#winrates, standard 
#same seed with different positions

for m in agent_matches: 
	print('new agent match')
	print('agents: ', m)
	match_players = m
	for s in range(3):
		rng.seed(s) #each match plays with 3 random seeds
		total = [0, 0, 0]
		for hand in range(num_hands):
			first = hand % 3
			second = (first + 1) % 3
			third = (second + 1) % 3
			this_players = [match_players[first], match_players[second], match_players[third]]
			our_position = this_players.index(all_players[0])
			(state, delta, our_card, p2a, p2s, p2sd, p3a, p3s, p3sd) = dealer.play_hand(this_players, deck.shuffled(rng), our_position)
			for i in range(3):
				total[(first + i)%3] += delta[i]
			#opponent 1
			for i in range(len(p2a)):
				opp1_counter['our_card' + 'our_position' + 'p3sd' + 'p2s[i]' + 'p2sd' + 'p2a[i]'] += 1 
			#opponent 2
			for i in range(len(p3a)):
				opp2_counter['our_card' + 'our_position' + 'p2sd' 'p3s[i]' + 'p3sd' + 'p3a[i]'] += 1 
		for i in range(3):
			print(match_players[i], total[i])


for m in other_matches:
	print('new non-agent match')
	print('agents: ', m)
	match_players = m
	for s in range(3):
		rng.seed(s) #each match plays with 3 random seeds
		total = [0, 0, 0]
		for hand in range(num_hands):
			first = hand % 3
			second = (first + 1) % 3
			third = (second + 1) % 3
			this_players = [match_players[first], match_players[second], match_players[third]]
			(state, delta, our_card, p2a, p2s, p2sd, p3a, p3s, p3sd) = dealer.play_hand(this_players, deck.shuffled(rng))
			for i in range(3):
				total[(first + i)%3] += delta[i]
		for i in range(3):
			print(match_players[i], total[i])

			 


# the_players = [players.Chump(0.99, 0.01, 0.0), 
#     players.Chump(0.95, 0.05, 0.0), players.Bluffer(0.5) ]

# total = [0, 0, 0]
# for hand in range(num_hands):
# 	print('HAND ', hand)
# 	first          = hand % 3
# 	second         = (first + 1) % 3
# 	third          = (second + 1) % 3

# 	print(first, second, third)
# 	this_players   = [the_players[first], the_players[second], the_players[third]]

# 	(state, delta) = dealer.play_hand(this_players, deck.shuffled(rng))
# 	print(delta)
# 	for i in range(3):
# 		total[(first + i)%3] += delta[i]

# 	print(total)

# for i in range(3):
# 	print(the_players[i], total[i])