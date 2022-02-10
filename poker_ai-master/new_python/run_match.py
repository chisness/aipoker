from kuhn3p import deck, dealer, players
from random import Random
from collections import defaultdict

opp1_counter = defaultdict(float)
opp2_counter = defaultdict(float)
#format is 'our_card 0/1/2/3' + 'our_position 0/1/2' + 'other opponent card 0/1/2/3/n' + hand_history use b/p' + 'showdown card 0/1/2/3/n' + 'opponent_action use b/p'
#values are tuple of (pass, bet) counters
#add other player showdown card
#do we need our position (also implies other player's position)?

rng = Random()
rng.seed(337)  # each seed corresponds to a different set of hands

num_hands = 2

all_players = [players.Bluffer(0.7), players.Chump(0.1, 0.1, 0.0), players.Bluffer(0.5)]#players.Kevin()]#, players.AgentSouth, players.Bayes, players.EFS, players.Kevin,
# 				players.Shark, players.TE, players.Thief, players.Ultimate, players.Nash]

the_players = [all_players[0], all_players[1], all_players[2]]

total = [0, 0, 0]

#assume we are player 0
#get each other player action sequence with optional parameter for showdown

history_counter = {}
#our_agent = ___ #set this to our agent name 


for hand in range(num_hands):
	first          = hand % 3
	second         = (first + 1) % 3
	third          = (second + 1) % 3

	if first == 0: #assume we are the_players[0]
		our_position = 0
	elif second == 0:
		our_position = 1
	elif third == 0:
		our_position = 2

	print('our position', our_position)

	this_players   = [the_players[first], the_players[second], the_players[third]]

	print('this players', this_players)
	#c = deck.shuffled(rng)
	#print('c', c)

	#our_position = this_players.index(the_players[0])

	(state, delta, our_card, p2a, p2s, p2sd, p3a, p3s, p3sd) = dealer.play_hand(this_players, deck.shuffled(rng), our_position)
	#(state, delta) = dealer.play_hand(this_players, deck.shuffled(rng))
	for i in range(3):
		total[(first + i)%3] += delta[i]
	
	#opponent 1
	for i in range(len(p2a)):
		opp1_counter['our_card' + 'our_position' + 'p3sd' + 'p2s[i]' + 'p2sd' + 'p2a[i]'] += 1 

	#opponent 2
	for i in range(len(p3a)):
		opp2_counter['our_card' + 'our_position' + 'p2sd' 'p3s[i]' + 'p3sd' + 'p3a[i]'] += 1 

for i in range(3):
	print(the_players[i], total[i])

# print(f'Our card: {our_card}')
# print(f'Player 2 states: {p2s}')
# print(f'Player 2 actions: {p2a}')
# print(f'Player 2 showdowns: {p2sd}')
# print(f'Player 3 states: {p3s}')
# print(f'Player 3 actions: {p3a}')
# print(f'Player 3 showdowns: {p3sd}')

print(opp1_counter)
print(opp2_counter)
