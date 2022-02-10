from kuhn3p import betting, deck

def winner(state, cards):
	assert betting.is_terminal(state)
	if betting.is_showdown(state):
		print('showdown true')
		best_player = -1
		best_card   = -1
		for i in range(3):
			if betting.at_showdown(state, i) and cards[i] > best_card:
				best_player = i
				best_card   = cards[i] 
		return best_player
	else:
		return betting.bettor(state)


player_2_actions = []
player_2_sd = -1
player_3_actions = []
player_3_sd = -1
our_card = []
p2_position = -1
p3_position = -1
player_2_states = []
player_3_states = []

# def play_hand(players, cards, our_position):
# 	state = betting.root()
# 	hand_history = []

# 	for i in range(3):
# 		players[i].start_hand(i, cards[i])

# 	our_position = our_position
# 	our_card = cards[our_position]

# 	while not betting.is_terminal(state):
# 		player = betting.actor(state)
# 		action = players[player].act(state, cards[player])
# 		state  = betting.act(state, action)
# 		print('player', player)
# 		print('action', action)
# 		print('state', state)
# 		print('term', betting.is_terminal(state))

# 	shown_cards = [cards[i] if betting.at_showdown(state, i) else None for i in range(3)]
# 	print('shown cards', shown_cards)

# 	for i in range(3):
# 		players[i].end_hand(i, cards[i], state, shown_cards)

# 	the_winner = winner(state, cards)
# 	print('the winner', the_winner)
# 	pot_size   = betting.pot_size(state)

# 	return (state, [pot_size*(i == the_winner) - betting.pot_contribution(state, i) for i in range(3)])

def play_hand(players, cards, our_position=False):
	state = betting.root()
	hand_history = []

	for i in range(3):
		players[i].start_hand(i, cards[i])

	our_position = our_position
	our_card = cards[our_position]

	# print('our position', our_position)
	# print('our card', our_card)
	#our_card.append(cards[our_position])
	if our_position == 0:
		p2_position = 1
		p3_position = 2
	elif our_position == 1:
		p2_position = 2
		p3_position = 0
	elif our_position == 2:
		p2_position = 0
		p3_position = 1

	while not betting.is_terminal(state):
		player = betting.actor(state)
		print('player', player)
		action = players[player].act(state, cards[player])
		print('action', action)

		if not hand_history:
			if action == 0:
				hand_history.append('p')
			elif action == 1:
				hand_history.append('b')
		elif len(hand_history) == 1:
			if hand_history[-1] == 'b':
				if action == 0:
					hand_history.append('b')
				elif action == 1:
					hand_history.append('p')
			elif hand_history[-1] == 'p':
				if action == 0:
					hand_history.append('p')
				elif action == 1:
					hand_history.append('b')
		else:
			if hand_history[-1] == 'b' or hand_history[-2] == 'b':
				if action == 0:
					hand_history.append('b')
				elif action == 1:
					hand_history.append('p')
			else:
				if action == 0:
					hand_history.append('p')
				elif action == 1:
					hand_history.append('b')

		if player == p2_position:
			player_2_actions.append(action)
			player_2_states.append(hand_history)
		elif player == p3_position:
			player_3_actions.append(action)
			player_3_states.append(hand_history)

		print('game state', state)
		state  = betting.act(state, action)
		print('term', betting.is_terminal(state))

	shown_cards = [cards[i] if betting.at_showdown(state, i) else None for i in range(3)]
	#make sure that it's showing only showdown cards by everyone who doesn't fold
	player_2_sd = shown_cards[p2_position]
	if player_2_sd is None:
		player_2_sd = 'n'
	player_3_sd = shown_cards[p3_position]
	if player_3_sd is None:
		player_3_sd = 'n'

	for i in range(3):
		players[i].end_hand(i, cards[i], state, shown_cards)

	the_winner = winner(state, cards)
	pot_size   = betting.pot_size(state)

	print(cards)
	print(hand_history)
	print(pot_size)
	print(the_winner)

	return (state, [pot_size * (i == the_winner) - betting.pot_contribution(state, i) for i in range(3)], our_card, player_2_actions, player_2_states, player_2_sd, player_3_actions, player_3_states, player_3_sd)
	# return (state, [pot_size*(i == the_winner) - betting.pot_contribution(state, i) for i in range(3)])
