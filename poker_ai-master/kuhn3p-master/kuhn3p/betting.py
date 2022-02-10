#
# there are 4 betting situations for any player:
#   1) it is checked to them
#   2) it was raised by the person on their right
#   3) it was raised two to their right, and the middle player called
#   4) it was raised two to their right, and the middle player folded
#
# => there are 4*3 = 12 betting decisions states
#
# in situation 1, the player can bet or check.
# in situation 2, the player can call or fold.
#
# in situation 3 and 4, any action ends the betting
# in situation 1 for player 3, checking ends the betting
#
# => there are 2*2*3 + 1 = 13 terminal betting sequences
#

CHECK = 0
CALL  = 0
BET   = 1
FOLD  = 1

def num_internal():
	return 12

def num_terminals():
	return 13

def num_states():
	return num_internal() + num_terminals()

def is_valid(state):
	return state >= 0 and state < num_states()

def is_internal(state):
	assert is_valid(state)
	return state < num_internal()

def is_terminal(state):
	assert is_valid(state)
	return not is_internal(state)

def root():
	return 0

def actor(state):
	assert is_internal(state)
	return state % 3

def to_decision(state):
	assert is_internal(state)
	return state / 3

def can_bet(state):
	return to_decision(state) == 0

def can_call(state):
	assert is_internal(state)
	return True

def can_fold(state):
	return not can_bet(state)

def facing_bet(state):
    return can_fold(state)

def facing_bet_call(state):
    return to_decision(state) == 3

def facing_bet_fold(state):
    return to_decision(state) == 2

def call_closes_action(state):
    return facing_bet_call(state) or facing_bet_fold(state) 

def num_actions(state):
	assert is_internal(state)
	return 2

def act(state, action):
	assert action < num_actions(state)
	player, decision = actor(state), to_decision(state)

	if decision == 0:
		if action == 0:
			if player == 2:
				return num_internal()
			else:
				return player + 1
		else:
			return (player+1)%3 + 3
	elif decision == 1:
		if action == 0:
			return (player+1)%3 + 3*3
		else:
			return (player+1)%3 + 3*2
	elif decision == 2:
		if action == 0:
			return num_internal() + 1 + (player+1)%3 + 3*2
		else:
			return num_internal() + 1 + (player+1)%3 
	else:
		if action == 0:
			return num_internal() + 1 + (player+1)%3 + 3*3
		else:
			return num_internal() + 1 + (player+1)%3 + 3

def action_name(state, action):
    assert action < num_actions(state)

    if action == 0:
        return 'c'

    if can_bet(state):
        return 'r'
    else:
        return 'f'

def is_showdown(state):
	assert is_terminal(state)

	if state == num_internal():
		return True

	return (state - (num_internal() + 1)) / 3 > 0

def is_fold(state):
	return not is_showdown(state)

def folded(state, player):
	assert is_terminal(state)

	if state == num_internal():
		return False
	else:
		t       = state - (num_internal() + 1)
		bettor  = t%3
		first   = t/3%2
		second  = t/3/2

		return not (player == bettor or (player == (bettor+1)%3 and first==1) or (player == (bettor+2)%3 and second==1))

def at_showdown(state, player):
	assert is_terminal(state)

	if is_showdown(state):
		return not folded(state, player)
	else:
		return False

def bettor(state):
	assert is_terminal(state)
	assert state >= num_internal() + 1

	return (state - (num_internal() + 1))%3

def pot_size(state):
	assert is_terminal(state)

	if state == num_internal():
		return 3
	else:
		t       = state - (num_internal() + 1)
		first   = t/3%2
		second  = t/3/2
                return 4 + first + second

def pot_contribution(state, player):
	assert is_terminal(state)

	if state == num_internal():
		return 1
	else:
		return 1 + (not folded(state, player))

def to_string(state):
	assert is_terminal(state)

	t       = state - (num_internal() + 1)
	bettor  = t%3
	first   = t/3%2
	second  = t/3/2

	if state == num_internal():
		return 'ccc'
	else:
		return 'c'*bettor + 'r' + (first and 'c' or 'f') + (second and 'c' or 'f')

def string_to_state(string):
    state = root()

    for action in string:
        assert is_internal(state)
        if action == 'c':
            state = act(state, 0)
        elif action == 'r':
            assert can_bet(state)
            state = act(state, 1)
        elif action == 'f':
            assert can_fold(state)
            state = act(state, 1)
        else:
            assert False

    return state
