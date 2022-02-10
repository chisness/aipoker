from kuhn3p import betting, deck


def winner(state, cards):
    assert betting.is_terminal(state)
    if betting.is_showdown(state):
        best_player = -1
        best_card = -1
        for i in range(3):
            if betting.at_showdown(state, i) and cards[i] > best_card:
                best_player = i
                best_card = cards[i]
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

def play_hand(players, cards, our_position):
    state = betting.root()

    for i in range(3):
        players[i].start_hand(i, cards[i])

    our_card.append(cards[our_position])
    if our_position == 0:
        p2_position = 1
        p3_position = 2
    elif our_position == 1:
        p2_position = 2
        p3_position = 0
    elif our_position == 2:
        p2_position = 1
        p3_position = 2

    while not betting.is_terminal(state):
        player = betting.actor(state)
        print('player', player)
        action = players[player].act(state, cards[player])
        print('action', action)
        if player == p2_position:
            player_2_actions.append(action)
            player_2_states.append(state)
        elif player == p3_position:
            player_3_actions.append(action)
            player_3_states.append(state)
        state = betting.act(state, action)

    shown_cards = [cards[i] if betting.at_showdown(state, i) else None for i in range(3)]
    player_2_sd = shown_cards[p2_position]
    player_3_sd = shown_cards[p3_position]

    for i in range(3):
        players[i].end_hand(i, cards[i], state, shown_cards)

    the_winner = winner(state, cards)
    pot_size = betting.pot_size(state)

    return (state, [pot_size * (i == the_winner) - betting.pot_contribution(state, i) for i in range(3)], our_card, player_2_actions, player_2_states, player_2_sd, player_3_actions, player_3_states, player_3_sd)
