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


def play_hand(players, cards):
    state = betting.root()

    for i in range(3):
        players[i].start_hand(i, cards[i])

    while not betting.is_terminal(state):
        player = betting.actor(state)
        action = players[player].act(state, cards[player])
        state = betting.act(state, action)

    shown_cards = [cards[i] if betting.at_showdown(state, i) else None for i in range(3)]

    for i in range(3):
        players[i].end_hand(i, cards[i], state, shown_cards)

    the_winner = winner(state, cards)
    pot_size = betting.pot_size(state)

    return (state, [pot_size * (i == the_winner) - betting.pot_contribution(state, i) for i in range(3)])
