from kuhn3p import deck, dealer, players
from random import Random

rng = Random()
rng.seed(31372)  # each seed corresponds to a different set of hands

num_hands = 10
the_players = [players.UltimateAiKhun(),
               players.Chump(0.01, 0.01, 0.0),
               players.Bluffer(0.2)]

total = [0, 0, 0]

#assume we are player 0
#get each other player action sequence with optional parameter for showdown

history_counter = {}
our_position = 0

for hand in range(num_hands):
    first = hand % 3
    second = (first + 1) % 3
    third = (second + 1) % 3

    if first == 0:
        our_position = 0
    elif second == 0:
        our_position = 1
    elif third == 0:
        our_position = 2
        
    print('our position', our_position)
                
    this_players = [the_players[first], the_players[second], the_players[third]]
    print('this players', this_players)

    #we are always player 0
    c = deck.shuffled(rng)
    print('c', c)

    (state, delta, our_card, p2a, p2s, p2sd, p3a, p3s, p3sd) = dealer.play_hand(this_players, c, our_position)
    for i in range(3):
        total[(first + i) % 3] += delta[i]

for i in range(3):
    print(f'Player {i}: {the_players[i]} profit {total[i]}')

print(f'Our card: {our_card}')
print(f'Player 2 states: {p2s}')
print(f'Player 2 actions: {p2a}')
print(f'Player 2 showdowns: {p2sd}')
print(f'Player 3 states: {p3s}')
print(f'Player 3 actions: {p3a}')
print(f'Player 3 showdowns: {p3sd}')