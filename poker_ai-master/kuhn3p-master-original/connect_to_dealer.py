import io, re, socket, sys
import kuhn3p.betting, kuhn3p.deck, kuhn3p.Player

import kuhn3p.players.Chump

player  = kuhn3p.players.Chump(1, 1, 1)
address = sys.argv[1]
port    = int(sys.argv[2])

sock    = socket.create_connection((address, port))
sockin  = sock.makefile(mode='rb')

sock.send('VERSION:2.0.0\r\n')

state_regex = re.compile(r"MATCHSTATE:(\d):(\d+):([^:]*):([^|]*)\|([^|]*)\|(.*)")

position = None
hand     = None
while 1:
    line = sockin.readline().strip()

    if not line:
        break

    state = state_regex.match(line)

    def maybe_suited_card_string_to_card(x):
        if not (x == None) and len(x) == 2:
            return kuhn3p.deck.string_to_card(x[0])
        else:
            return None
   
    this_position, this_hand = map(lambda x: int(x), state.group(1, 2))
    betting                  = kuhn3p.betting.string_to_state(state.group(3))
    cards                    = map(maybe_suited_card_string_to_card, state.group(4, 5, 6))

    if not (this_hand == hand):
        assert hand == None
        position = this_position
        hand     = this_hand

        assert not (cards[position] == None)
        player.start_hand(position, cards[position])

    assert not (hand == None)
    if kuhn3p.betting.is_internal(betting) and kuhn3p.betting.actor(betting) == position:
        assert not (cards[position] == None)
        action = player.act(betting, cards[position])

        response = '%s:%s\r\n' % (line, kuhn3p.betting.action_name(betting, action))
        sock.send(response)

    if kuhn3p.betting.is_terminal(betting):
        assert not (cards[position] == None)
        player.end_hand(position, cards[position], betting, cards)

        hand = None
