import io, re, socket, sys
import kuhn3p.betting, kuhn3p.deck

version = sys.argv[3]
player = kuhn3p.players.UltimateAiKhun2(version) 
address = sys.argv[1]
port = int(sys.argv[2])

sock = socket.create_connection((address, port))
sockin = sock.makefile(mode='rb')

sock.send(('VERSION:2.0.0\r\n').encode())

state_regex = re.compile(r"MATCHSTATE:(\d):(\d+):([^:]*):([^|]*)\|([^|]*)\|(.*)")

position = None
hand = None
while 1:
    line = sockin.readline().strip()

    if not line:
        break
    
    
    state = state_regex.match(line.decode())

    node = state.group(3)
    
    def maybe_suited_card_string_to_card(x):
        if not (x is None) and len(x) == 2:
            return kuhn3p.deck.string_to_card(x[0])
        else:
            return None


    this_position, this_hand = [int(x) for x in state.group(1, 2)]
    betting = kuhn3p.betting.string_to_state(state.group(3))
    cards = list(map(maybe_suited_card_string_to_card, state.group(4, 5, 6)))

    if not (this_hand == hand):
        assert hand is None
        position = this_position
        hand = this_hand

        assert not (cards[position] is None)
        player.start_hand(position, cards[position])
    

    assert not (hand is None)

    # print(betting)
    
    # print(kuhn3p.betting.is_internal(betting))

    if kuhn3p.betting.is_internal(betting) and kuhn3p.betting.actor(betting) == position:
        assert not (cards[position] is None)
        action = player.act(betting, cards[position], node)

        response = '%s:%s\r\n' % (line, kuhn3p.betting.action_name(betting, action))

        sock.send(response.encode())

    if kuhn3p.betting.is_terminal(betting):
        assert not (cards[position] is None)
        player.end_hand(position, cards[position], betting, cards)

        hand = None
