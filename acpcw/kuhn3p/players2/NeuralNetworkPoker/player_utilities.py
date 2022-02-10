# KKK
def utility_func1(i, cards):
    if list(cards).index(max(cards)) == i:
        return 2
    return -1


# KKBFF
def utility_func2(i, cards):
    if list(cards).index(max(cards)) == i:
        return 2
    return -1


# KKBFC
def utility_func3(i, cards):
    if list(cards).index(max(cards)) == i:
        return 3
    return -1 if i == 0 else -2


# KKBCF
def utility_func4(i, cards):
    if list(cards).index(max(cards)) == i:
        return 3
    return -1 if i == 1 else -2


# KKBCC
def utility_func5(i, cards):
    if list(cards).index(max(cards)) == i:
        return 4
    return -2


# KBFF
def utility_func6(i, cards):
    if i == 0:
        return -1
    elif i == 2:
        return -1
    return 2


# KBFC
def utility_func7(i, cards):
    if i == 2:
        return -1
    elif i == 0:
        if cards[i] > cards[1]:
            return 3
        else:
            return -2
    elif i == 1:
        if cards[i] > cards[0]:
            return 3
        else:
            return -2


# KBCF
def utility_func8(i, cards):
    if i == 0:
        return -1
    elif i == 1:
        if cards[i] > cards[1]:
            return 3
        else:
            return -2
    elif i == 2:
        if cards[i] > cards[0]:
            return 3
        else:
            return -2


# KBCC
def utility_func9(i, cards):
    if list(cards).index(max(cards)) == i:
        return 4
    return -2


# BFF
def utility_func10(i, cards):
    if i == 1:
        return -1
    elif i == 2:
        return -1
    return 2


# BFC
def utility_func11(i, cards):
    if i == 1:
        return -1
    temp = cards
    list(temp).remove(cards[1])
    # print temp
    m = temp.index(max(temp))
    return 3 if m == i else -2


# BCF
def utility_func12(i, cards):
    if i == 2:
        return -1
    elif i == 0:
        if cards[i] > cards[1]:
            return 3
        else:
            return -2
    elif i == 1:
        if cards[i] > cards[0]:
            return 3
        else:
            return -2


# BCC
def utility_func13(i, cards):
    if cards.index(max(cards)) == i:
        return 4
    else:
        return -2


UTILITY_DICT = {
    "ccc":   utility_func1,
    "crcc":  utility_func9,
    "ccrcf": utility_func5,
    "ccrcc": utility_func4,        
    "ccrfc": utility_func3,
    "ccrff": utility_func2,
    "crff":  utility_func6,
    "crfc":  utility_func7,
    "crcf":  utility_func8,
    "rfc":   utility_func11,
    "rcc":   utility_func13,
    "rcf":   utility_func12,
    "rff":   utility_func10,
}


STATE_DICT = {
    'i':       1, # i
    'c':       1, # k
    'cc':      1,  # kk
    'ccr':     2, # kkb
    'r':       2, # b    
    'cr':      2, # kb  
    'crf':     3, # kbf      
    'ccrf':    3, # kkbf
    'rf':      3, # bf  
    'crc':     4, # kbc      
    'ccrc':    4, # kkbc
    'rc':      4, #bc
    # terminal states
    'ccc':     5,  # kkk    
    'crcc':    5,  # kbcc
    'ccrcf':   5,  # kkbcf
    'ccrcc':   5,  # kkbcc
    'ccrfc':   5,  # kkbfc
    'ccrff':   5,  # kkbff
    'crfc':    5,  # kbfc
    'crff':    5,  # kbff
    'crcf':    5,  # kbcf
    'rfc':     5,  # bfc
    'rcc':     5,  # bcc
    'rcf':     5,  # bcf
    'rff':     5   # bff
}

PAYOUT_DICT = {
    # 'i':       1, # i
    # 'c':       1, # k
    # 'cc':      1,  # kk
    'ccr':     -1, # kkb
    # 'r':       2, # b    
    # 'cr':      2, # kb  
    'crf':     -1, # kbf      
    'ccrf':    -1, # kkbf
    # 'rf':      3, # bf  
    'crc':     -1, # kbc      
    'ccrc':    -1, # kkbc
    # 'rc':      4, #bc
}