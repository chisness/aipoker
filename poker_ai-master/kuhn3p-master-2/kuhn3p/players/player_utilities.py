# KKK
def utility_func1(i, hand):
    if list(hand).index(max(hand)) == i:
        return 2
    return -1


# KKBFF
def utility_func2(i, hand):
    if list(hand).index(max(hand)) == i:
        return 2
    return -1


# KKBFC
def utility_func3(i, hand):
    if list(hand).index(max(hand)) == i:
        return 3
    return -1 if i == 0 else -2


# KKBCF
def utility_func4(i, hand):
    if list(hand).index(max(hand)) == i:
        return 3
    return -1 if i == 1 else -2


# KKBCC
def utility_func5(i, hand):
    if list(hand).index(max(hand)) == i:
        return 4
    return -2


# KBFF
def utility_func6(i, hand):
    if i == 0:
        return -1
    elif i == 2:
        return -1
    return 2


# KBFC
def utility_func7(i, hand):
    if i == 2:
        return -1
    elif i == 0:
        if hand[i] > hand[1]:
            return 3
        else:
            return -2
    elif i == 1:
        if hand[i] > hand[0]:
            return 3
        else:
            return -2


# KBCF
def utility_func8(i, hand):
    if i == 0:
        return -1
    elif i == 1:
        if hand[i] > hand[1]:
            return 3
        else:
            return -2
    elif i == 2:
        if hand[i] > hand[0]:
            return 3
        else:
            return -2


# KBCC
def utility_func9(i, hand):
    if list(hand).index(max(hand)) == i:
        return 4
    return -2


# BFF
def utility_func10(i, hand):
    if i == 1:
        return -1
    elif i == 2:
        return -1
    return 2


# BFC
def utility_func11(i, hand):
    if i == 1:
        return -1
    temp = hand
    list(temp).remove(hand[1])
    # print temp
    m = temp.index(max(temp))
    return 3 if m == i else -2


# BCF
def utility_func12(i, hand):
    if i == 2:
        return -1
    elif i == 0:
        if hand[i] > hand[1]:
            return 3
        else:
            return -2
    elif i == 1:
        if hand[i] > hand[0]:
            return 3
        else:
            return -2


# BCC
def utility_func13(i, hand):
    if hand.index(max(hand)) == i:
        return 4
    else:
        return -2


UTILITY_DICT = {
    "kkk": utility_func1,
    "kkbff": utility_func2,
    "kkbfc": utility_func3,
    "kkbcf": utility_func4,
    "kkbcc": utility_func5,
    "kbff": utility_func6,
    "kbfc": utility_func7,
    "kbcf": utility_func8,
    "kbcc": utility_func9,
    "bff": utility_func10,
    "bfc": utility_func11,
    "bcf": utility_func12,
    "bcc": utility_func13
}
