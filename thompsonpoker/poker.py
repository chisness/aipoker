# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:59:13 2020

@author: mthom
"""


import numpy as np
import pandas as pd


def rank(hand): #Card Rank 2, 3, .., King, Ace (labeled 0 thru 12)
    return [i%13 for i in hand]

def suit(hand): #Suits 0 thru 3
    return [i//13 for i in hand]

#Hand Rankings
#8 straight flush (incl. royal) tie_cards 1
#7 four of a kind tie_cards 2
#6 full house tie_cards 2 
#5 flush tie_cards 5
#4 straight val4, tie_cards 1
#3 three of a kind val3, tie_cards 3
#2 two pair val2, tie_cards 3
#1 one pair val1, tie_cards 4
#0 High card val 0, tie_cards 5

#preflop hand rankings according to
#preflophands.com (consider verifying via simulaiton. Note hands are not transitive)
def hole_ranks(ranks, suits):
    hole = pd.DataFrame(ranks)
    hole = hole[[0,1,2,3]]
    hole_suit = pd.DataFrame(suits)
    hole_suit = hole_suit[[0,1,2,3]]
    
    hole = pd.concat([hole,hole_suit], axis=1)
    
    hole['c1'] = hole.apply(lambda df: max(df[0], df[1]), axis=1, raw=True)
    hole['c2'] = hole.apply(lambda df: min(df[0], df[1]), axis=1, raw=True)
    hole['c3'] = hole.apply(lambda df: max(df[2], df[3]), axis=1, raw=True)
    hole['c4'] = hole.apply(lambda df: min(df[2], df[3]), axis=1, raw=True)
    
    hole['suited1'] = hole.apply(lambda df: int(df[4]==df[5]), axis=1, raw=True)
    hole['suited2'] = hole.apply(lambda df: int(df[6]==df[7]), axis=1, raw=True)
    
    
    hole_rank = pd.read_csv('hole_rank.csv')
    
    hole_rank.columns = ['c1','c2','suited1','rank1','suitless1']
    lenCheck = len(hole)
    hole = hole.merge(hole_rank, how='left', on=['c1','c2','suited1'])
    if lenCheck != len(hole):
        raise Exception('merge error')
        
    hole_rank.columns = ['c3','c4','suited2','rank2','suitless2']
    hole = hole.merge(hole_rank, how='left', on=['c3','c4','suited2'])
    if lenCheck != len(hole):
        raise Exception('merge error')
        
        
    hole = hole[['rank1','rank2','suitless1','suitless2']]
    
    return hole



def check_straight(rank):
    straight, high_card = False, -1
    rank.sort()
    #Check A thru 5 Straight
    if rank == [0,1,2,3,12]:
        straight, high_card = True, 3
    
    elif [rank[i+1]-rank[i] for i in range(4)] == [1]*4:
        straight, high_card = True, rank[-1]
    
    return straight, high_card

def hand_strength(rank, suit, include_tie_break=True):
    #0 is a 2, 1 is a 3, ..., 9 is Jack, 10 is Queen, 11 is King, 12 is Ace
    #suits are 0 through 3
    
    
    p1_rank = [rank[i] for i in range(len(rank)) if i not in [2,3]]
    p2_rank = [rank[i] for i in range(len(rank)) if i not in [0,1]]
    
    p1_suit = [suit[i] for i in range(len(suit)) if i not in [2,3]]
    p2_suit = [suit[i] for i in range(len(suit)) if i not in [0,1]]
    
    
    #look for straights
    p1_straight, p1_high_card_straight = check_straight(p1_rank.copy())
    p2_straight, p2_high_card_straight = check_straight(p2_rank.copy())
        
    p1_flush = np.bincount(p1_suit).max()>=5
    p1_max_freq = np.bincount(p1_rank).max()
    
    p2_flush = np.bincount(p2_suit).max()>=5
    p2_max_freq = np.bincount(p2_rank).max()
    
    p1_four = p1_max_freq == 4
    p1_three = p1_max_freq == 3
    p1_pair = p1_max_freq == 2
    
    p2_four = p2_max_freq == 4
    p2_three = p2_max_freq == 3
    p2_pair = p2_max_freq == 2
    
    
    p1_rank_unique = list(set(p1_rank))
    p1_rank_unique.sort()
    p1_two_pair = [p1_rank.count(x) for x in p1_rank_unique].count(2)>=2
    p2_rank_unique = list(set(p2_rank))
    p2_rank_unique.sort()
    p2_two_pair = [p2_rank.count(x) for x in p2_rank_unique].count(2)>=2
    
    p1_full_house = ([p1_rank.count(x) for x in p1_rank_unique].count(3) and [p1_rank.count(x) for x in p1_rank_unique].count(2)) or [p1_rank.count(x) for x in p1_rank_unique].count(3) == 2
    p2_full_house = ([p2_rank.count(x) for x in p2_rank_unique].count(3) and [p2_rank.count(x) for x in p2_rank_unique].count(2)) or [p2_rank.count(x) for x in p2_rank_unique].count(3) == 2
    
    
    #check for straight flush
    if p1_flush and p1_straight:
        p1_flush_suit = np.bincount(p1_suit).argmax()
        p1_flush_cards = [p1_rank[i] for i, x in enumerate(p1_suit) if x==p1_flush_suit]
        p1_straight_flush, p1_straight_flush_high_card = check_straight(p1_flush_cards.copy())
    else:
        p1_straight_flush = False
    
    #check for straight flush
    if p2_flush and p2_straight:
        p2_flush_suit = np.bincount(p2_suit).argmax()
        p2_flush_cards = [p2_rank[i] for i, x in enumerate(p2_suit) if x==p2_flush_suit]
        p2_straight_flush, p2_straight_flush_high_card = check_straight(p2_flush_cards.copy())
    else:
        p2_straight_flush = False
    
    if p1_straight_flush:
        p1_strength = 8
    elif p1_four:
        p1_strength = 7
    elif p1_full_house:
        p1_strength = 6
    elif p1_flush:
        p1_strength = 5
    elif p1_straight:
        p1_strength = 4
    elif p1_three:
        p1_strength = 3
    elif p1_two_pair:
        p1_strength = 2
    elif p1_pair:
        p1_strength = 1
    else:
        p1_strength = 0
    
    if p2_straight_flush:
        p2_strength = 8
    elif p2_four:
        p2_strength = 7
    elif p2_full_house:
        p2_strength = 6
    elif p2_flush:
        p2_strength = 5
    elif p2_straight:
        p2_strength = 4
    elif p2_three:
        p2_strength = 3
    elif p2_two_pair:
        p2_strength = 2
    elif p2_pair:
        p2_strength = 1
    else:
        p2_strength = 0
    
    
    
    #determine tie breaks 
    if include_tie_break:
        if p1_strength == 8:
            p1_tie = [p1_straight_flush_high_card]
        elif p1_strength == 7:
            p1_quads = np.bincount(p1_rank).argmax()
            p1_kicker = p1_rank_unique[-1]
            if p1_kicker == p1_quads:
                p1_kicker = p1_rank_unique[-2]
            p1_tie = [np.bincount(p1_rank).argmax(), p1_kicker]
        elif p1_strength == 6:
            threes = [x for x in p1_rank_unique if p1_rank.count(x)==3]
            twos = [x for x in p1_rank_unique if p1_rank.count(x)==2]
            if len(threes) == 2:
                p1_tie = [max(threes), min(threes)]
            else:
                p1_tie = [max(threes), max(twos)]
        elif p1_strength ==5:
            p1_tie = [p1_rank[i] for i, suit in enumerate(p1_suit) if suit == np.bincount(p1_suit).argmax()]
            p1_tie.sort()
            p1_tie = p1_tie[-5::][::-1]
        elif p1_strength == 4:
            p1_tie = [p1_high_card_straight]
        elif p1_strength == 3:
            trips = np.bincount(p1_rank).argmax()
            p1_kicker = [x for x in p1_rank_unique if x!=trips][-1]
            p1_kicker2 = [x for x in p1_rank_unique if x!=trips][-2]
            p1_tie = [trips, p1_kicker, p1_kicker2]
        elif p1_strength == 2:
            pairs = [x for x in p1_rank_unique if p1_rank.count(x)==2]
            pairs = pairs[::-1]
            pairs = pairs[:2]
            p1_kicker = [x for x in p1_rank_unique if x not in pairs][-1]
            p1_tie = [pairs[0], pairs[1], p1_kicker]
        elif p1_strength == 1:
            pair = [x for x in p1_rank_unique if p1_rank.count(x)==2]
            pair = pair[0]
            p1_kicker = [x for x in p1_rank if x != pair]
            p1_kicker.sort()
            p1_kicker = p1_kicker[::-1]
            p1_tie = [pair, p1_kicker[0], p1_kicker[1], p1_kicker[2]]
        elif p1_strength == 0:
            p1_tie = p1_rank.copy()
            p1_tie.sort()
            p1_tie = p1_tie[::-1]
            p1_tie = p1_tie[0:5]
        
        if p2_strength == 8:
            p2_tie = [p2_straight_flush_high_card]
        elif p2_strength == 7:
            p2_quads = np.bincount(p2_rank).argmax()
            p2_kicker = p2_rank_unique[-1]
            if p2_kicker == p2_quads:
                p2_kicker = p2_rank_unique[-2]
            p2_tie = [np.bincount(p2_rank).argmax(), p2_kicker]
        elif p2_strength == 6:
            threes = [x for x in p2_rank_unique if p2_rank.count(x)==3]
            twos = [x for x in p2_rank_unique if p2_rank.count(x)==2]
            if len(threes) == 2:
                p2_tie = [max(threes), min(threes)]
            else:
                p2_tie = [max(threes), max(twos)]
        elif p2_strength ==5:
            p2_tie = [p2_rank[i] for i, suit in enumerate(p2_suit) if suit == np.bincount(p2_suit).argmax()]
            p2_tie.sort()
            p2_tie = p2_tie[-5::][::-1]
        elif p2_strength == 4:
            p2_tie = [p2_high_card_straight]
        elif p2_strength == 3:
            trips = np.bincount(p2_rank).argmax()
            p2_kicker = [x for x in p2_rank_unique if x!=trips][-1]
            p2_kicker2 = [x for x in p2_rank_unique if x!=trips][-2]
            p2_tie = [trips, p2_kicker, p2_kicker2]
        elif p2_strength == 2:
            pairs = [x for x in p2_rank_unique if p2_rank.count(x)==2]
            pairs = pairs[::-1]
            pairs = pairs[:2]
            p2_kicker = [x for x in p2_rank_unique if x not in pairs][-1]
            p2_tie = [pairs[0], pairs[1], p2_kicker]
        elif p2_strength == 1:
            pair = [x for x in p2_rank_unique if p2_rank.count(x)==2]
            pair = pair[0]
            p2_kicker = [x for x in p2_rank if x != pair]
            p2_kicker.sort()
            p2_kicker = p2_kicker[::-1]
            p2_tie = [pair, p2_kicker[0], p2_kicker[1], p2_kicker[2]]
        elif p2_strength == 0:
            p2_tie = p2_rank.copy()
            p2_tie.sort()
            p2_tie = p2_tie[::-1]
            p2_tie = p2_tie[0:5]
        
        
        if p1_strength > p2_strength:
            p1_win = 1
        elif p1_strength < p2_strength:
            p1_win = -1
        else:
            p1_win = 0
            for i, j in zip(p1_tie, p2_tie):
                if p1_win == 0:
                    if i>j:
                        p1_win =1
                    elif i<j:
                        p1_win = -1
        
        board_cards = p1_rank[2:].copy()
        board_cards.sort()  
        p1_hand_subvalue = -1 #make sure all values are set
        if p1_strength >=7: #>=7 is hands all grouped together (four of a kind, straight flush (incl. royal)) (i.e. just assumes you'll win)
            p1_hand_subvalue = 0
        
        if p1_strength ==6: #full house
            if p1_rank[0] == p1_rank[1]:
                if p1_rank[0] != np.bincount(p1_rank).argmax(): #implies three of a kind is from board
                    if p1_rank[0] == max(p1_rank):
                        p1_hand_subvalue = 1 #pocket pair > board 3ofakind
                    else:
                        p1_hand_subvalue = 0 #pocket pair < board 3ofakind
                else: #pocket pair part of three of a kind
                    if p1_rank[0] == max(p1_rank):
                        p1_hand_subvalue = 3 #pocket pair + board > board pair
                    else:
                        p1_hand_subvalue = 2 #pocket pair + board < board pair
            else:#one hole matches board pair, other hole matches other board card
                if np.bincount(p1_rank[2:]).argmax() == max(p1_rank[2:]):
                    p1_hand_subvalue = 5 #board pair is greater than other board card
                else:
                    p1_hand_subvalue = 4 #board pair is less than other board card - this is worse since opp might have pocket pair of other board card
                
        if p1_strength == 5 or p1_strength == 4: #4 is straight; 5 is flush
            p1_hand_subvalue = board_cards[2] #highest board card gives more info now that it includes hole cards in state
        
        if p1_strength == 3:
            if p1_rank[0]==p1_rank[1]: #pocket pair matches one board card
                if p1_rank[0] == board_cards[2]:
                    p1_hand_subvalue = 9
                elif p1_rank[0] == board_cards[1]:
                    p1_hand_subvalue = 8
                elif p1_rank[0] == board_cards[0]:
                    p1_hand_subvalue = 7
            elif max(p1_rank[0:2]) == np.bincount(p1_rank).argmax(): #paired higher hole card with board pair
                if max(p1_rank[0:2]) == board_cards[2]: #high hole card matches high board pair
                    p1_hand_subvalue = 6
                else:
                    p1_hand_subvalue = 5 #high hole card matches low board pair
            elif max(p1_rank[0:2]) == np.bincount(p1_rank).argmax(): #paired lower hole card with board pair
                if min(p1_rank[0:2]) == board_cards[2]: #low hole card matches high board pair
                    p1_hand_subvalue = 4
                else:
                    p1_hand_subvalue = 3 #low hole card matches low board pair
            else: #three of a kind on is from the board
                if min(p1_rank[0:2]) > board_cards[0]:
                    p1_hand_subvalue = 2 #both hole cards higher than board 3 of a kind
                elif max(p1_rank[0:2]) > board_cards[0]:
                    p1_hand_subvalue = 1 #high hole card higher than board 3 of a kind
                else:
                    p1_hand_subvalue = 0 #both hole cards lower than board 3 of a kind
                    
            
        if p1_strength == 2: #two pair
            if np.bincount(p1_rank[2:]).max() < 2: # no board pair ==> holes cards match 2 board cards
                if min(p1_rank[0:2]) == board_cards[1]: #hole cards match top two board cards:
                    p1_hand_subvalue = 10
                elif max(p1_rank[0:2]) == board_cards[2]: #hole cards match low/high board cards:
                    p1_hand_subvalue = 9
                else: #hole cards match bottom two board cards
                    p1_hand_subvalue = 8
            elif  p1_rank[0]==p1_rank[1]: #pocket pair
                if p1_rank[0] > board_cards[2]: #pocket pair greater than board
                    p1_hand_subvalue = 7
                elif p1_rank[0] > board_cards[1]: #pocket pair greater than board pair
                    p1_hand_subvalue = 6
                elif p1_rank[0] > board_cards[0]: #pocket pair greater than low board card but lower than high board pair
                    p1_hand_subvalue = 5
                else: #pocket pair less than all three board cards
                    p1_hand_subvalue = 4
            else: #board pair and one hole card pairs other board card
                if max(p1_rank[0:2])==[idx for idx, val in enumerate(np.bincount(p1_rank[2:])==1) if val][0]: #high hole card matches unmatched board card
                    if max(p1_rank[0:2]) == board_cards[2]:
                        p1_hand_subvalue = 3 #matched board card higher than board pair
                    else:
                        p1_hand_subvalue = 2 #matched board card lower than board pair
                else: #lower hole card matches unmatched board card
                    if min(p1_rank[0:2]) == board_cards[2]:
                        p1_hand_subvalue = 1 #matched board card higher than board pair
                    else:
                        p1_hand_subvalue = 0 #matched board card lower than board pair
                        
                        
          
        if p1_strength == 1: #pair
            if p1_rank[0]==p1_rank[1]:#pocket pair
                if p1_rank[0] > board_cards[2]:
                    p1_hand_subvalue = 18
                elif p1_rank[0] > board_cards[1]:
                    p1_hand_subvalue = 17
                elif p1_rank[0] > board_cards[0]:
                    p1_hand_subvalue = 16
                else:
                    p1_hand_subvalue = 15
            else:
                if any([max(p1_rank[:2]) == x for x in p1_rank[2:]]): #max hole card is a match
                    if max(p1_rank[:2]) == board_cards[2]:
                        p1_hand_subvalue = 14
                    elif max(p1_rank[:2]) == board_cards[1]:
                        p1_hand_subvalue = 13
                    elif max(p1_rank[:2]) == board_cards[1]:
                        p1_hand_subvalue = 12
                    else:
                        p1_hand_subvalue = 11
                elif any([min(p1_rank[:2]) == x for x in p1_rank[2:]]): #min hole card is a match
                    if min(p1_rank[:2]) == board_cards[2]:
                        p1_hand_subvalue = 10
                    elif min(p1_rank[:2]) == board_cards[1]:
                        p1_hand_subvalue = 9
                    elif min(p1_rank[:2]) == board_cards[1]:
                        p1_hand_subvalue = 8
                    else:
                        p1_hand_subvalue = 7
                else: #board pair
                    if min(p1_rank[0:2]) > board_cards[2]:
                        p1_hand_subvalue = 6
                    elif max(p1_rank[0:2]) > board_cards[2]:
                        p1_hand_subvalue = 5
                    elif min(p1_rank[0:2]) > board_cards[1]:
                        p1_hand_subvalue = 4
                    elif max(p1_rank[0:2]) > board_cards[1]:
                        p1_hand_subvalue = 3
                    elif min(p1_rank[0:2]) > board_cards[0]:
                        p1_hand_subvalue = 2
                    elif max(p1_rank[0:2]) > board_cards[0]:
                        p1_hand_subvalue = 1
                    else:
                        p1_hand_subvalue = 0
                        
        if p1_strength == 0:
            p1_hand_subvalue = board_cards[2] #set to max board value
            
        
        p2_hand_subvalue = -1 #make sure all values are set
        if p2_strength >=7: #>=7 is hands all grouped together (four of a kind, straight flush (incl. royal)) (i.e. just assumes you'll win)
            p2_hand_subvalue = 0
        
        if p2_strength ==6: #full house
            if p2_rank[0] == p2_rank[1]:
                if p2_rank[0] != np.bincount(p2_rank).argmax(): #implies three of a kind is from board
                    if p2_rank[0] == max(p2_rank):
                        p2_hand_subvalue = 1 #pocket pair > board 3ofakind
                    else:
                        p2_hand_subvalue = 0 #pocket pair < board 3ofakind
                else: #pocket pair part of three of a kind
                    if p2_rank[0] == max(p2_rank):
                        p2_hand_subvalue = 3 #pocket pair + board > board pair
                    else:
                        p2_hand_subvalue = 2 #pocket pair + board < board pair
            else:#one hole matches board pair, other hole matches other board card
                if np.bincount(p2_rank[2:]).argmax() == max(p2_rank[2:]):
                    p2_hand_subvalue = 5 #board pair is greater than other board card
                else:
                    p2_hand_subvalue = 4 #board pair is less than other board card - this is worse since opp might have pocket pair of other board card
                
        if p2_strength == 5 or p2_strength == 4: #4 is straight; 5 is flush
            p2_hand_subvalue = board_cards[2] #highest board card gives more info now that it includes hole cards in state
        
        if p2_strength == 3:
            if p2_rank[0]==p2_rank[1]: #pocket pair matches one board card
                if p2_rank[0] == board_cards[2]:
                    p2_hand_subvalue = 9
                elif p2_rank[0] == board_cards[1]:
                    p2_hand_subvalue = 8
                elif p2_rank[0] == board_cards[0]:
                    p2_hand_subvalue = 7
            elif max(p2_rank[0:2]) == np.bincount(p2_rank).argmax(): #paired higher hole card with board pair
                if max(p2_rank[0:2]) == board_cards[2]: #high hole card matches high board pair
                    p2_hand_subvalue = 6
                else:
                    p2_hand_subvalue = 5 #high hole card matches low board pair
            elif max(p2_rank[0:2]) == np.bincount(p2_rank).argmax(): #paired lower hole card with board pair
                if min(p2_rank[0:2]) == board_cards[2]: #low hole card matches high board pair
                    p2_hand_subvalue = 4
                else:
                    p2_hand_subvalue = 3 #low hole card matches low board pair
            else: #three of a kind on is from the board
                if min(p2_rank[0:2]) > board_cards[0]:
                    p2_hand_subvalue = 2 #both hole cards higher than board 3 of a kind
                elif max(p2_rank[0:2]) > board_cards[0]:
                    p2_hand_subvalue = 1 #high hole card higher than board 3 of a kind
                else:
                    p2_hand_subvalue = 0 #both hole cards lower than board 3 of a kind
                    
            
        if p2_strength == 2: #two pair
            if np.bincount(p2_rank[2:]).max() < 2: # no board pair ==> holes cards match 2 board cards
                if min(p2_rank[0:2]) == board_cards[1]: #hole cards match top two board cards:
                    p2_hand_subvalue = 10
                elif max(p2_rank[0:2]) == board_cards[2]: #hole cards match low/high board cards:
                    p2_hand_subvalue = 9
                else: #hole cards match bottom two board cards
                    p2_hand_subvalue = 8
            elif  p2_rank[0]==p2_rank[1]: #pocket pair
                if p2_rank[0] > board_cards[2]: #pocket pair greater than board
                    p2_hand_subvalue = 7
                elif p2_rank[0] > board_cards[1]: #pocket pair greater than board pair
                    p2_hand_subvalue = 6
                elif p2_rank[0] > board_cards[0]: #pocket pair greater than low board card but lower than high board pair
                    p2_hand_subvalue = 5
                else: #pocket pair less than all three board cards
                    p2_hand_subvalue = 4
            else: #board pair and one hole card pairs other board card
                if max(p2_rank[0:2])==[idx for idx, val in enumerate(np.bincount(p2_rank[2:])==1) if val][0]: #high hole card matches unmatched board card
                    if max(p2_rank[0:2]) == board_cards[2]:
                        p2_hand_subvalue = 3 #matched board card higher than board pair
                    else:
                        p2_hand_subvalue = 2 #matched board card lower than board pair
                else: #lower hole card matches unmatched board card
                    if min(p2_rank[0:2]) == board_cards[2]:
                        p2_hand_subvalue = 1 #matched board card higher than board pair
                    else:
                        p2_hand_subvalue = 0 #matched board card lower than board pair
                        
                        
          
        if p2_strength == 1: #pair
            if p2_rank[0]==p2_rank[1]:#pocket pair
                if p2_rank[0] > board_cards[2]:
                    p2_hand_subvalue = 18
                elif p2_rank[0] > board_cards[1]:
                    p2_hand_subvalue = 17
                elif p2_rank[0] > board_cards[0]:
                    p2_hand_subvalue = 16
                else:
                    p2_hand_subvalue = 15
            else:
                if any([max(p2_rank[:2]) == x for x in p2_rank[2:]]): #max hole card is a match
                    if max(p2_rank[:2]) == board_cards[2]:
                        p2_hand_subvalue = 14
                    elif max(p2_rank[:2]) == board_cards[1]:
                        p2_hand_subvalue = 13
                    elif max(p2_rank[:2]) == board_cards[1]:
                        p2_hand_subvalue = 12
                    else:
                        p2_hand_subvalue = 11
                elif any([min(p2_rank[:2]) == x for x in p2_rank[2:]]): #min hole card is a match
                    if min(p2_rank[:2]) == board_cards[2]:
                        p2_hand_subvalue = 10
                    elif min(p2_rank[:2]) == board_cards[1]:
                        p2_hand_subvalue = 9
                    elif min(p2_rank[:2]) == board_cards[1]:
                        p2_hand_subvalue = 8
                    else:
                        p2_hand_subvalue = 7
                else: #board pair
                    if min(p2_rank[0:2]) > board_cards[2]:
                        p2_hand_subvalue = 6
                    elif max(p2_rank[0:2]) > board_cards[2]:
                        p2_hand_subvalue = 5
                    elif min(p2_rank[0:2]) > board_cards[1]:
                        p2_hand_subvalue = 4
                    elif max(p2_rank[0:2]) > board_cards[1]:
                        p2_hand_subvalue = 3
                    elif min(p2_rank[0:2]) > board_cards[0]:
                        p2_hand_subvalue = 2
                    elif max(p2_rank[0:2]) > board_cards[0]:
                        p2_hand_subvalue = 1
                    else:
                        p2_hand_subvalue = 0
                        
        if p2_strength == 0:
            p2_hand_subvalue = board_cards[2] #set to max board value
     
        
        return p1_strength, p1_hand_subvalue, p2_strength, p2_hand_subvalue, p1_win
    return p1_strength, p2_strength


