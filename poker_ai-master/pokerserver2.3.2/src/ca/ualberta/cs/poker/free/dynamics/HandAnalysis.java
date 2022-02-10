/*
 * HandAnalysis.java
 *
 * This class is designed to compare two seven-card hands
 * in Texas Hold-Em. The initialization function HandAnalysis(Card[] cards)
 * does a lot of the work of determining the hand type (Straight flush, pair, et cetera)
 * as well as arranging the information in a way such that two hands of the
 * same type can be easily compared. For example, in two pair, the higher ranked pair
 * is recorded as the "most frequent" rank. In cases like 4 4 5 5 6 6 6, 6 is the most frequent
 * rank and 5 (not 4) is the second most frequent rank.
 * Possible "kickers" (cards that don't effect the hand type but might win the hand in case of
 * a tie) are also put in an array by index.
 *
 * Functions of the form "test___" are initialization functions that test various aspects
 * of the hand. In many cases, there is a particular order in which they must be called.
 * 
 * NOTE: This is not optimized code: it is intended primarily for server-side
 * use, where for every showdown there are several TCP/IP transactions that lead to it.
 * 
 * Created on April 18, 2006, 7:20 PM
 */

package ca.ualberta.cs.poker.free.dynamics;



/**
 *
 * @author Martin Zinkevich
 */
public class HandAnalysis {
    public enum HandType{
        STRAIGHTFLUSH(8),FOUROFAKIND(7),FULLHOUSE(6),FLUSH(5),STRAIGHT(4),THREEOFAKIND(3),TWOPAIR(2),PAIR(1),HIGHCARD(0);
        int strength;
        HandType(int strength){
            this.strength = strength;
        }
        public boolean strongerThan(HandType other){
            return (this.strength)>(other.strength);
        }
    }
    
    HandType handType;
    
    boolean containsFlush;
    int flushSuit;
    boolean[] flushRanks;
    
    boolean containsStraightFlush;
    boolean containsStraight;
    int straightHighCard;
    
    int mostFrequentRank;
    int secondMostFrequentRank;
    
    Card[] cards;
    int[] suitCounts;
    int[] rankCounts;
    int[] kickers;
    public HandAnalysis(Card[] cards){
        this.cards = cards;
        suitCounts=new int[Card.Suit.values().length];
        rankCounts = new int[Card.Rank.values().length];
        for(int i=0;i<cards.length;i++){
            suitCounts[cards[i].suit.index]++;
            rankCounts[cards[i].rank.index]++;
        }
        testContainsFlush();
        if (containsFlush){
            testFlushRanks();
            testContainsStraightFlush();
            if (containsStraightFlush){
                handType = HandType.STRAIGHTFLUSH;
                testKickers();
                return;
            }
        }
        mostFrequentRank=-1;
        secondMostFrequentRank=-1;
        for(int i=0;i<rankCounts.length;i++){
            if ((mostFrequentRank==-1)||rankCounts[i]>=rankCounts[mostFrequentRank]){
                secondMostFrequentRank = mostFrequentRank;
                mostFrequentRank = i;
            } else if ((secondMostFrequentRank==-1)||rankCounts[i]>=rankCounts[secondMostFrequentRank]){
                secondMostFrequentRank = i;
            }
        }
        
        testContainsStraight();
        
        if (rankCounts[mostFrequentRank]==4){
            handType=HandType.FOUROFAKIND;
        } else if (rankCounts[mostFrequentRank]==3&& rankCounts[secondMostFrequentRank]>=2){
            handType=HandType.FULLHOUSE;
        } else if (containsFlush){
            handType=HandType.FLUSH;
        } else if (containsStraight){
            handType=HandType.STRAIGHT;
        } else if (rankCounts[mostFrequentRank]==3){
            handType=HandType.THREEOFAKIND;
        } else if (rankCounts[mostFrequentRank]==2){
            if (rankCounts[secondMostFrequentRank]==2){
                handType=HandType.TWOPAIR;
            } else {
                handType=HandType.PAIR;
            }
        } else {
            handType=HandType.HIGHCARD;
        }
        testKickers();
    }
    
    public void testContainsFlush(){
        for(int i=0;i<suitCounts.length;i++){
            if (suitCounts[i]>=5){
                containsFlush = true;
                flushSuit = i;
                return;
            }
        }
        containsFlush = false;
    }
    
    public void testContainsStraight(){
        int runningCount = 0;
        for(int i=rankCounts.length-1;i>=0;i--){
            if (rankCounts[i]>0){
                runningCount++;
                if (runningCount==5){
                    straightHighCard = i+5;
                    containsStraight = true;
                    return;
                }
            } else {
                runningCount=0;
            }
        }
        if (runningCount==4 && rankCounts[12]>0){
            straightHighCard=3;
            containsStraight = true;
        } else {
          containsStraight = false;
        }
    }
    
    
    public void testFlushRanks(){
        flushRanks = new boolean[rankCounts.length];
        for(int i=0;i<flushRanks.length;i++){
            flushRanks[i]=false;
        }
        for(int i=0;i<cards.length;i++){
            if (cards[i].suit.index==flushSuit){
                flushRanks[cards[i].rank.index]=true;
            }
        }
    }
    
    public void testContainsStraightFlush(){
        int runningCount = 0;
        for(int i=flushRanks.length-1;i>=0;i--){
            if (flushRanks[i]){
                runningCount++;
                if (runningCount==5){
                    straightHighCard = i+5;
                    containsStraightFlush = true;
                    return;
                }
            } else {
                runningCount=0;
            }
        }
        if (runningCount==4 && flushRanks[12]){
            straightHighCard=3;
            containsStraightFlush = true;
        } else {
          containsStraightFlush = false;
        }
    }
    
    public void testKickers(){
        switch(handType){
            case STRAIGHTFLUSH:
            case FULLHOUSE:
            case FLUSH:
            case STRAIGHT:
                kickers = new int[0];
                return;
            case TWOPAIR:
                kickers = new int[1];
                for(int i=rankCounts.length-1;i>=0;i--){
                    if (i!=mostFrequentRank&&i!=secondMostFrequentRank&&rankCounts[i]>0){
                        kickers[0]=i;
                        return;
                    }
                }
                throw new RuntimeException("Reached a strange place in HandAnalysis.");
            default:
            //case HIGHCARD:
            //case PAIR:
            //case THREEOFAKIND:
            //case FOUROFAKIND:
                int currentIndex = 0;
                kickers = new int[5-rankCounts[mostFrequentRank]];
                for(int i=rankCounts.length-1;i>=0;i--){
                    if (i!=mostFrequentRank&&rankCounts[i]>0){
                        kickers[currentIndex++]=i;
                        if (currentIndex==kickers.length){
                            return;
                        }
                    }
                }
                throw new RuntimeException("Reached a strange place in HandAnalysis(2).");

        }
    }
    
    /** Determines the winner's index.
     * -1 is a tie, 
     * 0 is the first player wins,
     * and 1 is the second player wins.
     * This function assumes that there are only two players.
     */
    public static int determineWinner(String[] hole, String board){
        Card[][] holeCards = new Card[2][];
        Card[] boardCards = Card.toCardArray(board);
        holeCards[0]=Card.toCardArray(hole[0]);
        holeCards[1]=Card.toCardArray(hole[1]);
        return determineWinner(holeCards,boardCards);
    }
    

    /**
     * Determine which players win or tie.
     */
    public static boolean[] determineWinners(Card[][] hole, Card[]
    board){
      Card[][] playersToCompare = new Card[2][];
      playersToCompare[0]=hole[0];
      boolean[] winner = new boolean[hole.length];
      for(int i=0;i<winner.length;i++){
        winner[i]=false;
      }
      winner[0]=true;
      for(int i=1;i<hole.length;i++){
        playersToCompare[1] = hole[i];
	int winnerIndex = determineWinner(playersToCompare,board);
	if (winnerIndex==-1){
	  winner[i]=true;
	} else if (winnerIndex==1){
	  winner[i]=true;
	  for(int j=0;j<i;j++){
	    winner[j]=false;
	  }
	  playersToCompare[0]=hole[i];
	}
      }
      return winner;
    }
    /** Determines the winner's index.
     * -1 is a tie, 
     * 0 is the first player wins,
     * and 1 is the second player wins.
     * This function assumes that there are only two players.
     */
    public static int determineWinner(Card[][] hole, Card[] board){
        Card[][] playerHands;
        playerHands = new Card[2][7];
        for(int i=0;i<2;i++){
            for(int j=0;j<2;j++){
                playerHands[i][j]=hole[i][j];
            }
            for(int j=0;j<5;j++){
                playerHands[i][j+2]=board[j];
            }
        }
        
        HandAnalysis seat0 = new HandAnalysis(playerHands[0]);
        HandAnalysis seat1 = new HandAnalysis(playerHands[1]);
        if (seat0.handType.strongerThan(seat1.handType)){
            return 0;
        }
        if (seat1.handType.strongerThan(seat0.handType)){
            return 1;
        }
        switch(seat0.handType){
        case STRAIGHTFLUSH:
        case STRAIGHT:
            if (seat0.straightHighCard>seat1.straightHighCard){
                return 0;
            }
            if (seat1.straightHighCard>seat0.straightHighCard){
                return 1;
            }
            return -1;
        case FLUSH:
            int numCards=0;
            for(int i=12;i>=0;i--){
                if(seat0.flushRanks[i]){
                    if (seat1.flushRanks[i]){
                        numCards++;
                        if (numCards==5){
                            return -1;
                        }
                    } else {
                        return 0;
                    }
                } else if (seat1.flushRanks[i]){
                    return 1;
                }
            }
            return -1;
        case FOUROFAKIND:
        case THREEOFAKIND:
        case PAIR:
        case HIGHCARD:
            if (seat0.mostFrequentRank>seat1.mostFrequentRank){
                return 0;
            } 
            if (seat0.mostFrequentRank<seat1.mostFrequentRank){
                return 1;
            }
            return kickerTest(seat0,seat1);
        case FULLHOUSE:
        case TWOPAIR:
            if (seat0.mostFrequentRank>seat1.mostFrequentRank){
                return 0;
            }
            if (seat0.mostFrequentRank<seat1.mostFrequentRank){
                return 1;
            }
            if (seat0.secondMostFrequentRank>seat1.secondMostFrequentRank){
                return 0;
            }
            if (seat0.secondMostFrequentRank<seat1.secondMostFrequentRank){
                return 1;
            }
            return kickerTest(seat0,seat1);
        }
        throw new RuntimeException("Reached a strange place in HandAnalysis(3).");
    }
    
    public static int kickerTest(HandAnalysis seat0, HandAnalysis seat1){
        for(int i=0;i<seat0.kickers.length;i++){
            if (seat0.kickers[i]>seat1.kickers[i]){
                return 0;
            }
            if (seat0.kickers[i]<seat1.kickers[i]){
                return 1;
            }
        }
        return -1;   
    }    
}
