/*
 * Card.java
 * 
 * This is an implementation of a card in a standard
 * deck, adapted from the basic Java example for enumeration.
 * In the future, reverting to a more standard representation
 * might be helpful.
 *
 * Created on April 18, 2006, 11:31 AM
 */

package ca.ualberta.cs.poker.free.dynamics;
import java.security.SecureRandom;

/**
 *
 * @author Martin Zinkevich
 */
public class Card {
    public static final int DECKSIZE = 52;
    /**
     * An enumeration class for Rank.
     * It is unclear if there is a significant gain from this, and it may be deprecated in future versions.
     */
    public enum Rank{
        TWO ('2',0), THREE ('3',1), FOUR ('4',2), FIVE ('5',3), SIX ('6',4), SEVEN ('7',5), EIGHT ('8',6), NINE ('9',7), TEN ('T',8), JACK ('J',9), QUEEN ('Q',10), KING ('K',11), ACE ('A',12);
        public char representation;
        public int index;
        Rank(char representation, int index){
            this.representation = representation;
            this.index = index;
        }
        
        /**
         * A one character string representation.
         * 2-9,T,J,Q,K
         */
        public String toString(){
          return ""+representation;
        }
        
        /**
         * toRank is implemented in a slow fashion: could be optimized.
         * returns 0-12
         */
        public static Rank toRank(char c){
            for(Rank r:values()){
                if (r.representation==c){
                    return r;
                }
            }
            throw new RuntimeException("Did not find rank of card.");
        }
        
        public static Rank toRank(int i){
        	for(Rank r:values()){
        		if (r.index==i){
        			return r;
        		}
        	}
        	throw new RuntimeException("Did not find rank of card.");
        }
        
    }
    
    /**
     * An enumeration class for Suit.
     * It is unclear if there is a significant gain from this, and it may be deprecated in future versions.
     */
    public enum Suit{
        CLUBS('c',0), DIAMONDS('d',1), HEARTS('h',2), SPADES('s',3);
        public char representation;
        public int index;
        Suit(char representation, int index){
            this.representation = representation;
            this.index = index;
        }
        public String toString(){
          return ""+representation;
        }
        
        public static Suit toSuit(char c){
            for(Suit s:values()){
                if (s.representation==c){
                    return s;
                }
            }
            throw new RuntimeException("Did not find suit of card.");
        }
        
        public static Suit toSuit(int i){
            for(Suit s:values()){
                if (s.index==i){
                    return s;
                }
            }
            throw new RuntimeException("Did not find suit of card.");
        }

    }
    public Rank rank;
    public Suit suit;
    /** Creates a new instance of Card */
    public Card(Rank rank, Suit suit) {
        this.rank = rank;
        this.suit = suit;
    }
    
    /**
     * Initializes a card from a rank/suit card string.
     */
    public Card(String cardString){
        rank = Rank.toRank(cardString.charAt(0));
        suit = Suit.toSuit(cardString.charAt(1));
    }
    
    /**
     * Converts a string of cards to a card array
     */
    public static Card[] toCardArray(String cards){
        Card[] result = new Card[cards.length()/2];
        int index=0;
        for(int i=0;i<cards.length();i+=2,index++){
            result[index]=new Card(cards.substring(i,i+2));
        }
        return result;
    }
    
    public static String arrayToString(Card[] cards){
    	String result = "";
    	for(Card c:cards){
    		result+=c;
    	}
    	return result;
    }
    public static Card[] getAllCards(){
        Card[] deck = new Card[DECKSIZE];
        int index = 0;
        for(Rank currentRank: Rank.values()){
            for(Suit currentSuit: Suit.values()){
              deck[index++]=new Card(currentRank,currentSuit);
            }
        }
        return deck;
    }
    /** Deals a certain number of cards into an array. */
    public static Card[] dealNewArray(SecureRandom random, int numCardsToDeal){
        Card[] deck = getAllCards();
        
        for(int i=0;i<numCardsToDeal;i++){
            int toSwap = random.nextInt(DECKSIZE-i)+i;
            Card temp = deck[i];
            deck[i] = deck[toSwap];
            deck[toSwap] = temp;
        }
        Card[] result = new Card[numCardsToDeal];
        for(int i=0;i<numCardsToDeal;i++){
            result[i]=deck[i];
        }
        return result;
    }
    
    /** Combine two cards using modulo addition of rank and card  */
    public static Card[] combine(Card[] first, Card[] second){
    	assert(first.length==second.length);
    	int[] firstIndices = cardsToExclusiveRankMajorIndex(first);
    	int[] secondIndices = cardsToExclusiveRankMajorIndex(second);
    	int[] resultIndices = new int[first.length];
    	for(int i=0;i<first.length;i++){
    		resultIndices[i]=(firstIndices[i]+secondIndices[i]) % (DECKSIZE-i);
    	}
    	return exclusiveRankMajorIndexToCards(resultIndices);
    }
    
    /**
     * Represents each card with its order in terms of untaken cards.
     * First, the cards are converted to rank major indices.
     * Suppose that this step yields:<BR>
     * 13 4 17 6 29<BR>
     * Then, we subtract from each index the number of cards that
     * precede it and have a lower index. Thus, the above
     * becomes:<BR>
     * 13 4 15 5 25<BR>
     * This should be read, "The thirteenth untaken card,
     * the fourth untaken card, the fifteenth untaken card," et cetera.
     * The advantage of this representation is that, for cards drawn
     * from a deck, the first number is
     * uniformly at random between 0 and 51 inclusive, the second is 
     * uniformly at random between 0 and 50 inclusive, the third is uniformly
     * at random between 0 and 49 inclusive, et cetera.
     * @param cards
     * @return
     */
    public static int[] cardsToExclusiveRankMajorIndex(Card[] cards){
    	// initialArray is given the raw rank major index of each card
    	int[] initialArray = new int[cards.length];
    	for(int i=0;i<cards.length;i++){
    		initialArray[i]=cards[i].getIndexRankMajor();
    	}
    	// finalArray is the exclusive rank major index of each card
    	int[] finalArray = new int[cards.length];
    	System.arraycopy(initialArray, 0, finalArray, 0, cards.length);
    	for(int i=0;i<cards.length;i++){
    		for(int j=0;j<i;j++){
    			if (initialArray[j]<initialArray[i]){
    				finalArray[i]--;
    			}
    		}
    	}
    	return finalArray;
    }
    
    /**
     * The inverse of Card.exclusiveRankMajorIndexToCards().
     * @see Card#exclusiveRankMajorIndexToCards(Card[])
     * @param cards
     * @return
     */
    public static Card[] exclusiveRankMajorIndexToCards(int[] indices){
    	// These have the same meaning as above.
    	Card[] cards = new Card[indices.length];
    	int[] finalArray = indices;
    	int[] initialArray = new int[cards.length];
    	boolean[] usedIndex = new boolean[DECKSIZE];
    	for(int i=0;i<usedIndex.length;i++){
    		usedIndex[i]=false;
    	}
    	
    	for(int i=0;i<cards.length;i++){
    		
    		int unusedIndex=finalArray[i];
    		int absoluteIndex = 0;
    		
    		while(true){
    			while(usedIndex[absoluteIndex]){
    				absoluteIndex++;
    			}
    			if (unusedIndex==0){
    				break;
    			}
    			absoluteIndex++;
    			unusedIndex--;
    		}
    		initialArray[i]=absoluteIndex;
    		usedIndex[absoluteIndex]=true;
    	}
    	for(int i=0;i<cards.length;i++){
    		cards[i]=getCardFromIndexRankMajor(initialArray[i]);
    	}
    	return cards;
    }

    public static Card getCardFromIndexRankMajor(int index){
    	int rankIndex = index/4;
    	int suitIndex = index % 4;
    	return new Card(Rank.toRank(rankIndex),Suit.toSuit(suitIndex));
    }
    
    public static Card getCardFromIndexSuitMajor(int index){
    	int rankIndex = index % 13;
    	int suitIndex = index / 13;
    	return new Card(Rank.toRank(rankIndex),Suit.toSuit(suitIndex));
    }
    
    
    public int getIndexSuitMajor(){
		return (suit.index * 13) + rank.index;
	}
    
	public int getIndexRankMajor(){
		return (rank.index * 4) + suit.index;
	}
    /**
     * Returns a string with the rank and suit.
     */
    public String toString(){
        return "" + rank + suit;
    }
    
    public boolean equals(Object other){
    	if (other instanceof Card){
    		Card otherCard = (Card)other;
    		return otherCard.rank==rank && otherCard.suit==suit;
    	}
    	return false;
    }
    /**
     * Tests the conversion from cards to exclusive rank major index 
     * back to cards
     */
    public static void main(String[] args){
    	SecureRandom r = new SecureRandom();
    		Card[] deck = getAllCards();
    	for(Card original:deck){
    		int index = original.getIndexRankMajor();
    		Card result = Card.getCardFromIndexRankMajor(index);
    		if (!original.equals(result)){
    			System.err.println("From "+original+" to "+index+" to "+result);
    			System.exit(0);
    		}
    		if (index<0 || index>=52){
    			System.err.println("From "+original+" to "+index+" to "+result);
    			System.exit(0);    			
    		}
    	}
    	System.err.println("Indexing stable");
    	for(int i=0;i<100000;i++){
    		Card[] original = dealNewArray(r,9);
    		int[] indices = Card.cardsToExclusiveRankMajorIndex(original);
    		Card[] result = Card.exclusiveRankMajorIndexToCards(indices);
    		for(int j=0;j<original.length;j++){
    			if (!result[j].equals(original[j])){
    				
    				System.err.println("Original:"+arrayToString(original));
    				for(int k=0;k<original.length;k++){
    					
    					System.err.print(" "+original[k].getIndexRankMajor());
    				}
    				System.err.println();
    				System.err.print("Indices:");
    				for(int k=0;k<indices.length;k++){
    					System.err.print(" "+indices[k]);
    				}
    				System.err.println();
    				System.err.println("Result:"+arrayToString(result));
    				for(int k=0;k<result.length;k++){	
    					System.err.print(" "+result[k].getIndexRankMajor());
    				}
    				System.err.println();
    				System.exit(0);
    			}
    		}		
    	}
    	System.err.println("No errors observed");
    }
    
}
