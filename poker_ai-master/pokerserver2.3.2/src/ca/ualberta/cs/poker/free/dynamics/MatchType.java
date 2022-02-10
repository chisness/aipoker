package ca.ualberta.cs.poker.free.dynamics;

/**
 * This class represents the constants which define the match, both
 * at the level of the hand and the match as a whole.
 * @author maz
 *
 */
public class MatchType {
	public int doyleLimit=400;
	

    /**
     * The time per hand (in milliseconds)
     */
    public int timePerHand = 7000;


    /**
     * The number of hands in a match
     */
    public int numHands = 1000;
    
    

    /**
     * If chess clock is true, then there is a TOTAL time for a match.
     * If chess clock is false, then there is a TOTAL time for a hand.
     */
    public boolean chessClock = true;
    
    /**
     * What types of limits are there on the game?
     */
	public LimitType limitGame;

    
    /**
     * The size of the small blind in CHIPS
     */  
    public int smallBlindSize = 1;
    
    /**
     * The size of the big blind
     * CHIPS
     */
    public int bigBlindSize = 2;
    
    /**
     * The size of the small bet (used in
     * the first and second rounds)
     * CHIPS
     */
    public int smallBetSize = 2;
    
    /**
     * The size of the big bet
     * (used in the third and fourth rounds)
     * CHIPS
     */
    public int bigBetSize = 4;

    /**
     * Get whether the stacks are effectively infinite.
     * For Doyle's Game, this is false, because the betting
     * is explicitly capped at doyleLimit instead of the underlying
     * stack size.
     */
    public boolean stackBoundGame;

    /**
     * The initial stack size of a player in CHIPS
     */
    public int initialStackSize;
    
    public MatchType(LimitType type, boolean stackBoundGame, int initialStackSize, int numHands){
    	this.limitGame = type;
    	this.stackBoundGame = stackBoundGame;
    	this.initialStackSize = initialStackSize;
    	this.numHands = numHands;
    }
}
