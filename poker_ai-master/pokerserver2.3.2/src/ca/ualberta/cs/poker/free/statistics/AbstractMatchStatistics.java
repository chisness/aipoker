package ca.ualberta.cs.poker.free.statistics;

import java.util.Vector;


public interface AbstractMatchStatistics {

	
	public int getSmallBlindsInASmallBet();
	
	public Vector<String> getPlayers();


	/**
	 * chsmith use this for RandomVariable
	 * Gets the utility in small blinds for player
	 * @param player the player whose utility we are interested in
	 * @param opponent her opponent
	 * @param firstHand the first hand to consider
	 * @param lastHand the last hand to consider
	 */
	public int getUtility(String player, String opponent, int firstHand, int lastHand);
	
	/**
	 * chsmith use this for RandomVariable
	 * @param player
	 * @param opponent
	 * @return
	 */
	public boolean isDefined(String player, String opponent);
	
	/**
	 * Tests if two matches could be duplicate based upon 
	 * the players and the cards.
	 * @param other the match to compare to
	 * @return whether it is possible if the matches could be duplicate
	 */
	public boolean isDuplicate(AbstractMatchStatistics other);
	
	/**
	 * Tests if two matches could be duplicate based upon 
	 * the players and the cards.
	 * @param other the match to compare to
	 * @return whether it is possible if the matches could be duplicate
	 */
	public boolean isDuplicateCards(AbstractMatchStatistics other);
	
	public int getFirstHandNumber();
	
	public int getLastHandNumber();
	
	public int getNumberOfHands();
	
}