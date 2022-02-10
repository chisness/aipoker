package ca.ualberta.cs.poker.free.ringseat;
import java.security.SecureRandom;
import java.util.*;

import ca.ualberta.cs.poker.free.tournament.*;

/**
 * This class will produce and verify permutations for an arbitrary number
 * of players. To be used to create ring policies using permutations
 * to determine seating order.
 * 
 * @author Christian Smith
 * @author Martin Zinkevich
 */
public class Permutation{
  /**
   * players[i] is the index of
   * the player in the ith seat
   */
  public int[] players;


  /**
   * Check to make sure all players are used exactly once and that
   * the permutation is not the identity, which is not a valid
   * permutation
   */
  public boolean checkValid(){
   
	int[] playersWorking = players.clone();
	
	Permutation identity = new Permutation(players.length);
	
    Arrays.sort( playersWorking );
    
    boolean valid = true;
   
    // for each player, make sure its used only once
    for ( int i =0; i < playersWorking.length - 1; i ++ ) {
    	if ( playersWorking[i] == playersWorking[i+1] ) {
    		valid = false;
    	}
    }
    
    
    // make sure 0 to numPlayers - 1 is used at least once
    // if there are no duplicates, and the sorted permutation is
    // equal to the identity, once and only once is satisfied
    
    Permutation sortedPerm = new Permutation(playersWorking);
    System.out.println( sortedPerm.toString());
    
    if( ! sortedPerm.equals( identity )) valid = false;
    
    return valid;
   
  }
  /**
   * Construct a permutation from a string.
   * String is of the form:
   * &lt;players[0]&gt; &lt;players[1]&gt; &lt;players[2]&gt;...
   */
  public Permutation(String str){
    
	  // tokenize on " ", default behaviour
	  StringTokenizer st = new StringTokenizer( str );
	  
	  // initilise the array before filling it
	  players = new int[ st.countTokens()];
	  
	  int i = 0;
	  while ( st.hasMoreTokens() ) {
		  players[i++] = (int)Integer.parseInt( st.nextToken());  
	  }
	  
  }

  /**
   * Construct a random permutation from SecureRandom with 
   * numPlayers number of players.
   * 
   * @param numPlayers is the number of players.
   * @param random SecureRandom object for number generation
   */
  public Permutation(int numPlayers, SecureRandom random){
	  
	  // initialise array
	  players = new int[ numPlayers ];
	  
	  // treeset for quicker searching of the set for duplicates
	  TreeSet<Integer> picked = new TreeSet<Integer>();
	  

	  for ( int i = 0; i < numPlayers; i ++ ) {  
		  boolean found = false;
		  
		  while ( !found ) {
		  
			  // pick randomly
			  int candidate = random.nextInt(numPlayers); 
			  
			  // maintain uniqueness
			  if ( ! picked.contains(candidate)) {
				  players[i] = candidate;
				  picked.add( candidate );
				  
				  found = true;
			  }
		  }
		  
	  }
  }

  /**
   * Construct the identity permutation.
   * @param numPlayers is the number of players.
   */
  public Permutation(int numPlayers){
    players = new int[numPlayers];
    for(int i=0;i<numPlayers;i++){
      players[i] = i;
    }
  }

  /**
   * Raw field constructor
   * @param players players[i] is the player index in the ith seat.
   */
  public Permutation(int[] players){
    this.players = players;
  }
  
  /**
   * Shift all the players left
   * If this permutation is 0 1 2 3 4,
   * then the output should be 1 2 3 4 0.
   * This has the effect of moving the button clockwise.
   */
  public Permutation shiftLeft(){
    int[] result = new int[players.length];
    for(int i=0;i<players.length-1;i++){
      result[i]=players[i+1];
    }
    result[result.length-1]=players[0];
    return new Permutation(result);
  }

  /**
   * Returns the player in the seat
   */
  public int getPlayerFromSeat(int seat){
    return players[seat];
  }

  /**
   * Returns the seat the player is in.
   */
  public int getSeatFromPlayer(int player){
    
	  int ret = -1;
	  
	  for( int i = 0; i < players.length; i ++ ) {
		  if( players[i] == player ) ret = i;
	  }
	  
	  return ret;
  }

  /**
   * Return the number of players.
   */
  public int getNumberPlayers(){
    return players.length;
  }

  /**
   * Given a vector bots where bots[i] is the ith player,
   * returns a new vector such that result[i] is the ith seat.
   * The original vector is unchanged.
   */
  public Vector<BotInterface> mapPlayersToSeats(Vector<BotInterface> bots){
    //throw new RuntimeException("Not implemented");
    Vector<BotInterface> result = new Vector<BotInterface>();
	  
	for ( int i = 0; i < bots.size(); i++ ) {
		result.add( bots.get(players[i]));
	}
	 
	return result;
	  
  }

  /**
   * Determine if two permutations are equal without resorting to strings
   * @param p Permutation to compare
   * @return true if equal
   */
  public boolean equals( Permutation p ) {
	  
	  boolean equal = true;
	  
	  for( int i = 0; i < p.getNumberPlayers(); i++ ) {
		  
		  if ( players[i] != p.players[i]) {
			  equal = false;
		  }
	  }
	  return equal;
  }
  
  /**
   * Make a string that can be read in by the constructor.
   */
  public String toString(){
    String ret = new String();
    
    for ( int i = 0; i < players.length; i++ ) {
    	ret += Integer.toString(players[i]) + " ";
    }
    
    return ret;
  }
}
