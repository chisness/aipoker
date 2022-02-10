package ca.ualberta.cs.poker.free.ringseat;
import java.util.*;
import java.io.*;
import java.security.SecureRandom;
import ca.ualberta.cs.poker.free.tournament.*;

/**
 * This class represents a set of permutations. These
 * permutations may be generated at random, or 
 * designed to reduce variance.
 * 
 * @author Christian Smith
 * @author Martin Zinkevich
 */
public class RingPolicy{

  /**
   * The permutations used in this policy
   * They should all have the same number of players.
   */
  Vector<Permutation> permutations;

  /**
   * Basic constructor.
   */
  public RingPolicy(Vector<Permutation> permutations){
    this.permutations = permutations;
  }

  /**
   * Check to make sure all permutations are valid,
   * and that they all have the same number of players.
   */
  public boolean checkValid(Vector<Permutation> permutations){
    
	  boolean valid = true;
	  boolean numPlayersConsistant = true;
	  
	  if ( permutations != null ) {
	  
		  // check against the first element
		  int initialNumPlayers = permutations.firstElement().getNumberPlayers();
		  
		  for( Permutation p:permutations) {
			  // each permutation should be valid
			  if( p.checkValid() != true ){
				  valid = false;
			  }
			  // if any permutation has any other number of players
			  if( p.getNumberPlayers() != initialNumPlayers ) {
				  numPlayersConsistant = false;
			  }
		  }
	  }
	  else valid = false;
	  
	  
	  return valid && numPlayersConsistant;
  }
  
  /**
   * Check to make sure all permutations on the current ring policy are valid,
   * and that they all have the same number of players.
   */
  public boolean checkValid(){
    
	return checkValid(permutations);  
  }
  
  /**
   * Construct a ring policy from a file.
   * If encounters a line with POLICY_END, terminates.
   * Also terminate at end of file.
   */
  public RingPolicy(BufferedReader br){
   
    try {
    	
    	String str;
    	Vector<Permutation> perms = new Vector<Permutation>();
    	str = br.readLine();
		
    	if ( str != null ) { 
    		
    		// this will ignore POLICY_BEGIN and any other string, \n for example
	    	if ( str.contains("POLICY_BEGIN")) str = br.readLine();
	    	
	    	
	    	// read until the EOF or POLICY_END
	    	while ( (str != null && !str.contains( "POLICY_END" )) ) {
				
				Permutation candidate = new Permutation(str);
				if ( candidate.checkValid() ) {	
					perms.add( candidate );
				}

				// read the next line, null if EOF
				str = br.readLine();
			}
			permutations = perms;
	    }
    	
    } catch (IOException e) {
		System.err.println( "Exception while reading ringpolicy from file");
		//e.printStackTrace();
	}
 
  }
  
  /**
   * Generate a RingPolicy consisting of random permutations.
   * @param numPlayers the number of players in each permutation.
   * @param numGames the number of permutations.
   * @param random the randomness to use.
   */
  public RingPolicy(int numPlayers, int numPermutations, SecureRandom random){
      
	  Vector<Permutation> perm = new Vector<Permutation>();
	  
	  for ( int i = 0; i < numPermutations; i++ ) {
		  perm.add( new Permutation(numPlayers, random));
	  }
	  
	  permutations = perm;
	  
  }

  /**
   * The default policy is a rotation policy.
   * For five players, this looks like:<BR>
   * 0 1 2 3 4<BR>
   * 1 2 3 4 0<BR>
   * 2 3 4 0 1<BR>
   * 3 4 0 1 2<BR>
   * 4 0 1 2 3<BR>
   */
  public static RingPolicy getRotationPolicy(int numPlayers){
    Vector<Permutation> result=new Vector<Permutation>();
    // Start with the identity.
    Permutation current = new Permutation(numPlayers);
    for(int i=0;i<numPlayers;i++){
      result.add(current);
      // Shift the players in the current permutation left
      current = current.shiftLeft();
    }
    return new RingPolicy(result);
  }
  
  /**
   * Writes the policy to a file.
   * Do not include POLICY_BEGIN or POLICY_END.
   */
  public void write(Writer w) throws IOException{
    
	  BufferedWriter out = new BufferedWriter(w);
      
	  for ( Permutation p:permutations) {
		  out.write(p.toString() + "\n");  
	  }
      
	  out.close();
  }

  /**
   * Get the number of players for this policy.
   * Returns the number of players in the first permutation.
   */
  public int getNumberPlayers(){
    return permutations.get(0).getNumberPlayers();
  }

  /**
   * Read a sequence of policies from a file.
   * Policies begin with a line, POLICY_BEGIN, and end
   * with a line POLICY_END.
   */
  public static Vector<RingPolicy> read(String filename) throws  IOException{
    
    BufferedReader buffread = new BufferedReader(new FileReader( filename ));
	Vector<RingPolicy> policies = new Vector<RingPolicy>();
	
	RingPolicy policy = new RingPolicy( buffread );
	
	// the last policy will have permutations equal to null, and fail validity
	while ( policy.checkValid() ) {
		policies.add( policy );
		
		policy = new RingPolicy( buffread );
	}
	
	buffread.close();
	
	
//	str = br.readLine();
//	while ( (str != null ) ) {
//		
//		// begin new policy
//		if ( str.contains("POLICY_BEGIN") ) {
//			
//			//use .contains to avoid any endline wierdness
//			str = br.readLine();
//			while ( ! str.contains("POLICY_END")) {
//				perms.add( new Permutation(str));
//				str = br.readLine();
//			}
//			
//			count++;
//			policies.add(( new RingPolicy( perms ) ) );
//			
//		}
//		str = br.readLine();
//	}
//		
//	br.close();
//	
//	System.out.println( "Loaded " + count + " policies");
	
	return policies;
  }

  /**
   * Selects a policy based upon the number of players.
   */
  public static RingPolicy selectPolicy(Vector<RingPolicy> policies,
  Vector<BotInterface> players){
    for(RingPolicy policy:policies){
      if (policy.getNumberPlayers()==players.size()){
        return policy;
      }
    }
    return getRotationPolicy(players.size());
  }

  /**
   * Maps the players to seats according to all the permutations
   * in the policy.
   */
  public Vector<Vector<BotInterface> >mapPlayersToSeats(Vector<BotInterface> players){
    Vector<Vector<BotInterface> > result = new 
    Vector<Vector<BotInterface> > ();
    for(Permutation p:permutations){
      result.add(p.mapPlayersToSeats(players));
    }
    return result;
  }

  /**
   * Convert the permutations to a printable string
   */
  public String toString() {
	  String ret = new String();
	  if (permutations==null){
		  return "RingPolicy is null";
	  }
	  for(Permutation p:permutations){ 
		  ret += p.toString() + "\n";
	  }
	  return ret;
  }
}