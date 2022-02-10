package ca.ualberta.cs.poker.free.alien;

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;


/**
 * Represents the data for an account for a member of a team.
 */ 
public class AlienAccount{
/**
 * Generate an account from a line in a profile
 * @param str A string of the form {@literal <team> <username> <password>}
 */
	public AlienAccount(String str) throws IOException{
		StringTokenizer st = new StringTokenizer(str);
		try{
		team = st.nextToken();
		username = st.nextToken();
		password = st.nextToken();
		email = st.nextToken();
		while(st.hasMoreTokens()){
			String token = st.nextToken();
			if (token.equalsIgnoreCase("teamleader")){
				teamLeader = true;
			}
			if (token.equalsIgnoreCase("superuser")){
				superuser = true;
			}
		}
		} catch (NoSuchElementException nse){
			throw new IOException("Error parsing AlienAccount "+str);
		}
	}
	
	
	
  public String username;
  public String password;
  public String team;
  public String email;
  /**
   * Team leaders can change passwords of team members
   */
  public boolean teamLeader;
  public boolean superuser;
  
  public String toString(){
	  return team+" "+
	  username+" "+
	  password+" "+
	  team+" "+
	  email+" "+
	  ((teamLeader) ? "TEAMLEADER ": "")+
	  ((superuser) ? "SUPERUSER":"");
	  
  }

public AlienAccount(String username, String password, String team, String email, boolean teamLeader, boolean superuser) {
	this.username = username;
	this.password = password;
	this.team = team;
	this.email = email;
	this.teamLeader = teamLeader;
	this.superuser = superuser;
}
};
