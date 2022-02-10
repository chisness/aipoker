package ca.ualberta.cs.poker.free.alien;
import java.util.*;
import java.io.*;

public class ClientMatchRing extends ClientMatch {

	String bot2;

	/**
	 * Generate a new match by parameters
	 * @param game the type of game
	 * @param name the name of the match
	 * @param bot the name of the bot
	 * @param bot2 the name of the second bot
	 * @param opponent the name of the 3rd bot
	 */
	public ClientMatchRing(String game, String name, String bot, String bot2, String opponent){
		super(game, name, bot, opponent);
		this.bot2 = bot2; 
	}

	/**
	 * Create a client from a string
	 * {@literal <game> <name> <bot> <bot2> <opponent>}
	 * @param str
	 */
	public ClientMatchRing(String str) throws IOException{
		super(str);
		try{
			StringTokenizer st = new StringTokenizer(str);
			game = st.nextToken();
			name = st.nextToken();
			bot = st.nextToken();
			bot2 = st.nextToken();
			opponent = st.nextToken();
		} catch (NoSuchElementException e){
			throw new IOException("Could not parse as a match:"+str);
		}
	}
	public ClientMatchRing(ClientMatchRing other){
		this(other.game,other.name,other.bot,other.bot2,other.opponent);
	}

	public ClientMatchRing(ClientMatchRing other, int repeatCount){
		this(other.game,other.name+repeatCount,other.bot,other.bot2,other.opponent);	  
	}

	public String toString(){
		return game + " "+name+" "+bot+" "+bot2+" "+opponent;
	}

  public String matchRequest() {
	return "MATCHREQUEST:"+game+":"+name+":"+bot+":"+bot2+":"+opponent;
  }


	/**
	 * A client match is a 6-way duplicate match, and this returns the 6 matches.
	 */
	public Vector<String> getMatchNames(){
		String one = name+"."+bot+"."+bot2+"."+opponent;
		String two = name+"."+bot+"."+opponent+"."+bot2;
		String three = name+"."+bot2+"."+bot+"."+opponent;
		String four = name+"."+bot2+"."+opponent+"."+bot;
		String five = name+"."+opponent+"."+bot+"."+bot2;
		String six = name+"."+opponent+"."+bot2+"."+bot;
		Vector<String> result = new Vector<String>();
		result.add(one);
		result.add(two);
		result.add(three);
		result.add(four);
		result.add(five);
		result.add(six);
		return result;
	}
}
