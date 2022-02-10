package ca.ualberta.cs.poker.free.alien;
import java.util.*;
import java.io.*;

public class ClientMatch{

  String bot;
  String opponent;
  String game;
  String name;
  
  /**
   * Generate a new match by parameters
   * @param game the type of game
   * @param name the name of the match
   * @param bot the name of the bot
   * @param opponent the opponent (on the server)
   */
  public ClientMatch(String game, String name, String bot, String opponent){
    this.bot = bot;
    this.opponent = opponent;
    this.game = game;
    this.name = name;
  }
  
  /**
   * Create a client from a string
   * Note that gametype is HEADSUPLIMIT for now.
   * {@literal <game> <name> <bot> <opponent>}
   * @param str
   */
  public ClientMatch(String str) throws IOException{
	  try{
    StringTokenizer st = new StringTokenizer(str);
    game = st.nextToken();
    name = st.nextToken();
    bot = st.nextToken();
    opponent = st.nextToken();
	  } catch (NoSuchElementException e){
		  throw new IOException("Could not parse as a match:"+str);
	  }
  }
  public ClientMatch(ClientMatch other){
    this(other.game,other.name,other.bot,other.opponent);
  }
  
  public ClientMatch(ClientMatch other, int repeatCount){
	    this(other.game,other.name+repeatCount,other.bot,other.opponent);	  
  }
  
  public String toString(){
	  return game + " "+name+" "+bot+" "+opponent;
  }

  public String matchRequest() {
	return "MATCHREQUEST:"+game+":"+name+":"+bot+":"+opponent;
  }

  
  /**
   * A client match is a duplicate match, and this returns the match pair.
   */
  public Vector<String> getMatchNames(){
	  String forward = name+"."+bot+"."+opponent;
	  String reverse = name +"."+opponent+"."+bot;
	  Vector<String> result = new Vector<String>();
	  result.add(forward);
	  result.add(reverse);
	  return result;
  }
}
