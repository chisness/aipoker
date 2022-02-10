package ca.ualberta.cs.poker.free.statistics;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;

import ca.ualberta.cs.poker.free.dynamics.LimitType;

/**
 * A variant of MatchStatistics constructed from the
 * .res (result) file. Works only with latest code.
 * 1. Calculates identical cards by analyzing filenames.
 * 2. Reads players and winnings out of .res file.
 * 3. Is faster than MatchStatistics.
 * 4. Has a much smaller memory footprint.
 * 5. Can't analyze intermediate statistics.
 */
public class LightMatchStatistics implements AbstractMatchStatistics{
	LimitType limitType;
    /**
     * stackBound is not initialized yet
     */
	boolean stackBound;
	
	Vector<String> players;
	Vector<Integer> netSmallBlinds;
	String filename;
	
	int numHands;
	
	public int getSmallBlindsInASmallBet(){
		return 2;
	}
	
	public String toString(){
		
		String result = filename+"\n";
		for(int i:netSmallBlinds){
			result+=i+"|";
		}
		result += "\n";
		for(String s:players){
			result +=s+"|";
		}
		result+="\n";
		result+="Number_of_hands:"+numHands+"\n";
		result+="Limit Type:"+limitType+"\n";
		result+="Competition Name:"+getCompetitionName()+"\n";
		result+="Match Index:"+getMatchIndex()+"\n";
		return result;
	}
	
	public static void main(String[] args) throws FileNotFoundException,IOException{
		LightMatchStatistics stats = new LightMatchStatistics("data\\results\\limittest.HyperboreanA.Quick.match1fwd.res");
		System.out.println(stats.toString());
	}
	public LightMatchStatistics(String resfile) throws FileNotFoundException, IOException{
		//System.err.println("Reading "+logfile);
		filename = resfile;
		BufferedReader reader = new BufferedReader(new FileReader(new File(resfile)));
		//hands = new Vector<HandStatistics>();
		read(reader);
		reader.close();
	}
	
	public String getCompetitionName(){
		int initialChar = Math.max(filename.lastIndexOf("/"),filename.lastIndexOf("\\"));
		int finalChar = filename.indexOf(".",initialChar+1);
		return filename.substring(initialChar+1,finalChar);
	}
	
	public int getMatchIndex(){
		int lastPeriod = filename.lastIndexOf(".");
		int secondLastPeriod = filename.lastIndexOf(".",lastPeriod-1);
		
		String part = filename.substring(secondLastPeriod+1,lastPeriod);
		
		int firstChar = part.lastIndexOf("h");
		int lastChar = Math.max(part.indexOf("r"),part.indexOf("f"));
		return Integer.parseInt(part.substring(firstChar+1,lastChar));
	}
	
	static Vector<String> splitLine(String line){
		int currentBar = -1;
		int nextBar = -1;
		Vector<String> result=new Vector<String>();
		do{
			nextBar = line.indexOf("|",currentBar+1);
			if (nextBar==-1){
				result.add(line.substring(currentBar+1));
			} else {
				result.add(line.substring(currentBar+1,nextBar));
				currentBar = nextBar;
			}
			
		} while (nextBar!=-1);
		return result;
	}
	
	static String clipFront(String line){
		return line.substring(line.indexOf(":")+1);
	}
	
	//-241|241
	//GomelNoLimit2|GomelNoLimit1
	//Number_of_hands:1000
	//LimitType:DOYLE
	//StackBounds:false
	//Timeout per hand(ms):7000
	void read(BufferedReader reader) throws IOException{
		// First line: outcomes in total small blinds separated by |
		String outcomeLine = reader.readLine();
		if( outcomeLine == null ) {
			throw new IOException("No values present");
		}
		Vector<String> outcomes = splitLine(outcomeLine);
		netSmallBlinds = new Vector<Integer>();
		for(String s:outcomes){
			netSmallBlinds.add(new Integer(s));
		}
		
		// Second line: players separated by |
		
		String playerLine = reader.readLine();
		
		if( playerLine == null ) {
			throw new IOException("No players present");
		}
		
		players = splitLine(playerLine);
		
		
		// Third line: Number of hands 
		String numHandsLine = reader.readLine();
		if (numHandsLine == null || !numHandsLine.startsWith("Number_of_hands:")){
			throw new IOException("No line for number of hands");
		}
		
		numHands = Integer.parseInt(clipFront(numHandsLine));
		
		
		// Fourth line: Limit type
		String typeLine = reader.readLine();
		if (typeLine == null || !typeLine.startsWith("LimitType:")){
			throw new IOException("No line for type");
		}
		
		limitType = LimitType.parse(clipFront(typeLine));
		
	}
	
	public Vector<String> getPlayers(){
		return players;
	}
	

	/**
	 * chsmith use this for RandomVariable
	 * Gets the utility in small blinds for player
	 * @param player the player whose utility we are interested in
	 * @param opponent her opponent
	 * @param firstHand the first hand to consider
	 * @param lastHand the last hand to consider
	 */
	public int getUtility(String player, String opponent, int firstHand, int lastHand){
		if (firstHand!=0 || lastHand!=numHands-1){
			System.err.println("firstHand:"+firstHand+" lastHand:"+lastHand+" numHands:"+numHands);
			throw new RuntimeException("Unable to access hand-specific utilities");
			
		}
		
		int index = players.indexOf(player);
		return netSmallBlinds.get(index);
	}
	/**
	 * chsmith use this for RandomVariable
	 * @param player
	 * @param opponent
	 * @return
	 */
	public boolean isDefined(String player, String opponent){
		
		return (players.contains(player)&& players.contains(opponent)&& players.size()==2);
	}
	
	
	/**
	 * Tests if two matches could be duplicate based upon 
	 * the players and the cards.
	 * @param other the match to compare to
	 * @return whether it is possible if the matches could be duplicate
	 */
	public boolean isDuplicate(LightMatchStatistics other){
		return (this.getCompetitionName().equals(other.getCompetitionName()))&&
		(this.getMatchIndex()==other.getMatchIndex());
	}
	
	/**
	 * Tests if two matches could be duplicate based upon 
	 * the players and the cards.
	 * @param other the match to compare to
	 * @return whether it is possible if the matches could be duplicate
	 */
	public boolean isDuplicateCards(LightMatchStatistics other){
		return isDuplicate(other);
	}
	
	public int getFirstHandNumber(){
		return 0;
	}
	
	public int getLastHandNumber(){
		return numHands-1;
	}
	
			
	
	public int getNumberOfHands(){
		return numHands;
	}

	public boolean isDuplicate(AbstractMatchStatistics other) {
		
		return isDuplicate((LightMatchStatistics)other);
	}

	public boolean isDuplicateCards(AbstractMatchStatistics other) {
		
		return isDuplicate((LightMatchStatistics)other);
	}
}