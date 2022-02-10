package ca.ualberta.cs.poker.free.statistics;

import java.io.FileNotFoundException;
import java.io.IOException;

public class ViewingFilter {
	String hero;
	String adversary;
	String competition;
	String directory;
	int numberOfMatches;
	boolean namesFlipped;
	int outputHand;
	double heroStackBlinds;
	double adversaryStackBlinds;
	public ViewingFilter(String directory2, String competition2, String hero2, String adversary2, boolean flipped, int numMatches) {
		this.directory = directory2;
		this.competition = competition2;
		this.hero = hero2;
		this.adversary = adversary2;
		this.namesFlipped = flipped;
		this.numberOfMatches = numMatches;
		heroStackBlinds = 1000000;
		adversaryStackBlinds = 1000000;
	}

	public void filterAll() throws FileNotFoundException, IOException{
		for(int matchIndex=0;matchIndex<numberOfMatches;matchIndex++){
			String forwardName = getFilename(matchIndex,true);
			String reverseName = getFilename(matchIndex,false);
			MatchStatistics matchForward = new MatchStatistics(forwardName);
			MatchStatistics matchReverse = new MatchStatistics(reverseName);
			filter(matchForward,matchReverse);
		}
	}
	
	public String getFilename(int matchIndex, boolean forward){
		String names = (namesFlipped) ? (adversary + "."+ hero) : (hero + "."+ adversary);
		String matchString = "match"+ matchIndex + ((forward) ? "fwd.log" : "rev.log");
		return directory + competition + "." + names + "." + matchString;
	}
	public void filter(MatchStatistics matchForward, MatchStatistics matchReverse){
		for(int i=0;i<matchForward.getNumberOfHands();i++){
			HandStatistics forwardHand = matchForward.hands.get(i);
			HandStatistics reverseHand = matchReverse.hands.get(i);
			if (acceptHandPair(forwardHand,reverseHand)){
				outputHand(forwardHand);
				outputHand(reverseHand);
			}
		}
	}
	
	public void outputHand(HandStatistics hand){
		  System.err.println(hand.toPAString(hero,outputHand,heroStackBlinds,adversaryStackBlinds));
		  double heroStackChange = hand.getNetSmallBlinds(hero);
		  heroStackBlinds += heroStackChange/2;
		  adversaryStackBlinds -= heroStackChange/2;
		  outputHand++;
		
	}
	
	public boolean acceptHandPair(HandStatistics forward, HandStatistics reverse){
		double smallBlindsDifferenceThreshold=4.0;
		boolean turnSeenBothSides = ((forward.getLastRoundPlayed()>=2)&&(reverse.getLastRoundPlayed()>=2));
		double performanceDifference = forward.getNetSmallBlinds(hero) - reverse.getNetSmallBlinds(adversary);
		boolean closePlay = (performanceDifference<=smallBlindsDifferenceThreshold);
		return (!closePlay && turnSeenBothSides);
	}
	
	public static void showUsage(){
		System.err.println("Usage:java ca.ualberta.cs.poker.free.ViewingFilter <directory> <competition> <hero> <adversary> <flipped> <numMatches>");
	}
	
	public static void main(String[] args) throws FileNotFoundException, IOException{
		if (args.length!=6){
			showUsage();
			System.exit(0);
		}
		String directory = args[0];
		String competition = args[1];
		String hero = args[2];
		String adversary = args[3];
		boolean flipped = args[4].equalsIgnoreCase("true");
		int numMatches = Integer.parseInt(args[5]);
		ViewingFilter filter = new ViewingFilter(directory,competition,hero,adversary,flipped,numMatches);
		filter.filterAll();
	}
	
	
}
