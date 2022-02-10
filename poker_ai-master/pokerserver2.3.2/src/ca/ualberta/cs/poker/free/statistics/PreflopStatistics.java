package ca.ualberta.cs.poker.free.statistics;

import java.io.File;
import java.io.IOException;
import java.util.Vector;


import ca.ualberta.cs.poker.free.dynamics.Card;

/**
 * Crawls over a directory to evaluate a particular player's actions.
 * @author maz
 *
 */
public class PreflopStatistics {
	int[] occurrences;
	int[] performance;
	
	String playerToAnalyze;
	
	public PreflopStatistics(String playerToAnalyze){
		this.playerToAnalyze = playerToAnalyze;
		occurrences = new int[169];
		performance = new int[169];
	}
	
	public static void main(String[] args) throws IOException{
		analyze(args[0],args[1],args[2]);

	}

	public static void analyze(String file, String player, String opponent) throws IOException{
		PreflopStatistics ps = new PreflopStatistics(player);
		
		ps.analyze(file,opponent);
		System.out.println(ps);
	}
	public void analyze(String file,String opponent) throws IOException{
		File f = new File(file);
	if (!(f.exists())){
		System.err.println("File not found:"+file);
	} else if (f.isDirectory()){
		System.err.println("Descending into directory "+file);
		String[] files = f.list();
		if (!file.endsWith(File.separator)){
			file+=File.separator;
		}
		
		for(String subFile:files){
			analyze(file+subFile,opponent);
		}
	} else if (file.endsWith(".res")){
		//System.err.println("File "+file+" passed over.");
	} else {
	  //System.err.println("Loading match "+file+"...");
	  MatchStatistics m = new MatchStatistics(file);
	  //System.err.println("Loaded match:"+file);
	  Vector<String> names = m.hands.firstElement().names;
	  if (names.contains(playerToAnalyze)&&names.contains(opponent)){
		  analyze(m);
	  }
	}
	}
	
	
	
	public void analyze(MatchStatistics match) {
		for(HandStatistics hand:match.hands){
			analyze(hand);
		}
		
	}
	
	public void analyze(HandStatistics hand){
		int ourSeat = playerToAnalyze.equals(hand.names.get(0)) ? 0 : 1;
		int ourCards = cardPairToInt(hand.getHoleCards(ourSeat));
		int amountWon = hand.smallBlinds.get(ourSeat);
		occurrences[ourCards]++;
		performance[ourCards]+=amountWon;
	}
	
	public static String intToString(int n){
		int firstNumber = n / 13;
		int secondNumber = n % 13;
		if (firstNumber==secondNumber){
			return "pair of "+Card.Rank.toRank(firstNumber)+"s";
		} else if (firstNumber <= secondNumber){
			return ""+Card.Rank.toRank(secondNumber)+"-"+Card.Rank.toRank(firstNumber)+" offsuit";
		} else {
			return ""+Card.Rank.toRank(firstNumber)+"-"+Card.Rank.toRank(secondNumber)+" suited";
		}
	}
	
	public static int cardPairToInt(Card[] cards){
		assert(cards!=null);
		assert(cards.length==2);
		int firstIndex = cards[0].rank.index;
		int secondIndex = cards[1].rank.index;
		int maxIndex = Math.max(firstIndex,secondIndex);
		int minIndex = Math.min(firstIndex, secondIndex);
		if (cards[0].suit==cards[1].suit){
			return maxIndex * 13 + minIndex;
		} else {
			return minIndex * 13 + maxIndex;
		}
	}
	
	public String toString(){
		String result = playerToAnalyze+"\n";
		
		for(int i=0;i<occurrences.length;i++){
			result+="Observations of "+intToString(i)+":"+occurrences[i]+" Value:"+performance[i]+"\n";
		}
		return result;
	}
	
}
