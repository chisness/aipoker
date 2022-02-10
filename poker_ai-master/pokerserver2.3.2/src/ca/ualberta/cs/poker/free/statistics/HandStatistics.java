package ca.ualberta.cs.poker.free.statistics;

import java.io.IOException;
import java.util.Hashtable;
import java.util.StringTokenizer;
import java.util.Vector;

import ca.ualberta.cs.poker.free.dynamics.Card;

public class HandStatistics {
    /** The betting sequence (all rbf, no blind info) */
	public String bettingSequence;
	
	/** The net won (in small blinds) in seat order */
	public Vector<Integer> smallBlinds;
	/** The names of the bots in seat order */
	public Vector<String> names;
	
	/** The cards in the format rsrs|rsrs/rsrsrs/rs/rs 
	 * with the meaning noButtonPrivate|buttonPrivate/flop/turn/river
	 * */
	public String cards;
	/** The hand number */
	int handNumber;
	
	
	
	public HandStatistics(String bettingSequence, Vector<Integer> smallBlinds, 
			Vector<String> names, String cards, int handNumber) {
		this.bettingSequence = bettingSequence;
		this.smallBlinds = smallBlinds;
		this.names = names;
		this.cards = cards;
		this.handNumber = handNumber;
	}


	/**
	 * Test if this hand is a possible duplicate of another hand.
	 * True iff the cards are the same and the players are permuted.
	 * @param other the hand to be compared.
	 * @return if it is possible that this hand is a duplicate of other 
	 */
	public boolean isDuplicate(HandStatistics other){
		if (!other.names.containsAll(names)){
			return false;
		}
		if (!names.containsAll(other.names)){
			return false;
		}
		return cards.startsWith(other.cards)||other.cards.startsWith(cards);
	}
	
	/**
	 * Useful only for unbounded stack games.
	 * @return
	 */
	public String getRawCardsBuffered(){
		String result = remove(cards,'/');
		result = remove(result,'|');
		//System.err.println(result);
		return bufferCards(result,5+(names.size()*2));
	}

	public String getRawActions(){
		return remove(bettingSequence,'/');
	}
	/**
	 * Buffers observed cards with cards from the deck.
	 * @param original the original string of cards
	 * @param numCards the total number of cards at the end.
	 * @return a string of cards (no spaces)
	 */
	public static String bufferCards(String original, int numCards){
		Card[] array=Card.toCardArray(original);
		Vector<Card> result = new Vector<Card>();
		for(Card c:array){
			result.add(c);
			//System.err.println("Card:"+c);
		}
		//System.err.println("result.size()=="+result.size());
		if (result.size()<numCards){
			Card[] other=Card.getAllCards();
			for(Card c:other){
				if (!result.contains(c)){
					result.add(c);
					if (result.size()==numCards){
						String strResult="";
						for(Card c2:result){
							strResult += c2;
						}
						return strResult;
					}
				}
			}
		}
		return original;
		//throw new RuntimeException("Internal error in bufferCards: original="+original+", numCards="+numCards);
	}
	
	
	public static HandStatistics getUofAHandStatistics(String line){
		Vector<String> splitLine = split(line,':');
		int handNumber = Integer.parseInt(splitLine.get(0));
		Vector<String> names = split(splitLine.get(1),',');
		String weirdBetting = splitLine.get(2);
		weirdBetting=weirdBetting.replace('k', 'c');
		weirdBetting=weirdBetting.replace('b', 'r');
		weirdBetting=remove(weirdBetting,'0');
		String weirdCards = splitLine.get(3);
		weirdCards = remove(weirdCards,'|');
		weirdCards = weirdCards.replace(',','|');
		// Five dollars is a small blind
		Vector<String> dollars = split(splitLine.get(4),',');
		Vector<Integer> smallBlinds = new Vector<Integer>();
		for(String dollar:dollars){
			smallBlinds.add(Integer.parseInt(dollar)/5);
		}
		return new HandStatistics(weirdBetting,smallBlinds, 
				names, weirdCards, handNumber);
	}
	
	/**
	 * Gets small blinds in player order
	 * @param line
	 * @return
	 */
	public static Vector<Integer> getGameStateSmallBlinds(String line){
		Vector<String> splitLine = split(line,':');
		Vector<Integer> result = new Vector<Integer>();
		for(int i=0;i<2;i++){
			double dollars = Double.parseDouble(splitLine.get(i+5));
			int smallBlinds = (int)(dollars/5.0);
			result.add(smallBlinds);
		}
		return result;
	}
	/**
	 * 
	 * @param line a GAMESTATE entry as output by the 2006 AAAI server
	 * @param names names in game order
	 * @return A HandStatistics object representing the hand
	 */
	public static HandStatistics getGameStateHandStatistics(String line,Vector<String> names,Vector<Integer> previousSmallBlinds){
		//throw new RuntimeException("Not implemented");
		Vector<String> splitLine = split(line,':');
		int handNumber = Integer.parseInt(splitLine.get(2));
		String bettingSequence = splitLine.get(3);
		String cards = splitLine.get(4);
		boolean flipped = (handNumber % 2)==1;
		Vector<Integer> currentSmallBlinds = getGameStateSmallBlinds(line);
		Vector<Integer> bankrollChange = new Vector<Integer>();
		Vector<String> seatNames = new Vector<String>();
		if (flipped){
			bankrollChange.add(currentSmallBlinds.get(1)-previousSmallBlinds.get(1));
			bankrollChange.add(currentSmallBlinds.get(0)-previousSmallBlinds.get(0));
			seatNames.add(names.get(1));
			seatNames.add(names.get(0));
		} else {
			bankrollChange.add(currentSmallBlinds.get(0)-previousSmallBlinds.get(0));
			bankrollChange.add(currentSmallBlinds.get(1)-previousSmallBlinds.get(1));
			seatNames.add(names.get(0));
			seatNames.add(names.get(1));
		}
		
		return new HandStatistics(bettingSequence,bankrollChange, 
				seatNames, cards, handNumber);
	}
	
	public static HandStatistics getGameStateVersion2HandStatistics(String line){
		Vector<String> splitLine = split(line,':');
		Vector<String> names = split(splitLine.get(0),'|');
		int handNumber = Integer.parseInt(splitLine.get(1));
		String bettingSequence = splitLine.get(2);
		String cards = splitLine.get(3);
		Vector<String> smallBlindStrings = split(splitLine.get(4),'|');
		Vector<Integer> smallBlinds = new Vector<Integer>();
		for(String str:smallBlindStrings){
			smallBlinds.add(Integer.parseInt(str));
		}
		return new HandStatistics(bettingSequence,smallBlinds, 
				names, cards, handNumber);
	}
	
	/**
	 * Splits a string based upon a character.
	 * @see ca.ualberta.cs.poker.free.server.TimedSocket#parseByColons(String)
	 * TODO put this in a standard place (perhaps a util package?)
	 * @param str The string to split
	 * @param splitter a character upon which to split
	 * @return a vector of split strings, some possibly empty.
	 */
    public static Vector<String> split(String str, char splitter){
      Vector<String> result = new Vector<String>();
      int lastIndex=-1;
      while(true){
        int currentIndex = 0;
	currentIndex = str.indexOf(splitter,lastIndex+1);
	if (currentIndex==-1){
		result.add(str.substring(lastIndex+1));
	  return result;
	}
	result.add(str.substring(lastIndex+1,currentIndex));
	lastIndex=currentIndex;
      }
    }
    
    public static String remove(String str, char removed){
    	Vector<String> pieces = split(str,removed);
    	String result = "";
    	for(String piece:pieces){
    		result += piece;
    	}
    	return result;
    }
    
    public String toString(){
    	String result = "";
    	for(int i=0;i<names.size()-1;i++){
    		result+=names.get(i)+"|";
    	}
    	result+=names.lastElement();
    	result += (":" + handNumber + ":"+bettingSequence +":"+cards);
    	result += ":"+smallBlinds.firstElement();
    	for(int i=1;i<smallBlinds.size();i++){
    		result+=("|"+smallBlinds.get(i));
    	}
    	return result;
    }
    
    public String toUofAString(){
    	String result = "";
    	result += handNumber;
    	result += ":";
    	for(int i=0;i<names.size()-1;i++){
    		result+=names.get(i)+",";
    	}
    	result+=names.lastElement();
    	result+=":0";
    	result+=bettingSequence+":";
    	String holeCards = cards.substring(0, 9);
    	holeCards = holeCards.replace('|',',');
    	String afterHole = cards.substring(9);
    	result += holeCards + "|"+afterHole;
    	result += ":";
    	for(int i=0;i<smallBlinds.size()-1;i++){
    		result += (smallBlinds.get(i)* 5);
    		result += ",";
    	}
    	result+=(smallBlinds.lastElement()*5);
    	return result;
    }
    
    public double getNetSmallBlinds(String name){
    	return smallBlinds.get(names.indexOf(name));
    }
    
    public int getLastRoundPlayed(){
    	Vector<String> bettingPerRound  = HandStatistics.split(bettingSequence, '/');
    	return bettingPerRound.size()-1;
    }
    
    public String toPAString(String hero, int handIndex,
			double heroStackBlinds, double adversaryStackBlinds) {
		assert (names.size() == 2);
		int heroIndex = names.indexOf(hero);
		assert (heroIndex != -1);
		int adversaryIndex = 1 - heroIndex;

		String heroCards = Card.arrayToString(getHoleCards(heroIndex));
		String adversaryCards = Card
				.arrayToString(getHoleCards(adversaryIndex));

		String game = "Texas Hold'em";

		String heroName = hero;
		String adversaryName = names.get(adversaryIndex);

		double heroSmallBets = ((double) smallBlinds.get(heroIndex) / 2.0);
		double adversarySmallBets = ((double) smallBlinds.get(adversaryIndex) / 2.0);

		int button = (1 - heroIndex) * 5;

		int numPlayers = 2;

		String oldBoard = cards.substring(9);
		String paBoard = "";
		for (int i = 0; i < oldBoard.length(); i++) {
			if (oldBoard.charAt(i) != '/') {
				paBoard += oldBoard.charAt(i);
			}
		}
		for (int i = paBoard.length(); i < 10; i++) {
			paBoard += "?";
		}

		double ante = 0;

		double smallBlind = 5.0;

		String paBetting = "sB";
		String bets = "";
		String lastAction = "";
		int maxInPot = 2;
		for (int i = 0; i < bettingSequence.length(); i++) {
			char currentChar = bettingSequence.charAt(i);
			if (Character.isDigit(currentChar)) {
				lastAction += currentChar;
			} else {
				if (lastAction.length() > 0) {
					char actionChar = lastAction.charAt(0);
					switch (actionChar) {
					case 'b':
						break;
					case 'r':
						if (lastAction.length() > 1) {
							int finalAmount = Integer.parseInt(lastAction
									.substring(1));
							int raiseAmount = finalAmount - maxInPot;
							maxInPot = finalAmount;
							if (bets.length() == 0) {
								bets = ";BETS="
										+ (smallBlind * (double) raiseAmount);
							} else {
								bets += ","
										+ (smallBlind * (double) raiseAmount);
							}
						}
						// Continue on to add the action to the betting
					case 'f':
					case '/':
					case 'c':
					default:
						paBetting += actionChar;
						break;

					}
				}
				lastAction = "" + currentChar;
			}
		}
		if (lastAction.length() > 0) {
			char actionChar = lastAction.charAt(0);
			switch (actionChar) {
			case 'b':
				break;
			case 'r':
				if (lastAction.length() > 1) {
					int finalAmount = Integer.parseInt(lastAction.substring(1));
					int raiseAmount = finalAmount - maxInPot;
					maxInPot = finalAmount;
					if (bets.length() == 0) {
						bets = ";BETS=" + (smallBlind * (double) raiseAmount);
					} else {
						bets += "," + (smallBlind * (double) raiseAmount);
					}
				}
				// Continue on to add the action to the betting
			case 'f':
			case '/':
			case 'c':
			default:
				paBetting += actionChar;
				break;

			}
		}
    	
    	
    	
    	
    	
    	
    	String result = 
    		"PC0="+heroCards+
    		";BB=10;HERO="+heroName+
    		";GAME="+game+
    		";PV5="+adversarySmallBets+
    		";ID="+handIndex+
    		";TABLE=2007 AAAI Computer Poker Competition"+
    		";BTN="+button+
    		";PV0="+heroSmallBets+
    		";PB5="+adversaryStackBlinds/2.0+
    		";SBS="+button+
    		";TIME="+handIndex+
    		";PN5="+adversaryName+
    		";SEQ="+paBetting+
    		";PB0="+heroStackBlinds/2.0+
    		";PN0="+heroName+
    		bets+
    		";NP="+numPlayers+
    		";SB="+smallBlind+
    		";SITE=Poker Academy Pro"+
    		";BOARD="+paBoard+
    		";PC5="+adversaryCards+
    		";ANTE="+ante+";";
    	return result;
    }
    
    public Card[] getHoleCards(int seat){
    	int holeCardBeginIndex = seat * 5;
    	int holeCardEndIndex = holeCardBeginIndex + 4;
    	String holeCardString = cards.substring(holeCardBeginIndex,holeCardEndIndex);
    	return Card.toCardArray(holeCardString);
    }

    /**
     * At present this function is designed for two player limit with unlimited bets.
     * @param line
     * @return
     * @throws IOException 
     */
	public static HandStatistics parsePokerAcademy(String line) throws IOException {
		// TODO Auto-generated method stub
		StringTokenizer st = new StringTokenizer(line,";");
		Hashtable<String,String> handParts=new Hashtable<String,String>();
		//HandStatistics hand = new HandStatistics();
		Vector<Integer> playerNumbers = new Vector<Integer>();
		while(st.hasMoreTokens()){
			String token = st.nextToken();
			int esPos = token.indexOf("=");
			String key = token.substring(0, esPos);
			String entry = token.substring(esPos+1);
			handParts.put(key,entry);
			if (key.startsWith("PN")){
				playerNumbers.add(Integer.parseInt(key.substring(2)));
			}
		}
		if (playerNumbers.size()!=2){
			throw new IOException("Found "+playerNumbers.size()+" players in line:"+line);
		}
		if (!handParts.containsKey("BTN")){
			throw new IOException("Could not find BTN in line:"+line);
		}
		int PAButton = Integer.parseInt(handParts.get("BTN"));
		int vectorButton = -1;
		for(int i=0;i<playerNumbers.size();i++){
			if (playerNumbers.get(i)==PAButton){
				vectorButton = i;
			}
		}
		if (vectorButton==-1){
			throw new IOException("Unknown button for line:"+line);
		}
		if (vectorButton==0){
			Vector<Integer> newPlayerNumbers = new Vector<Integer>();
			newPlayerNumbers.add(playerNumbers.get(1));
			newPlayerNumbers.add(playerNumbers.get(0));
			playerNumbers = newPlayerNumbers;
		}
		
		
		//double smallBlindsValue = Double.parseDouble(handParts.get("SB"));
		
		Vector<String> names = new Vector<String>();
		Vector<String> holeCards = new Vector<String>();
		Vector<Integer> values = new Vector<Integer>(); 
		
		for(int i:playerNumbers){
			names.add(handParts.get("PN"+i));
			holeCards.add(handParts.get("PC"+i));
			double moneyValue = Double.parseDouble(handParts.get("PV"+i));
			values.add((int)(moneyValue*2));
		}
		
		if (!handParts.containsKey("SEQ")){
			throw new IOException("Could not find SEQ in line:"+line);
		}
		String PAbettingSequence = handParts.get("SEQ");
		String bettingSequence = PAbettingSequence.substring(2).replace('k','c').replace('b', 'r');
		
		String PAboard = handParts.get("BOARD");
		int questIndex = PAboard.indexOf('?');
		String board = PAboard;
		if (questIndex>=0){
			board = PAboard.substring(0,questIndex);
		}
		if (board.length()>6){
			board = board.substring(0,6)+"/"+board.substring(6);
		}
		if (board.length()>9){
			board = board.substring(0,9)+"/"+board.substring(9);
		}
		if (board.length()>12){
			board = board.substring(0,12)+"/"+board.substring(12);
		}
		if (board.length()>0){
			board = "/"+board;
		}
		
		String cardSequence = holeCards.get(0)+"|"+holeCards.get(1)+board;
		
		int handNumber = Integer.parseInt(handParts.get("ID"));
		
		return new HandStatistics(bettingSequence, values, names, cardSequence, handNumber);
		
	}	
}
