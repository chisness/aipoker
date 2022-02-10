package ca.ualberta.cs.poker.free.statistics;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.PipedReader;
import java.io.PipedWriter;
import java.util.Vector;


import ca.ualberta.cs.poker.free.dynamics.LimitType;
import ca.ualberta.cs.poker.free.dynamics.MatchType;
import ca.ualberta.cs.poker.free.dynamics.RingDynamics;

/**
 * Crawls over a directory to evaluate a particular player's actions.
 * @author maz
 *
 */
public class PlayerStatistics {
	int[][] raiseCounts;
	int[] limitRaiseCounts;
	int[] foldCounts;
	int[] callCounts;
	int[] positionCounts;
	int[][] raiseValues;
	int[] limitRaiseValues;
	int[] foldValues;
	int[] callValues;
	int[] positionValues;
	
	String playerToAnalyze;
	
	public PlayerStatistics(String playerToAnalyze){
		this.playerToAnalyze = playerToAnalyze;
	}
	
	public static void main(String[] args) throws IOException{
		analyze(args[0],args[1],args[2]);

	}

	public static void analyze(String file, String player, String opponent) throws IOException{
		PlayerStatistics ps = new PlayerStatistics(player);
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
	
	
	
	public boolean analyze(MatchStatistics match) {
		String[] names = match.hands.firstElement().names
				.toArray(new String[2]);
		MatchType info = new MatchType(match.limitType, match.stackBound,
				match.initialStack, match.hands.size());
		RingDynamics game = new RingDynamics(names.length,// numPlayers,
				info, names// botNames
		);
		if (positionCounts==null){
			initCounts(info);
		}
		return analyze(match, game);
	}
	
	public int getMaxBetAmount(MatchType info){
		switch(info.limitGame){
		case DOYLE:
			return info.doyleLimit;
		case LIMIT:
			return (info.smallBetSize + info.bigBetSize) * 8;
		case POTLIMIT:
		case NOLIMIT:
		default:
			return info.initialStackSize;
		}
	}
	
	
	public void initCounts(MatchType info){
		int maxBetAmount = getMaxBetAmount(info);
		if (info.limitGame==LimitType.LIMIT){
			limitRaiseCounts=new int[maxBetAmount+1];
			limitRaiseValues=new int[maxBetAmount+1];
			raiseCounts = null;
		} else {
			raiseCounts=new int[maxBetAmount+1][maxBetAmount+1];
			raiseValues=new int[maxBetAmount+1][maxBetAmount+1];
			limitRaiseCounts = null;
		}
		foldCounts=new int[maxBetAmount+1];
		callCounts=new int[maxBetAmount+1];
		positionCounts = new int[maxBetAmount+1];
		
		foldValues=new int[maxBetAmount+1];
		callValues=new int[maxBetAmount+1];
		positionValues = new int[maxBetAmount+1];

	}
	public void observeAction(RingDynamics game,HandStatistics hand, String action){
		if (hand.names.get(game.seatToAct).equals(playerToAnalyze)){
			int initialMaxPot = game.getMaxInPot();
			int value = hand.smallBlinds.get(game.seatToAct);
			switch(action.charAt(0)){
			case 'c':
				callCounts[initialMaxPot]++;
				callValues[initialMaxPot]+=value;
				break;
			case 'f':
				foldCounts[initialMaxPot]++;
				foldValues[initialMaxPot]+=value;
				break;
			case 'r':
			default:
				if (game.info.limitGame==LimitType.LIMIT){
					limitRaiseCounts[initialMaxPot]++;
					limitRaiseValues[initialMaxPot]+=value;
				} else {
					int raiseTo = Integer.parseInt(action.substring(1));
					raiseCounts[initialMaxPot][raiseTo]++;
					raiseValues[initialMaxPot][raiseTo]+=value;
				}
				break;
			}
			positionCounts[initialMaxPot]++;
			positionValues[initialMaxPot]+=value;
		}
	}
	
	public String toString(){
		String result = playerToAnalyze+"\n";
		if (callCounts==null){
			return result + "No games found.\n";
		}
		//System.err.println("toString:A");
		for(int i=0;i<callCounts.length;i++){
			//System.err.println("toString:B");			
			if (positionCounts[i]!=0){
				//System.err.println("toString:C");
				result+="Observations of "+i+":"+positionCounts[i]+" Value:"+positionValues[i]+"\n";
				result+="Calls from "+i+":"+callCounts[i]+" Value:"+callValues[i]+"\n";
				result+="Folds from "+i+":"+foldCounts[i]+" Value:"+foldValues[i]+"\n";
				if (limitRaiseCounts!=null){
					result+="Raises from "+i+":"+limitRaiseCounts[i]+" Value:"+limitRaiseValues[i]+"\n";
				} else {
					for(int j=0;j<raiseCounts[i].length;j++){
						if (raiseCounts[i][j]>0){
							result+="Raises from "+i+" to "+j+":"+raiseCounts[i][j]+" Value:"+raiseValues[i][j]+"\n";
						}
					}
			}
			}
		}
		return result;
	}
	
	public boolean analyze(MatchStatistics match, RingDynamics game) {
		// Pipe p = new Pipe();
		try {
			PipedWriter pw = new PipedWriter();
			PipedReader pr = new PipedReader(pw);
			BufferedReader br = new BufferedReader(pr);
			for (HandStatistics hand : match.hands) {
				// If we ever want no-limit ring games, we have to change this
				// here.
				//System.err.println("hand.getRawCardsBuffered()="+hand.getRawCardsBuffered());
				pw.write(hand.getRawCardsBuffered() + "\n");
				game.nextHand(br);
				String rawBetting = hand.getRawActions();
				String currentAction = "";
				//System.err.println("rawBetting:"+rawBetting);
				String verboseBetting = "";
				for (int i = 0; i < rawBetting.length(); i++) {
					char lastLetter = rawBetting.charAt(i);
					if (Character.isDigit(lastLetter)) {
						currentAction += lastLetter;
					} else {
						if (currentAction.length() > 0) {
                            //System.err.println(game.toString());
							//System.err.println("currentAction:"+currentAction+" seat: " + game.seatToAct + " round: "+game.roundIndex+" bettingSoFar: "+rawBetting.substring(0,i));
							int oldSeat = game.seatToAct;
							if (!currentAction.startsWith("b")){
								observeAction(game,hand,currentAction);
								game.handleAction(currentAction);
								
							}
							verboseBetting += currentAction + oldSeat+","+game.inPot[oldSeat];
							if (game.isGameOver()){
								verboseBetting += "*";
							} else if (game.firstActionOnRound){
								verboseBetting +="/";
							}
							//System.err.println("State:"+verboseBetting);
						}
						currentAction = "" + lastLetter;
					}
				}
				//System.err.println("currentAction:"+currentAction);
				if (!currentAction.startsWith("b")){
					observeAction(game,hand,currentAction);
					game.handleAction(currentAction);
					
				}
				verboseBetting += currentAction + game.seatToAct+","+game.inPot[game.seatToAct];
				if (game.isGameOver()){
					verboseBetting += "*";
				} else if (game.firstActionOnRound){
					verboseBetting +="/";
				}
				//System.err.println("State:"+verboseBetting);

				if (!game.isGameOver()) {
					System.err
							.println("Error: actions ended prematurely in hand:"
									+ hand);
					return false;
				} else if (game.amountWon == null) {
					System.err.println("Error 2 in hand:" + hand);
					return false;
				}
				for (int i = 0; i < hand.smallBlinds.size(); i++) {
					if (hand.smallBlinds.get(i) != game.amountWon[i]) {
						System.err.println("Error in the amount won in hand:"
								+ hand);
						System.err.println("Should have been "
								+ game.getGlobalState());
						return false;
					}

				}

			}
			return true;
		} catch (IOException io) {
			return false;
		}

	}
}
