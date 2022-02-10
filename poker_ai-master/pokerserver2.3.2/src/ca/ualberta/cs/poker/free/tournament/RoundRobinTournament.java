package ca.ualberta.cs.poker.free.tournament;

import java.net.InetAddress;
import java.util.Vector;
import java.security.SecureRandom;

import ca.ualberta.cs.poker.free.dynamics.MatchType;

/**
 * Runs a round-robin tournament of either heads-up limit or
 * heads-up no-limit. 
 * At present, only limit
 * @author Martin Zinkevich
 *
 */
public class RoundRobinTournament implements Node{
  /**
   * Bots in the tournament.
   */
  Vector<BotInterface> bots;
  
  /**
   * If false, instant runoff is used throughout.
   * If true, instant runoff is used only until there
   * are four or less bots, and then bankroll is calculated.
   */
  WinnerDeterminationType bankrollPart;
  
  
  /**
   * The type of game
   */
  public MatchType info;
  
  /**
   * Series between bots in the tournament.
   */
  Vector<HeadsUpSeries> series;


  /**
   * Root of all series names.
   */
  String rootSeriesName;
  
  /**
   * Root card file name.
   */
  String rootCardFileName;

  /** 
   * The number of duplicate match pairs per series.
   */
  int numDuplicatePairs;

  /**
   * The server IP.
   */
  InetAddress server;
  
  /**
   * Construct a tournament from a bot list,
   * root names, number of duplicate match pairs per series,
   * and server IP.
   */
  public RoundRobinTournament(Vector<BotInterface> bots,
  String rootSeriesName, String rootCardFileName, int
  numDuplicatePairs,InetAddress server, MatchType info, WinnerDeterminationType bankrollPart){
    this.bots = bots;
    this.rootSeriesName = rootSeriesName;
    this.rootCardFileName = rootCardFileName;
    this.numDuplicatePairs = numDuplicatePairs;
    this.server = server;
    this.info = info;
    this.bankrollPart = bankrollPart;
    initHeadsUpLimitSeries();
  }

  /**
   * Create the series objects.
   */
  public void initHeadsUpLimitSeries(){
    series = new Vector<HeadsUpSeries>();

    for(int i=0;i<bots.size();i++){
      for(int j=i+1;j<bots.size();j++){
        BotInterface playerA = bots.get(i);
	BotInterface playerB = bots.get(j);
	String rootMatchName = rootSeriesName + "."+
	playerA.getName()+"."+playerB.getName()+".match";
	series.add(new HeadsUpSeries(playerA,playerB,rootMatchName,
	rootCardFileName,numDuplicatePairs,server,info));
      }
    }
  }

  public void showStatistics(){
	Vector<Vector<BotInterface> > rankings = getRankings();
	int nextRank = 1;
	for(Vector<BotInterface> levelSet:rankings){
		for(BotInterface bot:levelSet){
		  System.out.println("Rank "+nextRank+":"+bot.getName());
		}
		nextRank += levelSet.size();
	}
	
	for(HeadsUpSeries s:series){
      s.showStatistics();
    }
  }
  
  /**
   * Get the utilities from the round robin tournament.
   */
  public Vector<Integer> getUtilities(){
    int[] result = new int[bots.size()];
    for(HeadsUpSeries s:series){
      Vector<BotInterface> seriesBots = s.getBots();
      Vector<Integer> seriesUtils = s.getUtilities();
      for(int i=0;i<seriesBots.size();i++){
        int botIndex = bots.indexOf(seriesBots.get(i));
	result[botIndex]+=seriesUtils.get(i);
      }
    }
    Vector<Integer> finalResult = new Vector<Integer>();
    for(int i=0;i<result.length;i++){
      finalResult.add(result[i]);
    }
    return finalResult;
  }
   
  /**
   * Gets a matrix of the bankrolls 
   * Entry result[4][5] is (roughly) the amount won by 4 against 5.
   * @return
   */
  public int[][] getUtilityMatrix(){
	  int[][] result = new int[bots.size()][bots.size()];
	  for(HeadsUpSeries s:series){
	      Vector<BotInterface> seriesBots = s.getBots();
	      if (seriesBots.size()!=2){
	    	  return null;
	      }
	      Vector<Integer> seriesUtils = s.getUtilities();
	      int botIndex1 = bots.indexOf(seriesBots.get(0));
	      int botIndex2 = bots.indexOf(seriesBots.get(1));
	      int botStack1 = seriesUtils.get(0);
	      int botStack2 = seriesUtils.get(1);
	      result[botIndex1][botIndex2]=botStack1;
	      result[botIndex2][botIndex1]=botStack2;
	    }
	      
	  return result;
  }
  

  public static int sign(int n){
	  if (n>0){
		  return 1;
	  }
	  if (n<0){
		  return -1;
	  }
	  return 0;
  }
  
  /**
   * Gets a matrix of the bankrolls 
   * Entry result[4][5] is:
   * +1 if 4 beat 5 in bankroll in a series
   * -1 if 5 beat 4 in bankroll in a series
   * 0 if they had an equal bankroll
   * @return
   */
  public int[][] getSeriesMatrix(){
	  int[][] result = new int[bots.size()][bots.size()];
	  for(HeadsUpSeries s:series){
	      Vector<BotInterface> seriesBots = s.getBots();
	      if (seriesBots.size()!=2){
	    	  return null;
	      }
	      Vector<Integer> seriesUtils = s.getUtilities();
	      int botIndex1 = bots.indexOf(seriesBots.get(0));
	      int botIndex2 = bots.indexOf(seriesBots.get(1));
	      int botStack1 = seriesUtils.get(0);
	      int botStack2 = seriesUtils.get(1);
	      result[botIndex1][botIndex2]=sign(botStack1-botStack2);
	      result[botIndex2][botIndex1]=sign(botStack2-botStack1);
	  }
	  return result;
  }
  
  public boolean isComplete(){
    //System.out.println("roundRobinTournament.isComplete()");
    for(HeadsUpSeries s:series){
      if (!s.isComplete()){
        //System.out.println("roundRobinTournament.isComplete() finished");
        return false;
      }
    }
    //System.out.println("roundRobinTournament.isComplete() finished");
    return true;
  }

  public void load(Forge f){
    for(HeadsUpSeries s:series){
      s.load(f);
    }
  }
  
  public Vector<BotInterface> getWinners(){
	  
	  return getRankings().get(0);
  }
  
  public Vector<Vector<BotInterface> > getRankings(){
	  switch(bankrollPart){
	  case INSTANTRUNOFFBANKROLL:
	  default:
		  InstantRunoffRule<BotInterface> rule = new InstantRunoffRule<BotInterface>(bots,getUtilityMatrix());
	  	  return rule.getRankings();
	  case TRUNCATEDBANKROLL:
		  InstantRunoffRule<BotInterface> rule2 = new InstantRunoffRule<BotInterface>(bots,getUtilityMatrix(),4);
		  return rule2.getRankings();
	  case INSTANTRUNOFFSERIES:
		  InstantRunoffRule<BotInterface> rule3 = new InstantRunoffRule<BotInterface>(bots,getSeriesMatrix());
		  return rule3.getRankings();
	  }
	  
  }	  
  
  public void generateCardFiles(SecureRandom random){
    for(HeadsUpSeries s:series){
      s.generateCardFiles(random);
    }

  }

  public boolean confirmCardFiles(){
	    for(HeadsUpSeries s:series){
	      if (!s.confirmCardFiles()){
	    	  return false;
	      }
	    }
	    return true;
	  }
  /**
   * Show the matches and the bots in the tournament.
   */
  public String toString(){
    String result = "Matches:\n";
    for(HeadsUpSeries s:series){
      result+=s.toString();
    }
    result+="Bots:\n";
    for(BotInterface b:bots){
      result+=(b.toString()+"\n");
    }
    return result;
  }

}
