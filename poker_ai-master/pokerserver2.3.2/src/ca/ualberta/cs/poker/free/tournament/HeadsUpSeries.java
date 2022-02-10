package ca.ualberta.cs.poker.free.tournament;

import java.net.InetAddress;
import java.util.Vector;

import ca.ualberta.cs.poker.free.dynamics.MatchType;

public class HeadsUpSeries extends AbstractSeries{	
  MatchType info;
  
  /**
   * The root of all match names for this series
   */
  String rootMatchName;

  /**
   * The root of all card file names for this series.
   */
  String rootCardFileName;

  /**
   * The number of duplicate pairs of matches.
   */
  int numDuplicatePairs;

  /**
   * The first bot
   */
  BotInterface playerA;

  /**
   * The second bot
   */
  BotInterface playerB;

  /**
   * The server IP.
   */
  InetAddress server;
  
  /**
   * Construct a new series, given the bots and the root names,
   * the number of duplicate pairs, and the server IP.
   */
  public HeadsUpSeries(BotInterface playerA, BotInterface playerB,
  String rootMatchName, String rootCardFileName, int
  numDuplicatePairs,InetAddress server,MatchType info){
    this.playerA = playerA;
    this.playerB = playerB;
    this.rootMatchName = rootMatchName;
    this.rootCardFileName = rootCardFileName;
    this.numDuplicatePairs = numDuplicatePairs;
    this.server = server;
    this.info = info;
  }

  /**
   * Get ALL matches that need to be run, are running,
   * or have been run. Return them in sets based upon
   * the cards used.
   */
  public Vector<Vector<MatchInterface> > getMatchSets(){
    Vector<BotInterface> forward = new Vector<BotInterface>();
    forward.add(playerA);
    forward.add(playerB);
    Vector<BotInterface> reverse = new Vector<BotInterface>();
    reverse.add(playerB);
    reverse.add(playerA);
    Vector<Vector<MatchInterface> > result = new Vector<Vector<MatchInterface> >();
    for(int i=0;i<numDuplicatePairs;i++){
      Vector<MatchInterface> smallResult = new Vector<MatchInterface>();
      HeadsUpMatch forwardMatch = new HeadsUpMatch(forward,
        rootCardFileName+i+".crd",server,rootMatchName+i+"fwd",info);

      HeadsUpMatch reverseMatch = new HeadsUpMatch(reverse,
        rootCardFileName+i+".crd",server,rootMatchName+i+"rev",info);
      smallResult.add(forwardMatch);
      smallResult.add(reverseMatch);
      result.add(smallResult);
    }
    return result;
  }

  /**
   * Get the bots for this series.
   */
  public Vector<BotInterface> getBots(){
    Vector<BotInterface> bots = new Vector<BotInterface>();
    bots.add(playerA);
    bots.add(playerB);
    return bots;
  }

  /**
   * Gets the utilities for this series.
   */
  /*
  public void showStatistics(){
    Vector<Vector<MatchInterface> > matches = getMatchSets();
    Vector<BotInterface> bots = getBots();
    int[] total = new int[bots.size()];
    int[] totalSq = new int[bots.size()];
    for(Vector<MatchInterface> duplicateSet:matches){
      int[] result = new int[bots.size()];
      for(MatchInterface match:duplicateSet){
        System.out.println("Match:"+match);
        Vector<Integer> utils = match.getUtilities();
	System.out.println("util[0]="+utils.get(0)+
	" util[1]="+utils.get(1));
        Vector<BotInterface> matchBots = match.getBots();
        for(int i=0;i<bots.size();i++){
          int botIndex = matchBots.indexOf(bots.get(i));
  	  result[i]+=utils.get(botIndex);
        }
      }
      for(int i=0;i<bots.size();i++){
        total[i]+=result[i];
	totalSq[i]+=(result[i]*result[i]);
      }
    }
    for(int i=0;i<bots.size();i++){
      int samples = matches.size();
      double avg = ((double)total[i])/samples;
      double avgOfSquares = ((double)totalSq[i])/samples;
      double biasedVarianceOfSingle = avgOfSquares - (avg*avg);
      double varianceOfSingle = (biasedVarianceOfSingle *
      samples)/(samples-1);
      double varianceOverall = varianceOfSingle/samples;
      double stddev = Math.sqrt(varianceOverall);
      System.out.println("bot:"+bots.get(i));
      System.out.println("mean:"+avg);
      System.out.println("stddev:"+stddev);
    }
    
  }
*/

}
