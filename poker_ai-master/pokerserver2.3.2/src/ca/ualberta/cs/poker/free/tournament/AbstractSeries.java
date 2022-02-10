package ca.ualberta.cs.poker.free.tournament;

import java.util.Vector;
import java.security.SecureRandom;

public abstract class AbstractSeries implements Node{

  /**
   * Get ALL matches that need to be run, are running,
   * or have been run. Return them in sets based upon
   * the cards used.
   */
  public abstract Vector<Vector<MatchInterface> > getMatchSets();

  /**
   * Gets the bots.
   */
  public abstract Vector<BotInterface> getBots();
  
  /**
   * Get ALL matches that need to be run, are running,
   * or have been run. The default is to flatten the match sets.
   */
  public Vector<MatchInterface> getMatches(){
    Vector<Vector<MatchInterface> > matches = getMatchSets();
    
    
    Vector<MatchInterface> result = new Vector<MatchInterface>();
    for(Vector<MatchInterface> set:matches){
      result.addAll(set);
    }
    return result;
  }

  /**
   * Gets the utilities for this series.
   */
  public void showStatistics(){
    Vector<Vector<MatchInterface> > matches = getMatchSets();
    Vector<BotInterface> bots = getBots();
    int[] total = new int[bots.size()];
    int[] totalSq = new int[bots.size()];
    for(Vector<MatchInterface> duplicateSet:matches){
      int[] result = new int[bots.size()];
      for(MatchInterface match:duplicateSet){
        Vector<Integer> utils = match.getUtilities();
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
      double stddev = 0;
      if (samples>1){
    	  double varianceOfSingle = (biasedVarianceOfSingle * samples)/(samples-1);
      double varianceOverall = varianceOfSingle/samples;
      stddev = Math.sqrt(varianceOverall);
      }
      System.out.println("bot:"+bots.get(i));
      System.out.println("mean:"+avg);
      if (samples>1){
    	  System.out.println("stddev:"+stddev);
      }
    }
    
  }

  /**
   * Gets the utilities for this series.
   */
  public Vector<Integer> getUtilities(){
    Vector<MatchInterface> matches = getMatches();
    Vector<BotInterface> bots = getBots();
    int[] result = new int[bots.size()];
    for(MatchInterface match:matches){
      Vector<Integer> utils = match.getUtilities();
      Vector<BotInterface> matchBots = match.getBots();
      for(int i=0;i<bots.size();i++){
        int botIndex = matchBots.indexOf(bots.get(i));
	result[i]+=utils.get(botIndex);
      }
    }
    Vector<Integer> finalResult = new Vector<Integer>();
    for(int i=0;i<result.length;i++){
      finalResult.add(result[i]);
    }
    return finalResult;
  }

  public boolean isComplete(){
    //System.out.println("HeadsUpLimitSeries.isComplete()");
    Vector<MatchInterface> matches = getMatches();
    for(MatchInterface match:matches){
      if (!match.isComplete()){
        //System.out.println("HeadsUpLimitSeries.isComplete() finished");
        return false;
      }
    }
    //System.out.println("HeadsUpLimitSeries.isComplete() finished");
    return true;
  }

  public void load(Forge f) {
		Vector<MatchInterface> matches = getMatches();
		for (MatchInterface match : matches) {
			if (!match.isComplete()) {
				if (!f.runningOrQueued(match)) {
					f.add(match);
				}
			}
		}
	}

  public void generateCardFiles(SecureRandom random){
    Vector<MatchInterface> matches = getMatches();
    for(MatchInterface m:matches){
      m.generateCardFile(random);
    }
  }

  public boolean confirmCardFiles(){
	    Vector<MatchInterface> matches = getMatches();
	    for(MatchInterface m:matches){
	      if (!m.confirmCardFile()){
	    	  return false;
	      }
	    }
	    return true;
	  }

  public Vector<BotInterface> getWinners(){
    Vector<Integer> utils = getUtilities();
    Vector<BotInterface> bots = getBots();
    int maxSoFar = utils.get(0);
    Vector<BotInterface> botsSoFar = new Vector<BotInterface>();
    botsSoFar.add(bots.get(0));
    for(int i=1;i<utils.size();i++){
      if (maxSoFar<utils.get(i)){
        botsSoFar = new Vector<BotInterface>();
        botsSoFar.add(bots.get(i));
	maxSoFar = utils.get(i);
      } else if (maxSoFar==utils.get(i)){
    	  botsSoFar.add(bots.get(i));
      }
    }
    return botsSoFar;
  }


  /**
   * A string representing the series.
   */
  public String toString(){
    String result = "";
    Vector<MatchInterface> matches = getMatches();
    for(MatchInterface m:matches){
      result += m.toString() + "\n";
    }
    return result;
  }

}
