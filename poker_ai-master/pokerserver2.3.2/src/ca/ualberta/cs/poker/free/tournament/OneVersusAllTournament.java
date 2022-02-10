package ca.ualberta.cs.poker.free.tournament;

import java.net.InetAddress;
import java.security.SecureRandom;
import java.util.Vector;

import ca.ualberta.cs.poker.free.dynamics.MatchType;

public class OneVersusAllTournament implements Node {

	  /**
	   * Bots in the tournament.
	   */
	  Vector<BotInterface> bots;	  
	  
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
	  
	  boolean reversed;

	  /**
	   * Construct a tournament from a bot list,
	   * root names, number of duplicate match pairs per series,
	   * and server IP.
	   */
	  public OneVersusAllTournament(Vector<BotInterface> bots, String rootSeriesName, String rootCardFileName, int numDuplicatePairs, InetAddress server, MatchType info, boolean reversed) {
		super();
		// TODO Auto-generated constructor stub
		this.bots = bots;
		this.info = info;
		this.rootSeriesName = rootSeriesName;
		this.rootCardFileName = rootCardFileName;
		this.numDuplicatePairs = numDuplicatePairs;
		this.server = server;
		this.reversed = reversed;
		initHeadsUpLimitSeries();
	}
	  

	  /**
	   * Create the series objects.
	   */
	  public void initHeadsUpLimitSeries(){
	    series = new Vector<HeadsUpSeries>();

	    for(int i=1;i<bots.size();i++){
	        BotInterface playerA = bots.get(0);
			BotInterface playerB = bots.get(i);
			if ( ! reversed ) {
				String rootMatchName = rootSeriesName + "."+ playerA.getName()+"."+playerB.getName()+".match";
				series.add(new HeadsUpSeries(playerA,playerB,rootMatchName, rootCardFileName,numDuplicatePairs,server,info));
			} else {
				String rootMatchName = rootSeriesName + "."+ playerA.getName()+"."+playerB.getName()+".match";
				series.add(new HeadsUpSeries(playerB,playerA,rootMatchName, rootCardFileName,numDuplicatePairs,server,info));
			}
			
		}
	  }

	
	
	public boolean isComplete() {
	    for(HeadsUpSeries s:series){
	      if (!s.isComplete()){
	        return false;
	      }
	    }
	    return true;
	}

	public void load(Forge f) {
	    for(HeadsUpSeries s:series){
	        s.load(f);
	      }
	}

	public Vector<BotInterface> getWinners() {
		return bots;
	}

	public void generateCardFiles(SecureRandom random) {
	    for(HeadsUpSeries s:series){
	        s.generateCardFiles(random);
	      }
	}

	public boolean confirmCardFiles() {
		for(HeadsUpSeries s:series){
		      if (!s.confirmCardFiles()){
		    	  return false;
		      }
		    }
		    return true;
	}

	public void showStatistics() {
		for(HeadsUpSeries s:series){
	      s.showStatistics();
	    }
	}

}
