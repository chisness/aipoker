package ca.ualberta.cs.poker.free.tournament;

import java.util.Hashtable;
import java.util.Vector;

/**
 * This class implements both a traditional instant runoff rule as well
 * as a bankroll/runoff rule.
 * 
 * Note that it is also possible to run this over series W/L/T or 
 * bankroll.
 * @author maz
 *
 * @param <A>
 */
public class InstantRunoffRule<A> {
	
	/**
	 * The number of bots eliminated by instant runoff before
	 * a traditional bankroll rule is used.
	 */
	int leftOver;
	
	Vector<A> bots;
	Hashtable<Pair<A,A>,Integer> utilityMatrix;
	Hashtable<A,Integer> utilityVector;
	
	public InstantRunoffRule(Vector<A> bots, int[][] utilities){
		this(bots,utilities,0);
	}
	public InstantRunoffRule(Vector<A> bots, int[][] utilities, int leftOver){
		this.bots = bots;
		this.utilityMatrix = new Hashtable<Pair<A,A>,Integer>();
		for(int i=0;i<utilities.length;i++){
			for(int j=0;j<utilities[i].length;j++){
				Pair<A,A> pair  = new Pair<A,A>(bots.get(i),bots.get(j));
				/*Pair<A,A> pair2 = new Pair<A,A>(bots.get(i),bots.get(j));
				System.out.println("FIRST:"+pair.first);
				System.out.println("SECOND:"+pair.second);	*/
				this.utilityMatrix.put(pair,utilities[i][j]);
				/*System.err.println("pair.hashCode()="+pair.hashCode());
				System.err.println("pair2.hashCode()="+pair2.hashCode());
				System.out.println("CONTAINS:"+this.utilityMatrix.containsKey(pair));
				System.out.println("CONTAINS2:"+this.utilityMatrix.containsKey(pair2));
				System.out.println("get:"+this.utilityMatrix.get(pair));
				System.out.println("get2:"+this.utilityMatrix.get(pair2));
				System.err.println("EQUALS:"+pair.equals(pair2));
				System.err.println("EQUALS2:"+pair2.equals(pair));
				*/
			}
		}
		this.leftOver = leftOver;
		initUtilityVector();
	}

	public InstantRunoffRule(Vector<A> bots, Hashtable<Pair<A,A>,Integer> utilities){
		this(bots,utilities,0);
	}
	
	public InstantRunoffRule(Vector<A> bots, Hashtable<Pair<A,A>,Integer> utilities, int leftOver){
		this.leftOver = leftOver;
		this.bots = bots;
		this.utilityMatrix = utilities;
		initUtilityVector();
		
	}
	
	public InstantRunoffRule<A> removeAllLosers(Vector<A> losers){
		Vector<A> remainingBots = new Vector<A>(bots);
		remainingBots.removeAll(losers);
		Hashtable<Pair<A,A>,Integer> remainingUtilities=new Hashtable<Pair<A,A>,Integer>();
		for(A b:remainingBots){
			for(A b2:remainingBots){
				Pair<A,A> botPair = new Pair<A,A>(b,b2);
				remainingUtilities.put(botPair, utilityMatrix.get(botPair));
			}
		}
		return new InstantRunoffRule<A>(remainingBots,remainingUtilities,leftOver);
	}
	
	public int getUtility(A bot){
		int totalSoFar = 0;
		for(A otherBot:bots){
			Pair<A,A> pair = new Pair<A,A>(bot,otherBot);
			/*System.out.println(utilityMatrix.containsKey(pair));
			System.out.println("FIRST:"+pair.first);
			System.out.println("SECOND:"+pair.second);*/
			totalSoFar += utilityMatrix.get(pair);
		}
		return totalSoFar;
	}
	
	public void initUtilityVector(){
		utilityVector = new Hashtable<A,Integer>();
		for(A bot:bots){
			utilityVector.put(bot,getUtility(bot));
		}
	}
	
	public int getMinUtility(){
		int minSoFar = Integer.MAX_VALUE;
		for(A bot:bots){
			int currentUtil = utilityVector.get(bot);
			if (currentUtil<minSoFar){
				minSoFar = currentUtil;
			}
		}
		return minSoFar;
	}
	
	public Vector<A> getLosers(){
		return getLosersInSet(bots);
		
	}
	
	public Vector<A> getLosersInSet(Vector<A> subset){
		int minSoFar = Integer.MAX_VALUE;
		Vector<A> losers = new Vector<A>();
		for(A bot:subset){
			int currentUtil = utilityVector.get(bot);
			if (currentUtil<minSoFar){
				minSoFar = currentUtil;
				losers = new Vector<A>();
				losers.add(bot);
			} else if (currentUtil==minSoFar){
				losers.add(bot);
			}
		}
		return losers;
	}
	
	/**
	 * Ranks players according to bankroll
	 * @return
	 */
	public Vector<Vector<A> > getBankrollRankings(){
		Vector<Vector<A> > result = new Vector<Vector<A> >(); 
		Vector<A> remainingBots = bots;
		while(remainingBots.size()>0){
			Vector<A> losers = getLosersInSet(remainingBots);
			remainingBots.removeAll(losers);
			result.insertElementAt(losers, 0);
		}
		return result;
	}
	
	/**
	 * Uses instant runoff to eliminate weak players, then
	 * rank the remaining bots through bankroll
	 * @param leftOver the maximum number to iterate through bankroll
	 * @return the ranking
	 */
	public Vector<Vector<A> > getRankings(){
		if (bots.size()<=leftOver){
			return getBankrollRankings();
		}
		Vector<A> losers = getLosers();
		InstantRunoffRule<A> remains = removeAllLosers(losers);
		Vector<Vector<A> > result = remains.getRankings();
		result.add(losers);
		return result;
	}
	
}