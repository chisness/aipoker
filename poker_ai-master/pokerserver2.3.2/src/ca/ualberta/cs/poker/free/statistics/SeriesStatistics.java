package ca.ualberta.cs.poker.free.statistics;

import java.util.Hashtable;
import java.util.Vector;

public class SeriesStatistics {
	Vector<Vector<MatchStatistics> > matches;
	public SeriesStatistics(){
		matches = new Vector<Vector<MatchStatistics> >();
	}
	
	public SeriesStatistics(MatchStatistics match){
		this();
		add(match);
	}
	
	public SeriesStatistics(Vector<MatchStatistics> moreMatches){
		this();
		addAll(moreMatches);
	}
	
	public void addAll(Vector<MatchStatistics> moreMatches){
		for(MatchStatistics match:moreMatches){
			add(match);
		}
	
	}
	/**
	 * Add a match to this series.
	 * TODO assert that the added match fits with this set
	 * @param match the match to add
	 */
	public void add(MatchStatistics match){
		for(Vector<MatchStatistics> matchSet:matches){
			MatchStatistics matchHere = matchSet.firstElement();
			if (matchHere.isDuplicate(match)){
				matchSet.add(match);
				return;
			}
		}
		Vector<MatchStatistics> newMatchSet = new Vector<MatchStatistics>();
		newMatchSet.add(match);
		matches.add(newMatchSet);
	}
	
	public Vector<String> getPlayers(){
		MatchStatistics match = matches.firstElement().firstElement();
		return match.getPlayers();
	}
	
	public Vector<MatchStatistics> getAllMatches(){
		Vector<MatchStatistics> result=new Vector<MatchStatistics>();
		for(Vector<MatchStatistics> matchSet:matches){
			result.addAll(matchSet);
		}
		return result;
	}
	
	public static Hashtable<String,Integer> getUtilities(Vector<MatchStatistics> matches, int firstHand, int lastHand){
		Hashtable<String,Integer> result=matches.firstElement().getUtilityMapInSmallBlinds(firstHand, lastHand);
		for(int i=1;i<matches.size();i++){
			matches.get(i).addUtility(result,firstHand,lastHand);
		}
		return result;
	}
	
	
	
	public Hashtable<String,Integer> getUtilities(int firstHand, int lastHand){
		Vector<MatchStatistics> allMatches=getAllMatches();
		return getUtilities(allMatches,firstHand,lastHand);
	}
	
	public Hashtable<String,Double> getStandardDeviation(int firstHand, int lastHand){
		return MapOperationsD.sqrt(getVarianceOfSampleMean(firstHand,lastHand));
	}
	public Hashtable<String,Double> getVarianceOfSampleMean(int firstHand, int lastHand){
		MapOperationsD sum = new MapOperationsD();
		MapOperationsD squareSum = new MapOperationsD();
		for(Vector<MatchStatistics> duplicateMatchSet:matches){
			MapOperationsD current = MapOperationsD.cast(getUtilities(duplicateMatchSet,firstHand,lastHand));
			Hashtable<String,Double> currentSq = current.square();			
			sum.increment(current);
			squareSum.increment(currentSq);
//			System.out.println("CURRENT:"+current);
//			System.out.println("SUM:"+sum);
//			System.out.println("currentSq:"+currentSq);
//			System.out.println("squareSum:"+squareSum);
		}
//		System.out.println("SUM:"+sum);
//		System.out.println("squareSum:"+squareSum);
		MapOperationsD average = sum.divide(matches.size());
		MapOperationsD averageSquared = average.square();
		MapOperationsD averageOfSquares = squareSum.divide(matches.size());
//		System.out.println("averageSquared:"+averageSquared);
//		System.out.println("averageOfSquares:"+averageOfSquares);
		MapOperationsD difference = averageOfSquares.subtract(averageSquared);
//		System.out.println("difference:"+difference);
		MapOperationsD unbiasedVariance = difference.divide(matches.size()-1);
		return unbiasedVariance;
	}
}
