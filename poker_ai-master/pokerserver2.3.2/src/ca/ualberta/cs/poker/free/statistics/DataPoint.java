package ca.ualberta.cs.poker.free.statistics;

import java.util.Vector;

/***
 * A datapoint is a vector of matches that contained a particular hand of cards
 * 
 * Datapoint is used for statistical analysis
 * 
 * @author Christian Smith
 *
 */
public class DataPoint {
	
	public Vector<AbstractMatchStatistics> matches;
	
	public DataPoint( Vector<AbstractMatchStatistics> _matches) {
		//System.out.println( "New datapoint of size " + _matches.size() );
		matches = new Vector<AbstractMatchStatistics>();
		matches = _matches;
	}
	
	public DataPoint( AbstractMatchStatistics _matches) {
		matches = new Vector<AbstractMatchStatistics>();
		matches.add( _matches );
	}
	
	public void addMatch( AbstractMatchStatistics match ) {
		matches.add( match );
	}
	
	public String toString() {
		
		String ret = "\nDataPoint:";
		ret += "\nsize " + matches.size();
		//ret += "\nmatches: " + matches.get(0).toString();
		
		
		return ret;
		
	}
	
}
