package ca.ualberta.cs.poker.free.statistics;

import java.util.Vector;

/**
 * SmallBetPerHand will calculate the average SmallBet (2 times the small blind) 
 * per hand, for a given player and opponent over the specified hands
 * 
 * The values returned are the average of the two matches with opposite seatings: 
 * (player vs. opponent) and (opponent vs. player)
 * 
 * If any DataPoint (card file) does not contain two matches (matchstatistics)
 * that datapoint will be ignored. The DataPoint must also be defined over the
 * hands (firsthand:lasthand), otherwise it will be ignored as well.
 * 
 * @author chsmith
 *
 */
public class SmallBetPerHandVariable extends RandomVariable {

	String player;

	String opponent;

	int lengthOfGame;

	int firstHand;

	int lastHand;

	double smallBlindsInASmallBet = 2.0;

	public SmallBetPerHandVariable(String _player, String _opponent, int _firstHand, int _lastHand, DataSet data) {
		super(data);
		player = _player;
		opponent = _opponent;
		firstHand = _firstHand;
		lastHand = _lastHand;
	}

	/**
	 * Does the DataPoint in question have player and opponent for the hands specified in
	 * the constructor
	 * 
	 * This will verify that match has the data we are looking for
	 * 
	 * @param match
	 * @return
	 */
	public boolean isDefined(DataPoint dp) {

		boolean ret = isDefined(dp,player,opponent)&& isDefined(dp,opponent,player);

		if (!ret){
			System.err.println("Match between "+player+" and "+opponent+" missing.");
		}
		//System.out.println("isDefined() returning " + ret);
		return ret;
	}

	public boolean isDefined(DataPoint dp, String a, String b) {

		boolean ret = false;

		//each datapoint is a vector of matches, check for the players and hands we are looking for
		for (AbstractMatchStatistics cur : dp.matches) {
			
			// make sure the players are correct, and the hands in question exist
			if ((cur.isDefined(a, b))
					&& cur.getFirstHandNumber() <= firstHand
					&& cur.getLastHandNumber() >= lastHand) {
				ret = true;
			}
		}

		//System.out.println("isDefined() returning " + ret);
		return ret;
	}

	/**
	 * Given a datapoint, check that it contains a matchstatistics that we are looking for,
	 * containing player vs. opponent or oppenent vs. player, get the value, and average 
	 * the result over the hands specified in the constructor
	 * 
	 * @param dp DataPoint to get the average smallbetsperhand from
	 */
	Double getValue(DataPoint dp) {

		Vector<AbstractMatchStatistics> matchStats = dp.matches;
		Vector<AbstractMatchStatistics> matchingMatches = new Vector<AbstractMatchStatistics>();
		double value = 0.0;

		// find the matches to average
		for (AbstractMatchStatistics cur : matchStats) {
			if (cur.isDefined(player, opponent)	|| cur.isDefined(opponent, player)) {
				matchingMatches.add(cur);
			}
		}

		// there should be 2 matches per datapoint, since we are switching seat positions each time
		for (AbstractMatchStatistics cur : matchingMatches) {
			value += getValue(cur);
		}
		
		value /= matchingMatches.size();

		return value;
	}

	/**
	 * Get the SmallBetsPerHand value from this match, for the hands specified in firstHand
	 * and LastHand. This will be averaged in getValue( DataPoint ), this method is simply
	 * to get the value we are interested in once we find matches we are interested in.
	 * 
	 * @param match
	 * @return
	 */
	double getValue(AbstractMatchStatistics match) {

		double value = match.getUtility(player, opponent, firstHand, lastHand) / (double)match.getSmallBlindsInASmallBet();
		
		return value / (lastHand + 1.0 - firstHand);
	}

}
