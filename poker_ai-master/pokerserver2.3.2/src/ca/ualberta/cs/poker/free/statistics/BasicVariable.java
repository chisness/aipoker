package ca.ualberta.cs.poker.free.statistics;

/**
 * Shell variable if you want to do RandomVariable operations if you have
 * an empirical array. This was needed once we want to take means and std.
 * deviations on means of other variables, such as in tournament grid
 * 
 * Only use if you have set the empArray manually
 * 
 * @author chsmith
 *
 */
public class BasicVariable extends RandomVariable {
	
	/**
	 * You must pass in an empArray, because the data
	 * may not necessarily correspond to the emparray
	 *
	 */
	public BasicVariable( DataSet data, double[] empArray ) {
		super(data);
		
		setEmpiracleArray(empArray);
		
	}
	
	Double getValue(DataPoint dp) {
		return null;
	}

	
	boolean isDefined(DataPoint dp) {
		return false;
	}

}
