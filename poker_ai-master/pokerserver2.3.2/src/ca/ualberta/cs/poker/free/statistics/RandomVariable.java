package ca.ualberta.cs.poker.free.statistics;

import java.util.ArrayList;

/**
 * RandomVariable is intended to provide a more robust interface to the data generated
 * by the poker server. It is intended to be extended by a class such as SmallBetPerHandVariable
 * which will define the abstract methods, isDefined and getValue. It is always initialised on
 * the dataset that it will use.
 * 
 * RandomVariable is abstract and should be extended for the type
 * of random variable you are defining. 
 * 
 * 
 * @author Christian Smith
 *
 */
public abstract class RandomVariable {

	private double[] empArray = null;
	private DataSet dataSet;
	
	abstract boolean isDefined( DataPoint dp );
	abstract Double getValue( DataPoint dp );
	
	/**
	 * Initialise the RandomVariable with a DataSet, since it is unlikely we will 
	 * ever want a RandomVariable that is not associated with a DataSet.
	 * 
	 * @param _dataSet
	 */
	public RandomVariable( DataSet _dataSet ) {
		dataSet = _dataSet;
	}
	
	/**
	 * Fill this randomvariable's empiracal array with data from the given dataset.
	 * For each datapoint, if it is defined, get its value (determined by the class
	 * extending randomvariable) and place its value in the array
	 * 
	 *
	 */
	protected void fillEmpiricalArray() {
		// fill an arraylist first, we dont know how many elements
		ArrayList<Double> empValues = new ArrayList<Double>();
		
		for( DataPoint dp: dataSet.dataPoints ) {	
			if ( isDefined( dp ) ) {
				empValues.add( new Double( getValue(dp) ) );
			}
		}
		
		// now we know how big emparray is going to be
		empArray = new double[ empValues.size() ];
		
		// now fill it
		for( int i = 0; i < empValues.size(); i ++ ) {
			empArray[i] = empValues.get(i).doubleValue();
		}
		
		// print this error, because we will nearly always want duplicate match pairs
		if ( empArray.length < 2 ) {
			System.out.println( "WARNING: Not enough data to compute the standard deviation");
		}
	}
	
	final double[] getEmpiracleArray() {
		return empArray;
	}
	
	final void setEmpiracleArray(double[] array) {
		empArray = array.clone();
		//fillEmpiricalArray();
	}
	
	/**
	 * Make sure the empirical array is filled before accessing it
	 *
	 */
	protected void validate() {
		if( empArray == null ) {
			fillEmpiricalArray();
		}
	}
	
	/**
	 * Return the sum of the empirical array for the values
	 * 
	 * This would be useful if you want to the total bankroll won, per hand, instead of the 
	 * mean bankroll per hand
	 * 
	 * Both units will remain in small bets/hand, but saves you from multiplying by the number
	 * of matches in question
	 * 
	 * @return
	 */
	final double getSumOfValues() {
		double sum = 0.0;
		validate();
		
		if ( empArray.length == 0 ) {
			return Double.NaN;
		}
		
		for ( int i = 0; i < empArray.length; i ++ ) {
			sum += empArray[i];
		}
		
		return sum;
	}
	
	/**
	 * Sum all values of the empArray, return that divided by the length
	 * 
	 */
	final double getSampleMean() {
		double sum = 0.0;
		validate();
		
		if ( empArray.length == 0 ) {
			return Double.NaN;
		}
		
		for ( int i = 0; i < empArray.length; i ++ ) {
			sum += empArray[i];
		}
		
		return sum / empArray.length;
	}
	
	/**
	 * Return the population variance. 
	 * 
	 * Return the sum of (values-mean) squared divided by n - 1
	 * 
	 * @return
	 */
	final double popVariance() {
		validate();
		double mean = getSampleMean();
		final int n = empArray.length;
		
		if (n < 2) {
			return Double.NaN;
		}

		// calculate the sum of differences squared
		double sum = 0;
		for (int i = 0; i < n; i++) {
			final double v = empArray[i] - mean;
			sum += v * v;
		}
		
		return sum / (n - 1);
		
	}
	
	/**
	 * Return the estimated (biased) variance. 
	 * 
	 * Return the sum of (values-mean) squared divided by n
	 * 
	 * @return
	 */
	final double estVariance() {
		validate();
		return popVariance() /  empArray.length;
	}
	
	/**
	 * Return the population standard deviation, which is the sqrt of the population variance
	 * 
	 * @return
	 */
	final double popStdDev() {
		return Math.sqrt( popVariance() );
	}

	/**
	 * Return the estimated standard deviation, which is the sqrt of the estimated variance
	 * 
	 * @return
	 */
	final double estStdDev() {
		return Math.sqrt( estVariance() );
	}
	
	final void printEmpiricalArray() {
		validate();
		
		System.out.print( "Empirical Array = " );
		for ( int i = 0; i < empArray.length; i++ ) {
			System.out.print( empArray[i] + " " );
		}
		System.out.print("\n");
	}
	
}
