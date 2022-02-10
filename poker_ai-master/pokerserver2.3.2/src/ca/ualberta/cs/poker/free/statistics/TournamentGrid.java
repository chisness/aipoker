package ca.ualberta.cs.poker.free.statistics;

import java.io.BufferedWriter;
import java.io.IOException;
import java.text.NumberFormat;
import java.util.Locale;
import java.util.Vector;


public class TournamentGrid {

	DataSet dataSet;
	RandomVariable[][] variables;
	Vector<String> players;
	boolean writeMean;
	boolean writePopulation;
	boolean writeEstimate;
	boolean writeAverages;
	boolean writeSums;
	boolean colorSignificance=true;
	NumberFormat numberFormat;
	Vector<String> rowTitles;
	Vector<String> columnTitles;
	int height;
	int width;
	
	/**
	 * 
	 * Produce a N x N chart in text or HTML form to show a grid of RandomVariables (specificly
	 * its subclasses) and its values. It is assumed we want to show the RandomVariable's mean
	 * and standard deviation, and black out the diagonal since a player playing itself in a
	 * tournament is unlikely
	 * 
	 * @param data DataSet to produce the chart
	 * @param fractionDigits number of digits to show in the chart
	 * @param mean display the mean
	 * @param population include population std dev
	 * @param estimate include estimate std dev
	 */
	public TournamentGrid( DataSet data, int fractionDigits, boolean mean, boolean population, boolean estimate) {
		
		dataSet = data;
		players = data.getPlayers();
		height = players.size();
		width = players.size();
		variables = new RandomVariable[players.size()][players.size()];
		writeMean = mean;
		writePopulation = population;
		writeEstimate = estimate;
		
		numberFormat = NumberFormat.getNumberInstance(Locale.getDefault() );
		numberFormat.setMaximumFractionDigits( fractionDigits );
		
		fillTopRowTitles();
		fillColumnTitles();
		
	}
	
	/**
	 * Constructor with common options, namely Mean +- EstimateStdDev
	 * 
	 * @param data
	 * @param fractionDigits
	 */
	public TournamentGrid( DataSet data, int fractionDigits) {
		this(data, fractionDigits, true, false, true );
	}
	
	/**
	 * Extended TournamentGrid including an end column containing the average or sums of each row
	 * Made players a paramter
	 * 
	 * @param data
	 * @param fractionDigits
	 * @param mean
	 * @param population
	 * @param estimate
	 * @param endColumnaverages
	 * @param endColumnSums
	 * @param _players
	 */
	public TournamentGrid( DataSet data, int fractionDigits, boolean mean, boolean population, boolean estimate, boolean endColumnaverages, boolean endColumnSums, Vector<String> _players) {
		
		dataSet = data;
		players = _players;
		height = players.size();
		width = players.size();
		
		/* take the extra column into account */
		if ( endColumnaverages || endColumnSums ) {
			width = players.size() + 1;
		}
		
		variables = new RandomVariable[height][width];
		writeMean = mean;
		writePopulation = population;
		writeEstimate = estimate;
		writeAverages = endColumnaverages;
		writeSums = endColumnSums;
		
		numberFormat = NumberFormat.getNumberInstance(Locale.getDefault() );
		numberFormat.setMaximumFractionDigits( fractionDigits );
		
		// fill the vectors for row and column headers (player names)
		fillTopRowTitles();
		fillColumnTitles();
	}
	/**
	 * Extended TournamentGrid including an end column containing the average or sums of each row
	 * 
	 * 
	 * @param data
	 * @param fractionDigits
	 * @param mean
	 * @param population
	 * @param estimate
	 * @param endColumnaverages
	 * @param endColumnSums
	 */
	public TournamentGrid( DataSet data, int fractionDigits, boolean mean, boolean population, boolean estimate, boolean endColumnaverages, boolean endColumnSums) {
		
		dataSet = data;
		players = data.getPlayers();
		height = players.size();
		width = players.size();
		
		/* take the extra column into account */
		if ( endColumnaverages || endColumnSums ) {
			width = players.size() + 1;
		}
		
		variables = new RandomVariable[height][width];
		writeMean = mean;
		writePopulation = population;
		writeEstimate = estimate;
		writeAverages = endColumnaverages;
		writeSums = endColumnSums;
		
		numberFormat = NumberFormat.getNumberInstance(Locale.getDefault() );
		numberFormat.setMaximumFractionDigits( fractionDigits );
		
		// fill the vectors for row and column headers (player names)
		fillTopRowTitles();
		fillColumnTitles();
	}
	
	/**
	 * Wrap the generation of row and column headers so we can 
	 * better handle multiple outputs
	 *
	 */
	public void fillTopRowTitles() {
		rowTitles = new Vector<String>(width);	
		rowTitles.addAll( players );
		if( writeAverages ) 
			rowTitles.add( "Average");
	}
	
	public void fillColumnTitles() {
		columnTitles = new Vector<String>(width);
		columnTitles.addAll(players);
		
	}
	
	public static void main(String[] args) {	}
	

	/**
	 * Fill the array of variables that will be used for the tables with
	 * SmallBetPerHand variables. You could be fancy and do this stuff on the
	 * fly, but since each RandomVariable will have a different constructor this
	 * gets overkill really quick
	 * 
	 * A method for each type of RandomVariable will have to be created
	 * 
	 * firstHand and lastHand are parameters in the interest of future competitions
	 * where [0,999] might not be the default
	 * 
	 * @param firstHand
	 * @param lastHand
	 */
	public void fillSmallBetsPerHand( int firstHand, int lastHand ) {
		for( int i = 0; i < players.size(); i++ ) {
			for( int j = 0; j < players.size(); j++ ) {
				if ( i == j ) {
					variables[i][j] = null;
				} else {
					variables[i][j] = new SmallBetPerHandVariable(players.get(i),players.get(j), firstHand, lastHand, dataSet );
				}
			}
		}
	}
	
	/**
	 * Fill the rightmost column with the averages of the row
	 *
	 * Treat the last column as a row of random variables, giving us
	 * access to sums, means, std dev, etc.
	 */
	public void fillEndColumnAverages( ) {
		
		// create an empiracle array for this grid
		for( int i = 0; i < height; i++ ) {
			WeightedRandomVariable wrv = new WeightedRandomVariable(dataSet);
			for( int j = 0; j < width-1; j++ ) {
				if ( i != j ){
					wrv.averageIn(variables[i][j]);
				}
			}	
			
			// instantiate a variable with the empArray
			variables[i][width-1] = wrv;
		}
	}
	
	/**
	 * Fill the rightmost column with the averages of the row
	 *
	 * Treat the last column as a row of random variables, giving us
	 * access to sums, means, std dev, etc.
	 */
	public void fillEndColumSums( ) {
		
		for( int i = 0; i < height; i++ ) {
			WeightedRandomVariable wrv = new WeightedRandomVariable(dataSet);
			for( int j = 0; j < width-1; j++ ) {
				if ( i != j ){
					wrv.add(variables[i][j]);
				}
			}	
			variables[i][width-1] = wrv;
			
		}
	}
	
	
	/**
	 * write an HTML file to disk containing a header for the grid, plus a colored output
	 * determined by the constructor values
	 * 
	 * @param writer
	 * @throws IOException
	 */
	
	/**
	 * write an HTML file to disk containing a header for the grid, plus a colored output
	 * determined by the constructor values
	 * 
	 * @param writer
	 * @throws IOException
	 */
	public void writeHTML(BufferedWriter writer) throws IOException {
		
		//writer.write( "<h4> " );
		//if ( writeMean ) writer.write( "Mean " );
		//if( writePopulation) writer.write( " &plusmn; Population Standard Deviation");
		//if( writeEstimate ) writer.write( " &plusmn; Estimate Standard Deviation");
		
		
		writer.write("<table style=\"color: black;\" align=\"center\" bgcolor=\"white\" border=\"1\">\n");

		// first row
		writer.write("<tbody><tr><td></td>\n");
		for (String player : rowTitles) {
			writer.write("<td>" + player + "</td>\n");
		}
		writer.write("</tr>\n");
		

		// each row
		for (int i = 0; i < height; i++) {
			writer.write("<tr>");
			
			// each column
			writer.write("<td> " + columnTitles.get(i) + " </td>");
			
			
			for (int j = 0; j < width-1; j++) {
				
				// fill player i vs player i with a black cell
				if (i == j) {
					writer.write("<td bgcolor=\"#888888\"></td>");
				} else {
					
					double meanValue = getMeanValue(i,j);
					double estimateStdDev = getEstimateStdDev(i,j);
					
					if (colorSignificance){
						//double significance = Math.abs(meanValue)/estimateStdDev;
						//int colorSignificance = (int)(255.0 * (1.0-(significance/4.0)));
						if (Math.abs(meanValue)<estimateStdDev*2.0){
							if( meanValue > 0 ){
								writer.write( "<td bgcolor=\"#CCFFCC\">");
							} else {
								writer.write( "<td bgcolor=\"#FFCCCC\">");						
							}
						} else {
							if( meanValue > 0 ){
								writer.write( "<td bgcolor=\"#88FF88\">");
							} else {
								writer.write( "<td bgcolor=\"#FF8888\">");						
							}
						}
					} else {
						if( meanValue > 0 ){
							writer.write( "<td bgcolor=\"#88FF88\">");
						} else {
							writer.write( "<td bgcolor=\"#FF8888\">");
						}
					}
					
					//System.out.println( " i = " + i + " j = " + j);
					if( writeMean ) {
						writer.write(numberFormat.format( meanValue ));
					}
					
					if ( writePopulation ) writer.write( " &plusmn; " + numberFormat.format( getPopulationStdDev(i, j)));
					
					if( writeEstimate ) writer.write( " &plusmn; " + numberFormat.format( getEstimateStdDev(i, j)));
					
					writer.write( "</td>\n");
				}
			}
			
			if( writeAverages||writeSums){
				writer.write("\n<td> " + numberFormat.format( variables[i][width-1].getSampleMean()) );
				if ( writePopulation ) writer.write( " &plusmn; " + numberFormat.format( getPopulationStdDev(i, width-1)));
				if( writeEstimate ) writer.write( " &plusmn; " + numberFormat.format( getEstimateStdDev(i, width-1)));
				writer.write( "</td>\n");
			}
			
			writer.write("</tr>\n");

		}

		writer.write("</tbody> </table> ");
		writer.close();
	}
	
	
	public void writeTXT() {
		
		System.out.println( "START CHART");
		// write the top row
		for ( String player: players ) {
			System.out.print( player + " ");
		}
		System.out.print("\n");
		
		for( int i = 0; i < players.size(); i++ ) {
			System.out.print( "\n" + players.get(i) + " ");
			for( int j = 0; j < players.size(); j++ ) {
				if ( i == j ) {
					System.out.print( "XXX ");
				} else {
					System.out.print( getMeanValue(i, j) + " " );
				}
			}
			System.out.print("\n");
			
		}

		System.out.println( "\nEND CHART");

	}
	
	public double getMeanValue( int i, int j ){
		return variables[i][j].getSampleMean();
	}
	public double getPopulationVariance( int i, int j ){
		return variables[i][j].popVariance();
	}
	public double getPopulationStdDev( int i, int j ){
		return variables[i][j].popStdDev();
	}
	public double getEstimateVariance( int i, int j ){
		return variables[i][j].estVariance();
	}
	public double getEstimateStdDev( int i, int j ){
		return variables[i][j].estStdDev();
	}
	
	
	public void fillVariables() {
		
		variables = new RandomVariable[players.size()][players.size()];
		
		for( int i = 0; i < players.size(); i++ ) {
			for( int j = 0; j < players.size(); j++ ) {
				if ( i == j ) {
					variables[i][j] = null;
				} else {
					variables[i][j] = new SmallBetPerHandVariable(players.get(i),players.get(j), 0, 999, dataSet );
				}
			}
		}
	}

	
	public void fillAverageDiffs(){
		for( int i = 0; i < players.size(); i++ ) {
			for( int j = 0; j < players.size(); j++ ) {
				if ( i == j ) {
					variables[i][j] = null;
				} else {					
					variables[i][j] = WeightedRandomVariable.getDifference(dataSet,variables[i][players.size()],variables[j][players.size()]);
				}
			}
		}
		
	}
}
