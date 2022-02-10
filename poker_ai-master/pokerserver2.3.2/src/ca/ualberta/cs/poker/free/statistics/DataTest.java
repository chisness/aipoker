package ca.ualberta.cs.poker.free.statistics;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

/**
 * This is designed to print out charts for the poker competition.
 * @author maz
 *
 */
public class DataTest {

	public static void main(String[] args) {
		// generate simple N x N grid
		//testGrid();
		
		//testWeighted();
		
		// generate HTML including the average of each column
		String type = "average";
		String directory = "data/results";
		String outputName = "output";
		int precision = 3;
		boolean sequence = false;
		Vector<String> players = new Vector<String>();
		
		for(int i=0;i<args.length;i++){
			if (args[i].equals("-t")){
				i++;
				if (i==args.length){
					System.err.println("Usage");
				}
				type = args[i];
			} else if (args[i].equals("-d")){
				i++;
				if (i==args.length){
					System.err.println("Usage");
				}
				directory = args[i];
			} else if (args[i].equals("-p")){
				i++;
				if (i==args.length){
					System.err.println("Usage");
				}
				precision = Integer.parseInt(args[i]);
			} else if (args[i].equals("-s")){
				sequence = true;
			} else if (args[i].equals("-o")){
				i++;
				if (i==args.length){
					System.err.println("Usage");
				}
				outputName = args[i];
			} else {
				players.add(args[i]);
			}
		}
		if (sequence){
			while (players.size() >= 3) {
				if (type.equals("average")) {
					testLightAverage(directory, players, precision, outputName + "Top"
							+ players.size() + ".html");
				} else if (type.equals("diff")) {
					testLightAverageDiff(directory, players, precision,
							outputName+"Top" + players.size() + ".html");
				}
				players.removeElementAt(players.size() - 1);

			}
		} else {
		if (type.equals("average")){
			testLightAverage(directory,players,precision,outputName +".html");
		} else if (type.equals("diff")){
			testLightAverageDiff(directory,players,precision,outputName+".html");
		}
		}
	}
	
	public void showUsage(){
		System.err.println("Usage:");
		System.err.println("DataTest [<ARG>|<PLAYER>]*");
		System.err.println("Where <PLAYER> is a name of a bot");
		System.err.println("And <ARG> can be <DIR>|<TYPE>|<PRECISION>");
		System.err.println("<DIR>:=\"-d\" <DIRECTORY>|<FILENAME>");
		System.err.println("Where <DIRECTORY> is a directory with results files and <FILENAME> is the name of a results file.");
		
		System.err.println("<TYPE>:=\"-t\" [\"average\"|\"diff\"]");
		System.err.println("<PRECISION>:=\"-p\" <NUMDIGITS>");
		System.err.println("Where <NUMDIGITS> is the number of digits after the decimal");
		System.exit(-1);
	}
	
	
	public static void testLightAverage(String directory,Vector<String> players, int precision, String output) {
		
		DataSet ds = DataSet.getLightDataSet(directory);
		TournamentGrid tg=null;
		if (players.isEmpty()){
			tg = new TournamentGrid( ds, precision, true, false, true, true, false);
		} else {
			tg = new TournamentGrid( ds, precision, true, false, true, true, false, players);	
		}
		
		//TournamentGrid tg = new TournamentGrid( ds, 3 );//, true, false, false, true);
		
		AbstractMatchStatistics first = ds.dataPoints.get(0).matches.get(0);
		tg.fillSmallBetsPerHand(first.getFirstHandNumber(),first.getLastHandNumber());
		tg.fillEndColumnAverages();
		//tg.fillAverages();
		//tg.writeTXT();
		
		try {
			tg.writeHTML(new BufferedWriter( new FileWriter( output )));
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		
	}

	public static void testLightAverageDiff(String directory,Vector<String> players, int precision, String output) {
		
		DataSet ds = DataSet.getLightDataSet(directory);
		TournamentGrid tg=null;
		if (players.isEmpty()){
			tg = new TournamentGrid( ds, precision, true, false, true, true, false);
		} else {
			tg = new TournamentGrid( ds, precision, true, false, true, true, false, players);	
		}
		
		//TournamentGrid tg = new TournamentGrid( ds, 3 );//, true, false, false, true);
		
		AbstractMatchStatistics first = ds.dataPoints.get(0).matches.get(0);
		tg.fillSmallBetsPerHand(first.getFirstHandNumber(),first.getLastHandNumber());
		tg.fillEndColumnAverages();
		tg.fillAverageDiffs();
		//tg.writeTXT();
		
		try {
			tg.writeHTML(new BufferedWriter( new FileWriter( output )));
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		
	}

	/**
	 * Generate chart including the average of each row in the right most column
	 *
	 */
	public static void testAverage() {
		
		DataSet ds = new DataSet("data/results" );
		TournamentGrid tg = new TournamentGrid( ds, 3, true, false, true, true, false);
		//TournamentGrid tg = new TournamentGrid( ds, 3 );//, true, false, false, true);
		
		tg.fillSmallBetsPerHand(0,2);
		tg.fillEndColumnAverages();
		
		//tg.fillAverages();
		//tg.writeTXT();
		
		try {
			tg.writeHTML(new BufferedWriter( new FileWriter( "output.html" )));
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		
	}
	
	/*
	 * Test for using weightedvariable
	 */
	public static void testWeighted() {
		DataSet ds = new DataSet("data/results" );
		
		RandomVariable[] vars = new RandomVariable[2];
		vars[0] =  new SmallBetPerHandVariable( "Random1", "Random2", 0, 2, ds );
		vars[1] =  new SmallBetPerHandVariable( "Random2", "Random3", 0, 2, ds );
		
		//SmallBetPerHandVariable smallbet = new SmallBetPerHandVariable( "Monash-BPP", "Monash", 0, 2, ds );
		//smallbet.printEmpiricalArray();
		//System.out.println( "Smallbet = " + smallbet.getSampleMean());
		
		System.out.println( "Mean of Random1 vs Random2 = " + vars[0].getSampleMean() );
		System.out.println( "Mean of Random2 vs Random3 = " + vars[1].getSampleMean() );
		
		
		//WeightedRandomVariable wvg = new WeightedRandomVariable(vars, ds);
		
		//System.out.println( "Value of weighted sum with default args is " + wvg.getWeightedSum());
		//System.out.println( "Value of weighted mean with default args is " + wvg.getWeightedMean());
		
		
	}
	
	public void testVariable() {
		
		DataSet ds = new DataSet("data/results" );
		SmallBetPerHandVariable smallbet = new SmallBetPerHandVariable( "Monash-BPP", "Monash", 0, 999, ds );
		
		smallbet.printEmpiricalArray();
		
		System.out.println( "Mean value = " + smallbet.getSampleMean());
		
		System.out.println( "Est Std Dev = " + smallbet.estStdDev() );
		System.out.println( "Est Variance = " + smallbet.estVariance() );
		System.out.println( "Pop Std Dev = " + smallbet.popStdDev() );
		System.out.println( "Pop Variance = " + smallbet.popVariance() );
		
	}
	                            
	/**
	 * Generate simple N x N grid
	 * 
	 * @throws IOException
	 */
	public static void testGrid() throws IOException {
		DataSet ds = new DataSet("data/results" );
		TournamentGrid tg = new TournamentGrid( ds, 3);
		
		
		tg.fillSmallBetsPerHand(0,2);
		//tg.writeTXT();
		
		tg.writeHTML(new BufferedWriter( new FileWriter( "output.html" )));
		
	}

}
