package ca.ualberta.cs.poker.free.statistics;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Iterator;
import java.util.Vector;

/**
 * 
 * This class is to turn the results from a set of matches, .res files, and
 * parse them into datapoints. Dataset is then intended to be used by a RandomVariable
 * or inherited classes to glean useful data from the points contained in the dataset.
 * 
 * 
 * @author chsmith
 *
 */
public class DataSet {
	
	public Vector<DataPoint> dataPoints;
	public Vector<AbstractMatchStatistics> matches;
	
//	public DataSet() {
//		dataPoints = new Vector<DataPoint>();
//	}
	
	
	
	public DataSet( Vector<DataPoint> _datapoints) {
		dataPoints = _datapoints;
	}
	
	public Vector<String> getPlayers() {
		Vector<String> players = new Vector<String>();
		
		// for every pair of players in each match, add that players name
		for ( DataPoint dp: dataPoints ) {
			for( AbstractMatchStatistics ms: dp.matches ) {
				for (String name:ms.getPlayers()){
					if (!players.contains(name)){
						players.add(name);
					}
				}
			}
		}

		return players;
	}
	
	/**
	 * Given a directory, find all result files, create datapoints of each
	 * one and add the matches to the dataset
	 * 
	 * @param directory
	 */
	public DataSet( String directory ) {
		
		dataPoints = new Vector<DataPoint>();
		matches = new Vector<AbstractMatchStatistics>();
		
		try {
			loadDataPoints(directory);
			addDataPoints();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println( "I/O error reading results files...");
			e.printStackTrace();
		}	
	}
	
	/**
	 * Initialize an empty data set
	 * 
	 */
    public DataSet( ) {
		dataPoints = new Vector<DataPoint>();
		matches = new Vector<AbstractMatchStatistics>();
    }
    
    public static DataSet getLightDataSet(String directory){
		DataSet result = new DataSet();
		try {
			result.loadLightDataPoints(directory);
			result.addDataPoints();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			System.out.println( "I/O error reading results files...");
			e.printStackTrace();
		}
		return result;
		
	}
	
	
	
	
	public void loadLightDataPoints(String file)throws FileNotFoundException, IOException {
		File f = new File(file);
		LightMatchStatistics newMatch;
		
		if (!(f.exists())){
			System.err.println("File not found:"+file);
		} else if (f.isDirectory()){
			System.err.println("Descending into directory "+file);
			String[] files = f.list();
			if (!file.endsWith(File.separator)){
				file+=File.separator;
			}
			
			for(String subFile:files){
				loadLightDataPoints(file+subFile);
			}
		} else if (!file.endsWith(".res")){
			//System.err.println("File "+file+" passed over.");
		} else {
		  //System.err.println("Loading match "+file+"...");
		  
		  
		  
	  // load the next match
	 
		  try {
				newMatch = new LightMatchStatistics(file);

				matches.add(newMatch);

				//System.err.println("Loaded match:" + file);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				System.out.println("Problem reading results file "
						+ file + " , ignoring...");
				e.printStackTrace();
			}
		

		}
		
	}
	/**
	 * 
	 * Given a 
	 * 
	 * @param file
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void loadDataPoints( String file ) throws FileNotFoundException, IOException {
		
		File f = new File(file);
		MatchStatistics newMatch;
		
		if (!(f.exists())){
			System.err.println("File not found:"+file);
		} else if (f.isDirectory()){
			System.err.println("Descending into directory "+file);
			String[] files = f.list();
			if (!file.endsWith(File.separator)){
				file+=File.separator;
			}
			
			for(String subFile:files){
				loadDataPoints(file+subFile);
			}
		} else if (file.endsWith(".res")){
			System.err.println("Result file "+file+" passed over.");
		} else {
		  System.err.println("Loading match "+file+"...");
		  
		  
		  
	  // load the next match
	 
		  try {
			newMatch = new MatchStatistics(file);
						  
			  // make sure we haven't seen the match already
//				  boolean duplicate = false;
//				  for( MatchStatistics existing: matches ) {
//					  if( newMatch.isDuplicate(existing) ) {
//						  duplicate = true;
//					  }
//				  }
//				  
//				  if( !duplicate ) {
				  matches.add(newMatch);
			  //}
			  
			  System.err.println("Loaded match:"+file);
			  if (!newMatch.confirm()){
				System.err.println("Match does not agree with ring policy:"+file);
				System.exit(0);
			  } else {
				System.err.println("Match appears good:"+file);
			  }
			  
			  
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println( "Found a null results file while parsing " + file + " , ignoring...");
			//e.printStackTrace();
		}
		

		}
	
	}
	
	/**
	 * Iterate over the vector of matchstatistics and create the array of 
	 * DataPoints
	 * 
	 * @param matchstats
	 */
	public void addDataPoints () {
		System.out.println( matches.size() + " matches to add" );
		
		//this will keep track of all the matchstatistics with matching cards
		Vector<AbstractMatchStatistics> dataPointMatches;
		
		
		// take each match, parse into datapoints for this dataset
		while( !matches.isEmpty() ) {
			
			Iterator<AbstractMatchStatistics> iter = matches.iterator();
			
			//pick the first element
			AbstractMatchStatistics first = iter.next();
			
			//remove the one we're starting with
			iter.remove();
			
			// the first element will be a datapoint even if no others are found
			dataPointMatches = new Vector<AbstractMatchStatistics>();
			dataPointMatches.add( first );
			
			
			// now iterate over the list looking for matches with dup cards
			while( iter.hasNext()){
				AbstractMatchStatistics candidate = iter.next(); 
														
				if( candidate.isDuplicateCards(first)) {
					//System.out.println( "Found match " + candidate.hashCode());
					dataPointMatches.add( candidate );
					iter.remove();
				}
			}
			
			//now create the datapoint associated with these matches we just found
			DataPoint dp = new DataPoint( dataPointMatches );
			dataPoints.add( dp );
		}
		
		System.out.println( "Done adding datapoints. Found " + dataPoints.size() + " datapoints.");
				
	}
	
	
}
