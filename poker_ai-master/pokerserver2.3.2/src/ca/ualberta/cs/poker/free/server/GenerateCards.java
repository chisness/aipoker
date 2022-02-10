package ca.ualberta.cs.poker.free.server;


import java.security.SecureRandom;
import java.util.Vector;
import java.io.*;

import ca.ualberta.cs.poker.free.dynamics.Card;


/**
 * Generates a file with cards to be used in a duplicate match.
 * @author Christian Smith, Martin Zinkevich
 *
 */
public class GenerateCards  {

	
	/**
     * The randomness for the game. null if on the client side.
     */
    //SecureRandom random;
    
	/*
	public static void main(String[] args) throws Exception {
	    
		// right number of arguments?
		if ( args.length != 3 ) {
			System.out.println( "\nERROR: Incorrect number of arguments\n\n");
			printUsage();
		}
		
		int competitors = Integer.parseInt( args[1]);
		int matches = Integer.parseInt( args[2]);
		
		
		// sanity check
		if ( (competitors < 0 || competitors > 100) && (matches < 0 || matches > 100) ) {
			System.out.println("Please make sure numCompetitors and numMatches is reasonable. ie 0 < num < 100");
			System.exit(0);
		}
		
		// competition type
		if ( args[0].contains( "SE" ) ) {
			GenerateSingleElimination( competitors, matches );
		}
		else if ( args[0].contains( "RR") ) {
			GenerateRoundRobin( competitors, matches );
		}
		else if ( args[0].contains( "BOTH" )) {
			GenerateSingleElimination( competitors, matches );
			GenerateRoundRobin( competitors, matches );
		}
		else {
			System.out.println( "Error: type is not a valid: use SE or RR or BOTH\n\nReceived " + args[0]);
			printUsage();
			
		}
		
		
		System.out.println( "\nCompleted" );
		
	}
	*/
	
	public static void main(String[] args) throws IOException{
		if (args.length<3){
			System.err.println("Usage:java ca.ualberta.cs.poker.free.server.GenerateCards <outputDirectory> <inputDirectory1> <inputDirectory2> [<inputDirectory>]^*");
			System.err.println("Combines the results of several card files generated.");
		}
		File outputDirectory = new File(args[0]);
		Vector<File> inputDirectory = new Vector<File>();
		for(int i=1;i<args.length;i++){
			inputDirectory.add(new File(args[i]));
		}
		
		combineDirectories(inputDirectory,outputDirectory);
	}
	/*
	public static  void printUsage() {
		
		System.out.println( "\nUsage: " + "java GenerateCards type numCompetitors numMatches");
		System.out.println( "\n\ntype: \t\tSE or RR or BOTH ");
		System.out.println( "numCompetitors: number of players");
		System.out.println( "numMatches: \tnumber of matches per round of play");
		System.out.println( "\nSE = Single Elimination - RR = Round Robin - BOTH = SE and RR");
		System.out.println( "\nExamples:\njava GenerateCards SE 8 4");
	    System.out.println( "java GenerateCards RR 7 2");
	    System.out.println( "\n\nCard files will be placed in currentDir/data/cards/");
	    
	    System.exit(0);
	    		
		
	}
	*/
	
	GenerateCards( ){ //SecureRandom random ){
		//super(random);	
	}
	
	
	public static void combineDirectories(Vector<File> directories, File outputDirectory) throws IOException{
		File[] files = directories.get(0).listFiles();
		for(File refFile:files){
			Vector<File> childFiles = new Vector<File>();
			for(File dir:directories){
				File childFile = new File(dir.getAbsolutePath(),refFile.getName());
				childFiles.add(childFile);
			}
			File outputFile = new File(outputDirectory,refFile.getName());
			combineFiles(childFiles,outputFile);
		}
	}
	
	
	/**
	 * Combine several files of cards using Card.combine().
	 * @see Card#combine(Card)
	 * 
	 * @param input input files
	 * @param output output file
	 * @throws FileNotFoundException 
	 */
	public static void combineFiles(Vector<File> input, File output) throws IOException{
		Vector<BufferedReader> reader = new Vector<BufferedReader>();
	    for(File f:input){
	    	BufferedReader br = new BufferedReader(new FileReader(f));
	    	reader.add(br);
	    }
	    FileWriter fw=new FileWriter(output);
	    while(true){
	    	Card[] partialResult=null;
		    for(BufferedReader br:reader){
		    	String line = br.readLine();
		    	if (line==null){
		    		// This is the end of the method
		    	    fw.close();
		    	    for(BufferedReader br2:reader){
		    	    	br2.close();
		    	    }
		    		return;
		    	}
		    	Card[] nextCards = Card.toCardArray(line);
		    	if (partialResult==null){
		    		partialResult = nextCards;
		    	} else {
		    		assert(partialResult.length==nextCards.length);
		    		partialResult = Card.combine(partialResult,nextCards);
		    	}
		    }
		    String result = "";
		    for(Card c:partialResult){
		    	result += c;
		    }
		    fw.write(result+"\n");
	    }	    
	}

	/**
	 * Tests if a file is designed for a particular match type.
	 * @param filename
	 * @param numDealsPerMatch
	 * @param numCards
	 * @return
	 */
	public static boolean confirmOneFile(String filename, int numDealsPerMatch, int numCards){
		File f = new File(filename);
		
		if (!f.exists()){
			return false;
		}
		int countLines=0;
		try {
			BufferedReader br = new BufferedReader(new FileReader(f));
			while(true){
				String line = br.readLine();
				if (line==null){
					break;
				}
				Card[] cards = Card.toCardArray(line);
				if (cards.length!=numCards){
					return false;
				}
				boolean[] used = new boolean[Card.DECKSIZE];
				for(int i=0;i<used.length;i++){
					used[i]=false;
				}
				for(int i=0;i<cards.length;i++){
				  int index = cards[i].getIndexRankMajor();
				  if (used[index]){
					  return false;
				  }
				  used[index]=true;
				}
				countLines++;
			}
		} catch (IOException e) {
			return false;
		}
		
		return (countLines==numDealsPerMatch);
	}
			
	/**
	 * generate just one card file
	 */
	public static void generateOneFile(String filename, SecureRandom
	random, int numDealsPerMatch, int numCards){
		System.out.println("Creating "+filename+" with "+numDealsPerMatch+" hands with "+numCards+" cards.");
		try{
		// create the directory (if necessary)
		File file = new File(filename);
		File parent = file.getParentFile();
		if (parent!=null && !parent.exists()){
		  System.out.println("Creating directory "+parent);
		  parent.mkdirs();
		}
		if (file.exists()){
		  System.out.println(file + " already exists");
		} else {
		  System.out.println("Generating "+file);
		}
		// create the file
		BufferedWriter cardFile = new BufferedWriter(new FileWriter(filename));
		Card[] dealt; // for storing the cards
		cardFile = new BufferedWriter(new FileWriter(filename));

		// for each deal write the cards to file
		for (int q = 0; q < numDealsPerMatch; q++) {
			// get some cards
			dealt = Card.dealNewArray(random, numCards);
			// make the string
			String dealtString = "";
			for (int l = 0; l < numCards; l++) {
				dealtString += dealt[l];
			}
			dealtString += "\n";
			// write to file
			cardFile.write(dealtString);
		}

		// all done
		cardFile.close();
		} catch (IOException io){
		  System.err.println("I/O Exception generating cards");
		  io.printStackTrace();
		}
	}

	/*
	 * Creates all the necessary card files on disk, depending on the number of bots
	 * number of matches and number of hands
	 * 
	 * prefix, suffix, 
	 * 
	 */
	public static void GenerateRoundRobin( int numCompetitors, int numMatches ) {
		
		SecureRandom random = new SecureRandom();
		BufferedWriter cardFile;
		Card[] dealt; // for storing the cards
		String filename = "";
		String directory = "data/cards/";
		String prefix = "bankroll";
		String compprefix = "comp";
		String suffix = ".crd";
		//int numMatches = 3;
		int numDealsPerMatch = 1000;
		
		System.out.println( "About to generate RR cards for " + numCompetitors + " competitors and " + numMatches + " matches.");
		
		
		// create directory date/cards if data/cards DNE
    	//boolean success = 
		(new File(directory)).mkdirs(); // if dir exists, false, but this is OK

    	// for each possible match
		for (int i = 1; i <= numCompetitors; i++) {
			for (int j = i + 1; j <= numCompetitors; j++) {
				for (int k = 1; k <= numMatches; k++) {
					try {
						filename = directory + prefix + compprefix + i + compprefix + j
								+ "match" + k + suffix;

						// create the file
						cardFile = new BufferedWriter(new FileWriter(filename));

						// for each deal write the cards to file
						for (int q = 0; q < numDealsPerMatch; q++) {
							// get some cards
							dealt = Card.dealNewArray(random, 9);

							// make the string
							String dealtString = "";
							for (int l = 0; l < 9; l++) {
								dealtString += dealt[l];
							}
							dealtString += "\n";

							// write to file
							cardFile.write(dealtString);
						}

						// all done
						cardFile.close();

					} catch (IOException e) {
						System.err.println("Failed writing the generated cards to file. IOException");
					}
				}
			}
		}
		
		System.out.println( "Done Round Robin Card Generation");
		
	}
	
	/*
	 * Creates all the necessary card files on disk, depending on the number of bots
	 * number of matches and number of hands
	 * 
	 * 
	 * 
	 */
	public static void GenerateSingleElimination( int numCompetitors, int numMatches ) {
		
		SecureRandom random = new SecureRandom();
		BufferedWriter cardFile;
		Card[] dealt; // for storing the cards
		String filename = "";
		String directory = "data/cards/";
		String prefix = "seseries";
		String suffix = ".crd";
		
		//int numMatches = 3;
		int numSeries = numCompetitors -1;
		int numDealsPerMatch = 1000;
		
		System.out.println( "About to generate Single Elimination cards for " + numCompetitors + " competitors and " + numMatches + " matches.");
		
		
		// create directory date/cards if data/cards DNE
    	//boolean success = 
		(new File(directory)).mkdirs(); // if dir exists, false, but this is OK

    	// for each possible match
		for (int i = 1; i <= numSeries; i++) {
				for (int j = 1; j <= numMatches; j++) {
					try {
						filename = directory + prefix + i + "match" + j + suffix;

						// create the file
						cardFile = new BufferedWriter(new FileWriter(filename));

						// for each deal write the cards to file
						for (int q = 0; q < numDealsPerMatch; q++) {
							// get some cards
							dealt = Card.dealNewArray(random, 9);

							// make the string
							String dealtString = "";
							for (int l = 0; l < 9; l++) {
								dealtString += dealt[l];
							}
							dealtString += "\n";

							// write to file
							cardFile.write(dealtString);
						}

						// all done
						cardFile.close();

					} catch (IOException e) {
						System.err.println("Failed writing the generated cards to file. IOException");
					}
				}
			}
	
	    
		System.out.println( "Done Single Elimination Card Generation");
		
	}
	
	
	
}
