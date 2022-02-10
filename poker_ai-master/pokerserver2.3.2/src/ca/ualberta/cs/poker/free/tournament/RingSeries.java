package ca.ualberta.cs.poker.free.tournament;

import java.net.InetAddress;
import java.util.Vector;
import java.security.SecureRandom;
import java.io.*;

import ca.ualberta.cs.poker.free.dynamics.MatchType;
import ca.ualberta.cs.poker.free.ringseat.RingPolicy;

/**
 * Implements a Ring series for a ring limit game
 * @author maz
 *
 */
public class RingSeries extends AbstractSeries {
	  MatchType info;
	  /**
	   * The root of all match names for this series
	   */
	  String rootMatchName;

	  /**
	   * The root of all card file names for this series.
	   */
	  String rootCardFileName;

	  /**
	   * The number of duplicate match sets.
	   */
	  int numDuplicateMatchSets;

	  /**
	   * The bots for this series.
	   */
	  Vector<BotInterface> bots;

	  /**
	   * The server IP.
	   */
	  InetAddress server;
	  
	  /**
	   * Random permutations, one for each match set.
	   * Length is the (@code numDuplicateMatchSets}
	   * numPlayers is {@code bots.size()}
	   */
	  RingPolicy randomPermutations;
	  
	  /**
	   * File in cards directory with the different
	   * random permutations.
	   */
	  String randomPermutationsFilename;
	  
	  /**
	   * A duplicate match set policy, drawn from the global policy.
	   */
	  RingPolicy duplicateMatchSetPolicy;
	  
	  /**
	   * Construct a new series, given the bots and the root names,
	   * the number of duplicate pairs, and the server IP.
	   */
	public RingSeries(Vector<BotInterface> bots,
	  String rootMatchName, String rootCardFileName, int
	  numDuplicateMatchSets,InetAddress server, Vector<RingPolicy> globalPolicy, MatchType info) throws IOException{
	  this.bots = bots;
      this.rootMatchName = rootMatchName;
      this.rootCardFileName = rootCardFileName;
	  this.numDuplicateMatchSets = numDuplicateMatchSets;
	  this.server = server;
	  this.info = info;
	  randomPermutationsFilename = "data/cards/"+rootCardFileName+".rng";
	  loadRandomPermutations();
	  duplicateMatchSetPolicy = RingPolicy.selectPolicy(globalPolicy,bots);
	}
	
	public void loadRandomPermutations() throws IOException{
		File file = new File(randomPermutationsFilename);
		if (file.exists()){
			try{
			randomPermutations = new RingPolicy(new BufferedReader(new FileReader(file)));
			} catch (FileNotFoundException fnf){
				fnf.printStackTrace(System.err);
				throw new RuntimeException("Strange disappearance of file "+file);
				
			}
		}
		
	}
	
	/**
	 * Generates duplicate match sets.
	 * Note that 
	 */
	public Vector<Vector<MatchInterface>> getMatchSets() {
		if (randomPermutations==null){
			return null;
		}
		Vector<Vector<MatchInterface> > result = new Vector<Vector<MatchInterface> >();
		Vector<Vector<BotInterface> > permutedBots = randomPermutations.mapPlayersToSeats(bots);
		for(int i=0;i<permutedBots.size();i++){
			String rootMatchSetName = rootMatchName + i;
			String matchSetCardFile = rootCardFileName + i;
			Vector<Vector<BotInterface> > duplicateSetBots = 
					duplicateMatchSetPolicy.mapPlayersToSeats(permutedBots.get(i));
			Vector<MatchInterface> duplicateMatchSet = new Vector<MatchInterface>();
			for(Vector<BotInterface> matchBots:duplicateSetBots){
				String matchName = rootMatchSetName;
				for(BotInterface bot:matchBots){
					matchName += ("." + bot.getName());
				}
				duplicateMatchSet.add(new RingLimitMatch(matchBots,matchSetCardFile,server,matchName,info));
			}
			result.add(duplicateMatchSet);
		}

		return result;
	}

	@Override
	public Vector<BotInterface> getBots() {
		return bots;
	}
	
	@Override
	public void generateCardFiles(SecureRandom random){
		// TODO Make the new permutation file here
		File file = new File(randomPermutationsFilename);
		if (!file.exists()){
			randomPermutations = new RingPolicy(bots.size(),numDuplicateMatchSets,random);
			try{
			randomPermutations.write(new FileWriter(file));
			System.err.println("Create random permutations:");
			} catch (IOException io){
				System.err.println("Error writing file "+randomPermutationsFilename);
				io.printStackTrace(System.err);
			}
		}
		super.generateCardFiles(random);
	}

	@Override
	public boolean confirmCardFiles(){
		// TODO Test to make sure the permutation files are there
		return super.confirmCardFiles();
	}

}
