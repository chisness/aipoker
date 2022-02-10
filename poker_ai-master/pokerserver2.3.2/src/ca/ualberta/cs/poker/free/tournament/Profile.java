package ca.ualberta.cs.poker.free.tournament;

import java.util.Vector;
import java.io.*;
import java.net.InetAddress;
import java.util.StringTokenizer;

import ca.ualberta.cs.poker.free.alien.AlienAccount;
import ca.ualberta.cs.poker.free.alien.AlienNode;
import ca.ualberta.cs.poker.free.dynamics.MatchType;
import ca.ualberta.cs.poker.free.dynamics.LimitType;
import ca.ualberta.cs.poker.free.ringseat.RingPolicy;


/*
 * This is the profile of the competition to be run.
 * 
 * @author Martin Zinkevich
 * 
 */
public class Profile{

  /**
   * Machines found in the profile
   */
  public MachineRoom machines;

  /**
   * Tournament found in the profile
   */
  public Node node;

  /**
   * Throws a runtime exception if another token is not found
   */
  public static void checkTokens(StringTokenizer st, String str) throws IOException{
    if (!st.hasMoreTokens()){
      throw new IOException("Error: cannot parse line:"+str);
    }
  }
  /**
   * Returns the next line that is non-empty
   * after comments are removed.
   * The line when returned has the comments removed.
   */
  public static String nextLine(BufferedReader in) throws IOException{
    while(true){
      String str = in.readLine();
      if (str==null){
    	  return null;
      }
      int commentIndex = str.indexOf("#");
      if (commentIndex!=-1){
        str = str.substring(0,commentIndex);
      }
      int i=0;
      if (str.length()==0){
        continue;
      }
      while(Character.isWhitespace(str.charAt(i))){
        i++;
      }
      str = str.substring(i);
      if (str.length()>0){
        return str;
      }
    }
  }
        
  /**
   * Read a profile from a file.
   */
  public Profile(String file) throws IOException{
    BufferedReader in = new BufferedReader(new FileReader(file));
    machines = new MachineRoom();
    String str = nextLine(in);
    Vector<Node> nodes = new Vector<Node>();
    
    while(str != null){  
      if (str.startsWith("STATUS_FILE")) {
    	  // store in machineroom due to Node being an interface
    	  machines.statusFileLocation = str;
      } else if (str.startsWith("BEGIN_THREE_PLAYER_TOURNAMENT")) {
	        str = nextLine(in);
		StringTokenizer st = new StringTokenizer(str);
		checkTokens(st, str);
		String type = st.nextToken();
		assert(type.equals("ThreePlayer"));
		checkTokens(st, str);
		String rootSeriesName = st.nextToken();
		checkTokens(st, str);
		String rootCardFileName = st.nextToken();
		checkTokens(st, str);
		int numDuplicateMatchSets = Integer.parseInt(st.nextToken());
		checkTokens(st, str);
		InetAddress server = null;
		try {
			server = InetAddress.getByName(st.nextToken());
		} catch (java.net.UnknownHostException E) {
			throw new RuntimeException("Unknown host in " + str);
		}
		checkTokens(st,str);
		String ringPolicyFile = st.nextToken();
		
		Vector<BotInterface> bots = new Vector<BotInterface>();
		while (true) {
			str = nextLine(in);
			if (str.startsWith("END_TOURNAMENT")) {
				break;
			}
			bots.add(BotFactory.generateBot(str));
		}

		Vector<RingPolicy> globalPolicy = RingPolicy.read(ringPolicyFile);
		
		Node temp = null;
		for(int i=0;i<bots.size();i++){
			for (int j = i+1; j < bots.size(); j++) {
				for (int k = j+1; k < bots.size(); k++) {
					Vector<BotInterface> currentBots = new Vector<BotInterface>();
					currentBots.add(bots.get(i));
					currentBots.add(bots.get(j));
					currentBots.add(bots.get(k));
					temp = new RingSeries(currentBots, rootSeriesName, rootCardFileName, numDuplicateMatchSets,
				  server, globalPolicy,new MatchType(LimitType.LIMIT,false,0,1000));
					nodes.add( temp );
				}
			}
		}
		
      } else if (str.startsWith("BEGIN_TOURNAMENT")){
    	  // read in the potentially multiple tournaments
    	  node = generateCompetition(in);
		  nodes.add( node );
      } else { 
    	  machines.add(MachineFactory.generateMachine(str));
      }
      
      str = nextLine(in);
     }
    
    in.close();
    
    if ( nodes.size() > 1 ) {
    	// need a multi node
    	node = new MultiNode(nodes);
    }
    
    
  } 
  
  
  /**
   * There are several formats for the various types:<BR>
   * HeadsUpLimitRoundRobin  (see {@link #generateHeadsUpRoundRobin(BufferedReader,String)})<BR>
   * HeadsUpNoLimitRoundRobin (see {@link #generateHeadsUpRoundRobin(BufferedReader,String)})<BR>
   * Alien (see {@link #generateAlienNode(BufferedReader,String)})<BR>
   * RingSeries<BR>
   */
  public Node generateCompetition(BufferedReader in) throws IOException{
    String str = nextLine(in);
    if (str.startsWith("HeadsUpLimitRoundRobin")){
      return generateHeadsUpRoundRobin(in,str);
    } else if (str.startsWith("HeadsUpNoLimitRoundRobin")){
        return generateHeadsUpRoundRobin(in,str);
    } else if (str.startsWith("HeadsUpLimitOneVersusAll")){
    	return generateOneVersusAll(in,str);
    } else if (str.startsWith("HeadsUpNoLimitOneVersusAll")){
    	return generateOneVersusAll(in,str);    	
    } else if (str.startsWith("Alien")){
      return generateAlienNode(in,str);
    } else if (str.startsWith("RingSeries")){
    	return generateRingSeries(in, str);
    }
    throw new IOException("Unexpected tournament type in "+str);
  }
  
  
    /**
	 * Get a list of accounts, terminated with END_ACCOUNTS.
	 * 
	 * @see ca.ualberta.cs.poker.free.alien.AlienAccount#AlienAccount(String)
	 */
	public Vector<AlienAccount> getAccounts(BufferedReader in)
			throws IOException {
		Vector<AlienAccount> accounts = new Vector<AlienAccount>();
		while (true) {
			String str = nextLine(in);
			if (str.startsWith("END_ACCOUNTS")) {
				break;
			}
			accounts.add(new AlienAccount(str));
		}
		return accounts;
	}
  
    /**
	 * Get a list of bots, terminated with END_BOTS.
	 * 
	 * @see BotFactory#generateBot(String)
	 */
    public Vector<BotInterface> getBots(BufferedReader in) throws IOException {
		Vector<BotInterface> bots = new Vector<BotInterface>();
		while (true) {
			String str = nextLine(in);
			if (str.startsWith("END_BOTS")) {
				break;
			}
			BotInterface bot = BotFactory.generateBot(str);
			System.err.println(bot);
			bots.add(bot);
		}
		return bots;
	}
  
/**
 * {@literal AlienNode <teamTokens> <firstAgentNumber> <serverIP> <port>}
 * 
 * @param in
 *            the input stream to read from
 * @param str
 *            the most recently read line
 * @return the AlienNode
 * @throws IOException
 */
  public Node generateAlienNode(BufferedReader in, String str)
			throws IOException {
		StringTokenizer st = new StringTokenizer(str);
		checkTokens(st, str);
		if (!st.nextToken().equals("AlienNode")) {
			throw new IOException("Unexpected tournament type in " + str);
		}
		try{
		checkTokens(st, str);
		int teamTokens = Integer.parseInt(st.nextToken());
		checkTokens(st, str);
		int firstAgentNumber = Integer.parseInt(st.nextToken());
		checkTokens(st, str);
		InetAddress server = null;
		try {
			server = InetAddress.getByName(st.nextToken());
		} catch (java.net.UnknownHostException E) {
			throw new RuntimeException("Unknown host in " + str);
		}
		checkTokens(st, str);
		int port = Integer.parseInt(st.nextToken());

		AlienNode result = new AlienNode(teamTokens, firstAgentNumber, server,
				port);

		while (true) {
			str = nextLine(in);
			if (str.equals("END_TOURNAMENT")) {
				return result;
			}
			if (str.startsWith("BEGIN_BOTS")) {
				StringTokenizer st2 = new StringTokenizer(str);
				String type = "HEADSUPLIMIT2006";
				st2.nextToken();
				if (st2.hasMoreTokens()){
					type = st2.nextToken();
				}
				Vector<BotInterface> bots = getBots(in);
				for (BotInterface bot : bots) {
					result.addOpponent(bot,type);
				}
			} else if (str.equals("BEGIN_ACCOUNTS")) {
				Vector<AlienAccount> accounts = getAccounts(in);
				for (AlienAccount account : accounts) {
					result.addAccount(account);
				}
			}
		}
	} catch (NumberFormatException nfe){
		System.err.println("Expected AlienNode <teamTokens> <firstAgentNumber> <serverIP> <port>");
		System.err.println("Saw "+str);
		throw new RuntimeException("Format exception");
	}
  }
  
  /**
   * Format<BR>
   * {@code RingSeries <rootSeriesName> <rootCardFile> <numSets> <serverName> <ringPolicyFile>}<BR>
   * where:<BR>
   * rootSeriesName is the name of the series<BR>
   * rootCardFile is the name of the card files used<BR>
   * ringPolicyFile is the name of the file where the duplicate technique is stored<BR>
   * numSets is the number of different card files used.<BR>
   * 
   * @param in
   * @param str
   * @return
   * @throws IOException
   */
  public Node generateRingSeries(BufferedReader in, String str)
			throws IOException {
		StringTokenizer st = new StringTokenizer(str);
		checkTokens(st, str);
		String type = st.nextToken();
		assert(type.equals("RingSeries"));
		checkTokens(st, str);
		String rootSeriesName = st.nextToken();
		checkTokens(st, str);
		String rootCardFileName = st.nextToken();
		checkTokens(st, str);
		int numDuplicateMatchSets = Integer.parseInt(st.nextToken());
		checkTokens(st, str);
		InetAddress server = null;
		try {
			server = InetAddress.getByName(st.nextToken());
		} catch (java.net.UnknownHostException E) {
			throw new RuntimeException("Unknown host in " + str);
		}
		checkTokens(st,str);
		String ringPolicyFile = st.nextToken();
		
		Vector<BotInterface> bots = new Vector<BotInterface>();
		while (true) {
			str = nextLine(in);
			if (str.startsWith("END_TOURNAMENT")) {
				break;
			}
			bots.add(BotFactory.generateBot(str));
		}

		Vector<RingPolicy> globalPolicy = RingPolicy.read(ringPolicyFile);
			  
		return new RingSeries(bots, rootSeriesName, rootCardFileName, numDuplicateMatchSets,
				  server, globalPolicy,new MatchType(LimitType.LIMIT,false,0,1000));
	}

 
  public Node generateOneVersusAll(BufferedReader in, String str) throws IOException{
	    StringTokenizer st = new StringTokenizer(str);
	    checkTokens(st,str);
	    String type = st.nextToken();
	    boolean limitGame = (type.equals("HeadsUpLimitOneVersusAll"));
	    if (!(type.equals("HeadsUpLimitOneVersusAll")||type.equals("HeadsUpNoLimitOneVersusAll"))){
	      throw new IOException("Unexpected tournament type in "+str);
	    }
	    //checkTokens(st,str);
	    //WinnerDeterminationType wdType = WinnerDeterminationType.parse(st.nextToken());
	    checkTokens(st,str);
	    String rootSeriesName = st.nextToken();
	    checkTokens(st,str);
	    String rootCardFileName = st.nextToken();
	    checkTokens(st,str);
	    int numDuplicatePairs = Integer.parseInt(st.nextToken());
	    checkTokens(st,str);
	    InetAddress server=null;
	    try{
	      server = InetAddress.getByName(st.nextToken());
	    } catch(java.net.UnknownHostException E){
	      throw new RuntimeException("Unknown host in "+str);
	    }
	    checkTokens(st,str);
	    boolean reversed = st.nextToken().equalsIgnoreCase("REVERSED");
	    
	    
	    Vector<BotInterface> bots=new Vector<BotInterface>();
	    while(true){
	      str = nextLine(in);
	      if (str.startsWith("END_TOURNAMENT")){
	        break;
	      }
	      bots.add(BotFactory.generateBot(str));
	    }
	    MatchType info = limitGame ?
	    		new MatchType(LimitType.LIMIT,false,0,3000) :
	    		new MatchType(LimitType.DOYLE,false,0,3000);
	    
	    return new OneVersusAllTournament(
	    bots,
	    rootSeriesName,
	    rootCardFileName,
	    numDuplicatePairs,
	    server,info, reversed);
	  
  }
  public Node generateHeadsUpRoundRobin(BufferedReader in,String str) throws IOException{
    StringTokenizer st = new StringTokenizer(str);
    checkTokens(st,str);
    String type = st.nextToken();
    boolean limitGame = (type.equals("HeadsUpLimitRoundRobin"));
    if (!(type.equals("HeadsUpLimitRoundRobin")||type.equals("HeadsUpNoLimitRoundRobin"))){
      throw new IOException("Unexpected tournament type in "+str);
    }
    checkTokens(st,str);
    WinnerDeterminationType wdType = WinnerDeterminationType.parse(st.nextToken());
    checkTokens(st,str);
    String rootSeriesName = st.nextToken();
    checkTokens(st,str);
    String rootCardFileName = st.nextToken();
    checkTokens(st,str);
    int numDuplicatePairs = Integer.parseInt(st.nextToken());
    checkTokens(st,str);
    InetAddress server=null;
    try{
      server = InetAddress.getByName(st.nextToken());
    } catch(java.net.UnknownHostException E){
      throw new RuntimeException("Unknown host in "+str);
    }
    
    Vector<BotInterface> bots=new Vector<BotInterface>();
    while(true){
      str = nextLine(in);
      if (str.startsWith("END_TOURNAMENT")){
        break;
      }
      bots.add(BotFactory.generateBot(str));
    }
    MatchType info = limitGame ?
    		new MatchType(LimitType.LIMIT,false,0,3000) :
    		new MatchType(LimitType.DOYLE,false,0,3000);
    
    return new RoundRobinTournament(
    bots,
    rootSeriesName,
    rootCardFileName,
    numDuplicatePairs,
    server,info,wdType);

  }
  
  /**
   * Get a Forge object with the machines and tournament from
   * the profile.
   */
  public Forge getForge(){
    return new Forge(machines,node);
  }
  
  
}
