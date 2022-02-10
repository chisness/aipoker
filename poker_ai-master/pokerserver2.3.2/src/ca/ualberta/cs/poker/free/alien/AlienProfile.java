package ca.ualberta.cs.poker.free.alien;
import ca.ualberta.cs.poker.free.tournament.*;
import java.io.*;
import java.util.*;


/**
 * A class to parse the data from a profile on the client side.
 * @author Martin Zinkevich
 */
public class AlienProfile {
    /**
     * The name of the user.
     */
    public String username;

    /**
     * The password
     */
    public String password;

    
    /**
     * The bot list
     */
    public Vector<BotInterface> bots;

    /**
     * The match list
     */
    public Vector<ClientMatch> matches;
    
    /**
     * list of opponents the player can play against
     */
    public Vector<String> opponents;
    
    /**
     * A mapping from descriptions to machines.
     */
	public Hashtable<String, MachineInterface> machines;

	public Vector<String> botDescriptions;

	/**
	 * Server address (of AlienNode)
	 */
	public String addr;
	
	/**
	 * port (of AlienNode)
	 */
    public int port;
    
    /** 
     * Creates a new instance of AlienClient from a file
     */
    public AlienProfile(String filename) throws IOException{
        Reader r = new FileReader(filename);
    	read(r);
    	r.close();
    }

    public void readBots(BufferedReader br) throws IOException{
      while(true){
        String nextLine = Profile.nextLine(br);
        if (nextLine.equals("END_BOTS")){
	  return;
	}
	bots.add(BotFactory.generateBot(nextLine));
	botDescriptions.add(nextLine);
      }
    }

    
    /**
     * Reads matches from reader until encounters END_MATCHES
     * @see ClientMatch#ClientMatch(String)
     * @param br
     * @throws IOException
     */
    public void readMatches(BufferedReader br) throws IOException{
      int repeatFactor = 1;
      while(true){
        String nextLine = Profile.nextLine(br);
	if (nextLine.equals("END_MATCHES")){
	  return;
	}
	if (nextLine.startsWith("REPEAT:")){
	  String repeatString = 
	  nextLine.substring("REPEAT:".length());
	  repeatFactor = Integer.parseInt(repeatString);
	  continue;
	}
	if (nextLine.startsWith("RING")) {
		ClientMatchRing match = new ClientMatchRing(nextLine);
		for(int i=0;i<repeatFactor;i++){
	  		matches.add(new ClientMatchRing(match,i));
		}
	} else {
	ClientMatch match = new ClientMatch(nextLine);
	for(int i=0;i<repeatFactor;i++){
	  matches.add(new ClientMatch(match,i));
	}
      }
    }
}

    /**
     * Reads the machine descriptions and puts them in a hash table.
     * @param br the stream to read from
     * @throws IOException
     */
    public void readMachines(BufferedReader br) throws IOException{
		while (true) {
			String nextLine = Profile.nextLine(br);
			if (nextLine.equals("END_MACHINES")) {
				return;
			}
			machines.put(nextLine,MachineFactory.generateMachine(nextLine));
		}
	}

    public void init(){
    	matches = new Vector<ClientMatch>();
        bots = new Vector<BotInterface>();
        botDescriptions = new Vector<String>();
        machines = new Hashtable<String, MachineInterface>();
        username = null;
        password = null;
        opponents = new Vector<String>();
    }

    public void readPassword(String str){
      password = str.substring("PASSWORD:".length());
    }

    public void readUsername(String str){
      username = str.substring("USERNAME:".length());
    }
    
    public void readServerIP(String str){
    	addr = str.substring("SERVERIP:".length());
    }
    
    public void readPort(String str){
    	port = Integer.parseInt(str.substring("PORT:".length()));
    }
    /**
     * Reads a profile. A profile is of the form:<BR>
     * USERNAME:BOB<BR>
     * PASSWORD:BOBISSNEAKY<BR>
     * BEGIN_BOTS<BR>
     * ...bots go here...<BR>
     * END_BOTS<BR>
     * BEGIN_MACHINES<BR>
     * ...machines go here...<BR>
     * END_MACHINES<BR>
     * BEGIN_MATCHES<BR>
     * ...matches go here...<BR>
     * END_MATCHES
     * @see ca.ualberta.cs.poker.free.tournament.MachineFactory#generateMachine(String)
     * @see ca.ualberta.cs.poker.free.tournament.BotFactory#generateBot(String)
     */
    public void read(Reader r) throws IOException{
      BufferedReader br = new BufferedReader(r);
      init();
      while(true){
        String nextLine = Profile.nextLine(br);
	if (nextLine==null){
          return;
	} else if (nextLine.startsWith("BEGIN_MACHINES")){
		readMachines(br);
	} else if (nextLine.startsWith("USERNAME:")){
	  readUsername(nextLine);
	} else if (nextLine.startsWith("PASSWORD:")){
	  readPassword(nextLine);
	} else if (nextLine.equals("BEGIN_BOTS")){
	  readBots(br);
	} else if (nextLine.equals("BEGIN_MATCHES")){
	  readMatches(br);
	} else if (nextLine.startsWith("SERVERIP:")){
		readServerIP(nextLine);
	} else if (nextLine.startsWith("PORT:")){
		readPort(nextLine);
	} else if (nextLine.equals("BEGIN_OPPONENTS")){
		readOpponents(br);
	} else {
	  System.err.println("Error parsing profile for AgentClient");
	  System.err.println("Unexpected line:"+nextLine);
	}
    }
    }

    /**
     * Read the opponents from the profile
     * 
     * @param br
     * @throws IOException
     */
	public void readOpponents(BufferedReader br) throws IOException {
    	while(true){
          String nextLine = Profile.nextLine(br);
          if (nextLine.equals("END_OPPONENTS")){
        	  return;
          }
          
          opponents.add( nextLine );
    	}
	}
    
    /**
     * Returns the next line that is non-empty
     * after comments are removed.
     * The line when returned has the comments removed.
     */
    /*
    public static String readLine(BufferedReader in) throws IOException{
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
    */
}
