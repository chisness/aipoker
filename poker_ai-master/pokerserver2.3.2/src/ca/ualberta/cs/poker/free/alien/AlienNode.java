package ca.ualberta.cs.poker.free.alien;
import ca.ualberta.cs.poker.free.dynamics.LimitType;
import ca.ualberta.cs.poker.free.dynamics.MatchType;
import ca.ualberta.cs.poker.free.tournament.*;



import java.text.DateFormat;
import java.util.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.net.*;
import java.security.SecureRandom;

/**
 * A node to feed the matches received by AlienAgent from AlienClient
 * into Forge.
 * 
 * Also, handles when AlienAgents are destroyed, removing all matches (by informing
 * the forge they are complete), and then removing all machines.
 * Also, has a function to generate unique names agents.
 */
public class AlienNode implements Node, Runnable {
	
	/**
	 * Need a pointer to the library to add/remove machines.
	 */
	MachineLibrary library;
	
	/**
	 * Queue of matches from client
	 */
	Vector<MatchInterface> matchQueue;
	
	/**
	 * Queue of machines from client
	 */
	Vector<MachineInterface> machineQueue;

	/**
	 * List of matches that need to be destroyed
	 */
	Vector<MatchInterface> matchWarrant;

	/**
	 * List of machines that need to be destroyed
	 */
	Vector<MachineInterface> machineWarrant;

	/**
	 * The random object for this node.
	 */
	SecureRandom random;

	/**
	 * Number of the next agent.
	 * Initialized from a profile.
	 */
	int agentNumber;

	/**
	 * The address of the poker server
	 */
	InetAddress addr;

	/**
	 * Alien accounts (indexed by username)
	 * Accounts contain static info, such as
	 * name, password, team, and e-mail.
	 */
	Hashtable<String, AlienAccount> accounts;

	/**
	 * Agents of the aliens. 
	 * An agent is created when a connection
	 * is made to the server. The agent creates
	 * Matches and adds them to the server.
	 */
	Vector<AlienAgent> agents;

	/**
	 * Map of a bots name, to its interface
	 * name -> BotInterface
	 */
	Hashtable<String, BotInterface> opponentBotNames;
	
	/**
	 * Map a BotInterface to a MatchType
	 * BotInterface -> MatchType
	 */
	Hashtable<BotInterface, MatchType> opponentBotMatchTypes;
	
	/**
	 * Retreive the account, or return null if the
	 * password is incorrect.
	 */
	public synchronized AlienAccount testLogin(String username, String password) {
		AlienAccount account = accounts.get(username);
		if (account == null) {
			return null;
		}
		if (!account.password.equals(password)) {
			return null;
		}
		return account;
	}

	/**
	 * Returns an account. This is used to change the password.
	 * @param username
	 * @return
	 */
	public synchronized AlienAccount getAccount(String username){
		return accounts.get(username);
	}
	
	

	/**
	 * The budget allocated to each team (in machines)
	 */
	public int teamTokens;

	/**
	 * The port for clients to connect to.
	 */
	public int port;
	
	/**
	 * True if superuser has issued a shutdown command.
	 */
	private boolean shutdownImminent;

	public AlienNode(int teamTokens, int firstAgentNumber, InetAddress serverAddress, int port) {
		random = new SecureRandom();
		matchQueue = new Vector<MatchInterface>();
		machineQueue = new Vector<MachineInterface>();
		matchWarrant = new Vector<MatchInterface>();
		machineWarrant = new Vector<MachineInterface>();
		agentNumber = firstAgentNumber;
		//opponentMatchBots = new Hashtable<MatchType, BotInterface>();
		//opponentMatchTypes = new Hashtable<String, MatchType>();
		opponentBotNames = new Hashtable<String, BotInterface>();
		opponentBotMatchTypes = new Hashtable<BotInterface, MatchType>();
		agents = new Vector<AlienAgent>();
		accounts = new Hashtable<String,AlienAccount>();
		addr = serverAddress;
	    this.port = port;
		this.teamTokens = teamTokens;
		this.shutdownImminent = false;
	}

	
	/**
	 * Create a MachineLibrary from a MachineRoom
	 */
	public MachineLibrary createLibrary(MachineRoom room){
		this.library = new MachineLibrary(room);
		Set<String> teamNames = new HashSet<String>();
		for(AlienAccount account:accounts.values()){
			teamNames.add(account.team);
		}
		for(String teamName:teamNames){
			library.addTeam(teamName,teamTokens);
		}
		return library;
	}
	/**
	 * Get an opponent from a name.
	 * If no opponent has the name, returns null
	 * @param name the name of the bot, (what is returned from getName())
	 */
	public BotInterface getOpponent(String name) {
		return opponentBotNames.get( name ); 
	}

	/**
	 * Add an opponent. Update the hash from a bot's name -> MatchType so
	 * we know what type of match and hands to play
	 * @param bot the opponent to be added.
	 */
	public void addOpponent(BotInterface bot, String type) {
		opponentBotMatchTypes.put( bot, getMatchTypeFromString(type));
		opponentBotNames.put( bot.getName(), bot);
	}

	/** 
	 * Test if all of the matches for this node have been
	 * completed.
	 */
	public boolean isComplete() {
		return false;
	}

	/**
	 * Load/Remove matches/machines into the forge/library.
	 * 
	 * Require AlienAgent token, and AlienNode token
	 * 
	 */
	public void load(Forge w) {
		Vector<AlienAgent> agentCopy = getAgents();
		if (shutdownImminent){
			for(AlienAgent agent:agentCopy){
				agent.suicide();
			}
		}
		
		internalLoad(w);
		
		agentCopy = getAgents();
		for(AlienAgent aa:agentCopy){
			aa.testCompletedMatches();
		}
		
	}
	/*
	 * Requires ALienNode token
	 */
	public synchronized Vector<AlienAgent> getAgents() { 
		return new Vector<AlienAgent> ( agents );
	}
	
	
	/**
	 * Load/Remove matches/machines into the forge/library.
	 * Hogs the token, but pushing stuff to the forge is crucial.
	 * Plus, must terminate after queues and warrants are emptied.
	 * 
	 * Requires AlienNode token
	 * 
	 */	
	private synchronized void internalLoad(Forge w){
		System.err.println("AlienNode.internalLoad()");
		Forge.showDate(System.err);
		
		matchQueue.removeAll(matchWarrant);
		machineQueue.removeAll(machineWarrant);
		System.err.println("Removing matches");
		for(MatchInterface match:matchWarrant){
			System.err.println("Removing match "+match);
			w.remove(match);
		}
		System.err.println("Clearing warrants");
		matchWarrant.clear();
		
		// NOTE: it is assumed that the machineWarrant machines are 
        // not being used for anything right now.
		for(MachineInterface machine:machineWarrant){
			System.out.println("Removing machine "+machine);
			library.remove(machine);
		}
		System.err.println("Machines removed: clearing warrants");
		machineWarrant.clear();
		
		if (!shutdownImminent){
		for(MatchInterface match:matchQueue){
			System.err.println("Queueing match "+match);
			w.add(match);
		}
		matchQueue.clear();
		
		for(MachineInterface machine:machineQueue){
			library.add(machine);
		}
		machineQueue.clear();
		
		
		
		}
		if (shutdownImminent){
			System.exit(0);
		}
	}

	/**
	 * push a new match onto the queue.
	 * 
	 * Requires AlienNode token
	 */
	public synchronized void pushBack(MatchInterface m) {
		matchQueue.add(m);
	}

	public synchronized void startShutdown(){
		shutdownImminent = true;
	}
	/**
	 * Add an agent to the node.
	 * 
	 * Requires AlienNode token
	 */
	public synchronized void addAgent(AlienAgent aa) {
		agents.add(aa);
	}
	
	/**
	 * Remove an agent from the node.
	 * 
	 * Requires AlienNode token
	 */
	public synchronized void removeAgent(AlienAgent aa) {
		agents.remove(aa);
		machineWarrant.addAll(aa.queuedMachines);
		matchWarrant.addAll(aa.queuedMatches);
		//throw new RuntimeException("Not implemented");
	}
	
	/**
	 * Generate a new server socket, and
	 * create new AlienAgents as connections are made.
	 * See ca.ualberta.cs.poker.free.server.PokerServer#run() for an example.
	 * 
	 * Requires AlienNode token
	 */
	public void run() {
		ServerSocket socket = null;
        try{
            socket = new ServerSocket(port);
            System.err.println("Socket:"+socket);
        }catch (IOException io){
            System.err.println("CRITICAL ERROR: CANNOT START SERVER");
            System.exit(0);
            return;
        }
        while(true){
              try{
            	  System.err.println("A client has attempted to connect");
              Socket childSocket = socket.accept();
              AlienAgent agent = new AlienAgent(childSocket, this);
              addAgent(agent);
              } catch (IOException io){
                System.err.println("Agent failed to login");
              }
              
              try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				try{
				socket.close();
				} catch (IOException io){
					io.printStackTrace(System.err);
				}
				return;
			}
        }
	}

	/**
	 * Gets a new name for a new agent
	 * @return the name
	 */
	public String getNewAgentName(){
		agentNumber++;
		return "session"+agentNumber;
	}
	
	/**
	 * Get the winner of this node.
	 * Does not make sense in this context.
	 * @return null
	 */
	public Vector<BotInterface> getWinners() {
		return new Vector<BotInterface>();
	}

	/**
	 * Does nothing.
	 * Cards are generated for this node when the match is queued, not here.
	 */
	public void generateCardFiles(SecureRandom random) {
		// Left intentionally blank
	}

	/**
	 * There is no way to test for card files for matches that we do not
	 * yet know will happen.
	 */
	public boolean confirmCardFiles(){
		return true;
	}
	/**
	 * Generate the card file for a match
	 * 
	 * Requires AlienNode token
	 */
	public synchronized boolean generateCardFile(MatchInterface match) {
		match.generateCardFile(random);
		return match.confirmCardFile();
	}

	/**
	 * Returns the match server.
	 */
	public InetAddress getServerIP() {
		return addr;
	}
	
	/**
	 * Slates a match to be terminated by putting out a warrant.
	 * The match will be terminated on the next load() operation.
	 * @param match the match to be terminated
	 * 
	 * Requires AlienNode token
	 */
	public synchronized void killMatch(MatchInterface match) {
		matchWarrant.add(match);
	}

	/**
	 * Slates a machine to be terminated by putting out a warrant.
	 * The machine will be terminated on the next load() operation.
	 * @param machine the machine to be terminated
	 */
	public synchronized void killMachine(MachineInterface machine) {
        machineWarrant.add(machine);
	}

	/**
	 * Shows the statistics of this node.
	 * Actually does nothing.
	 */
	public void showStatistics() {
		// Intentionally left blank.
	}

	/**
	 * Adds a machine to the MachineLibrary.
	 * Requires AlienNode token
	 */
	public synchronized void add(AlienMachine machine) {
		machineQueue.add(machine);
	}
	
	/**
	 * Usage:{@code AlienNode <profile>}
	 * @param args
	 */
	public static void main(String[] args) throws IOException{
		TarAndWeb.initStatic();
		Profile p = new Profile(args[0]);
		if (p.node instanceof AlienNode){
		  AlienNode n = (AlienNode)(p.node);
		  MachineLibrary ml = n.createLibrary(p.machines);
		  Forge f = new Forge(ml,n);
		  Thread alienThread = new Thread(n);
		  alienThread.start();
		  f.runTournament();
		} else {
			System.err.println("No AlienNode in "+args[0]);
		}
	}

/**
 * Adds an account to the node
 * @param account
 */
	public synchronized void addAccount(AlienAccount account) {
		if (library!=null&&!library.teamHasBudget(account.team)){
			library.addTeam(account.team, 1);
		}

		accounts.put(account.username, account);
		backup();
		
		// For result access
		// 1. create a directory for the results to be created
		// 2. add the information to the .htpasswd file
		// 3. create the .htaccess which will allow only this user to access
		File directoryFile = new File("/local/data/web/htdocs/" + account.username);
		directoryFile.mkdirs();
		
		String command = "/local/data/web/bin/htpasswd -m -b /local/data/.htpasswd " + account.username + " " + account.password;
		
		// this will place the resulting tarfile in the 
		Vector<Thread> looseThreads=new Vector<Thread>();
		try{
			    Process p = Runtime.getRuntime().exec(command);
			    StreamConnect sc = new StreamConnect( p.getInputStream(),System.out);
			    Thread tsc = new Thread(sc);
			    tsc.start();
			    looseThreads.add(tsc);
			    StreamConnect scerr = new StreamConnect(p.getErrorStream(),System.err);
			    Thread tscerr = new Thread(scerr);
			    tscerr.start();
			    looseThreads.add(tscerr);
			    p.waitFor();
			    
			  } catch (InterruptedException e){
				  // TODO Do something more intelligent here
			  } catch (IOException e) {
				e.printStackTrace();
			}
		for(Thread t:looseThreads){
			t.interrupt();
		}
		
		// here is the .htaccess file, limiting access to this user
		try {
			
			Writer writer = new FileWriter("/local/data/web/htdocs/" + account.username + "/.htaccess");
			writer.write( "AuthUserFile /local/data/.htpasswd" + "\n");
			writer.write( "AuthName EnterPassword" + "\n");
			writer.write( "AuthType Basic" + "\n");
			writer.write( "require user " + account.username + "\n");
			writer.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.err.println( "Failed writing .htaccess for new account");
			e.printStackTrace();
		}
		
	}

	public void backup(){
		try{
			Writer writer = new FileWriter("accountbackup.txt");
			writeAccounts(writer);
			writer.close();
			} catch (IOException io){
				System.err.println("Error creating new account listing");
				io.printStackTrace(System.err);
			}

	}
	public void writeAccounts(Writer writer) throws IOException{
		DateFormat df = DateFormat.getDateInstance();
    	DateFormat tf = DateFormat.getTimeInstance();
    	Date d = new Date();
    	writer.write(df.format(d)+" "+tf.format(d)+"\n");

		writer.write("BEGIN_ACCOUNTS\n");
		for(AlienAccount account:accounts.values()){
			writer.write(account.toString()+"\n");
		}
		writer.write("END_ACCOUNTS\n");
	}
	/**
	 * @param accountName the account to be changed
	 * @param password the new password
	 * 
	 * Requires AlienNode token
	 */
public synchronized void changePassword(String accountName, String password) {
	AlienAccount account = getAccount(accountName);
	account.password=password;
	backup();
	
	// change the .htpasswd file as well
	String command = "/local/data/web/bin/htpasswd -m -b /local/data/.htpasswd " + account.username + " " + account.password;
	
	// this will place the resulting tarfile in the 
	Vector<Thread> looseThreads=new Vector<Thread>();
	try{
		    Process p = Runtime.getRuntime().exec(command);
		    StreamConnect sc = new StreamConnect( p.getInputStream(),System.out);
		    Thread tsc = new Thread(sc);
		    tsc.start();
		    looseThreads.add(tsc);
		    StreamConnect scerr = new StreamConnect(p.getErrorStream(),System.err);
		    Thread tscerr = new Thread(scerr);
		    tscerr.start();
		    looseThreads.add(tscerr);
		    p.waitFor();
		    
		  } catch (InterruptedException e){
		  } catch (IOException e) {
			e.printStackTrace();
		}
	for(Thread t:looseThreads){
		t.interrupt();
	}
	
}

	/**
	 * Return the associated MatchType
	 * 
	 * @param opponentBot
	 * @return
	 */
	public MatchType getOpponentMatchType(BotInterface opponentBot) {
		return opponentBotMatchTypes.get( opponentBot );	
	}
	
	
	
	public MatchType getMatchTypeFromString(String typeName){
		if (typeName.equals("HEADSUPLIMIT2006")){
			MatchType info = new MatchType(LimitType.LIMIT,false,0,1000);
		    info.timePerHand = 60000;
            info.chessClock = false;
            return info;
		} else if (typeName.equals("HEADSUPLIMIT2007")){
			MatchType info = new MatchType(LimitType.LIMIT,false,0,3000);
		    info.timePerHand = 60000;
            info.chessClock = true;
            return info;
		} else if (typeName.equals("HEADSUPLIMIT2009")){
			MatchType info = new MatchType(LimitType.LIMIT,false,0,3000);
		    info.timePerHand = 60000;
            info.chessClock = true;
            return info;
		} else if (typeName.equals("HEADSUPNOLIMIT2009")){
			MatchType info = new MatchType(LimitType.DOYLE,false,0,3000);
		    info.timePerHand = 60000;
            info.chessClock = true;
            return info;
		} else if (typeName.equals("RING2009")){
			MatchType info = new MatchType(LimitType.LIMIT,false,0,1000);
		    info.timePerHand = 60000;
            info.chessClock = true;
            return info;
		} else {
			assert(typeName.equals("HEADSUPNOLIMIT2007"));
			MatchType info = new MatchType(LimitType.DOYLE,false,0,1000);
		    info.timePerHand = 60000;
            info.chessClock = true;
            return info;
		}

	}


}
