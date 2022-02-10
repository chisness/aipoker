// AlienClient.java
package ca.ualberta.cs.poker.free.alien;
import ca.ualberta.cs.poker.free.tournament.*;
import ca.ualberta.cs.poker.free.server.*;

import java.net.*;
import java.io.*;
import java.util.*;


/**
 * The client side code that loads a profile,
 * connects to a server, and starts bots.
 * Sends messages to the server and
 * receives messages from the server.
 * Messages received:<BR>
 * CLEANMACHINE:&lt;description&gt;<BR>
 * ASSIGNMACHINE:&lt;description&gt; (always sent before ASSIGNBOT)<BR>
 * ASSIGNBOT:&lt;name&gt;:&lt;serverIP&gt;:&lt;port&gt;(always sent after ASSIGNMACHINE)<BR>
 * MATCHSTARTED:&lt;matchname&gt;:<BR>
 * MATCHCOMPLETE:&lt;matchname&gt;<BR>
 * MATCHTERMINATE:&lt;matchname&gt;<BR>
 * ERROR:&lt;message&gt;<BR>
 * LOGINSUCCESS<BR>
 * Messages sent:<BR>
 * LOGIN:&lt;username&gt;:&lt;password&gt;<BR>
 * CREATEMACHINE:&lt;description&gt;<BR>
 * CREATEBOT:&lt;description&gt;<BR>
 * MATCHREQUEST:&lt;gametype&gt;:&lt;alienBotName&gt;:&lt;opponentBotName&gt;<BR>
 * MATCHTERMINATE:&lt;matchname&gt;<BR>
 * LOGOUT<BR>
 * TODO: Fix so the bot is far more patient
 * @author Martin Zinkevich
 */
public class AlienClient extends TimedSocket implements Runnable{
	AlienClientListener listener;
	
    /**
     * The machine that is about to be assigned.
     */
    MachineInterface currentMachine;

    /**
     * A set of machines indexed by descriptions
     */
    Hashtable<String,MachineInterface> machines;
    
    Vector<MachineInterface> runningMachines;
    
    /**
     * The name of the user.
     */
    public String username;

    /**
     * The password of the user.
     */
    public String password;

    /**
     * Client matches
     */
    Vector<ClientMatch> matches;

    Vector<ClientMatch> matchesStarted;
    Vector<String> completedMatchStrings;
    
    /**
     * Bots available locally
     */
    Hashtable <String, BotInterface> bots;


    
    /** 
     * Creates an empty instance of AlienClient
     */    
    public AlienClient(){
    	currentMachine=null;
        machines=new Hashtable<String,MachineInterface>();
        runningMachines=new Vector<MachineInterface>();
        username=null;
        password=null;
        matches=new Vector<ClientMatch>();
        matchesStarted = new Vector<ClientMatch>();
        completedMatchStrings=new Vector<String>();   
        bots = new Hashtable <String, BotInterface>();
    }
    
    
    
    /**
     * Connects to the server at the given IP address and port number.
     */
    public void connect(InetAddress iaddr, int port) throws IOException, SocketException{
        Socket s = new Socket(iaddr,port);
        setSocket(s);
        open();
    }
    

    public void addBots(Vector<BotInterface> bots) throws TimeoutException{
    	for(BotInterface bot:bots){
    		addBot(bot);
    	}
    }
    
    public void addBot(BotInterface bot) throws TimeoutException{
    	this.sendCreateBot(bot.toString());
    	this.bots.put(bot.getName(), bot);
	System.out.println("Added " + bot.getName() + " to the hash in AlienClient");
    }

    
    public void addMatches(Vector<ClientMatch> matches) throws TimeoutException{
    	for(ClientMatch match:matches){
    		addMatch(match);
    	}
    }
    
    public void addMatch(ClientMatch match) throws TimeoutException{
    	this.sendMatchRequest(match);
    	matches.add(match);
    }
    
    public void addMachines(Hashtable<String,MachineInterface> machines) throws TimeoutException{
    	for(String description:machines.keySet()){
    		addMachine(machines.get(description),description);
    	}
    }
    
    public void addMachine(MachineInterface machine, String description) throws TimeoutException{
    	this.sendCreateMachine(description);
    	machines.put(description, machine);
    }
    /**
     * Initialize account.
     * Send a LOGIN message.
     * TODO: handle the error message.
     */
    public boolean login() throws TimeoutException, InterruptedException{
      sendMessage("LOGIN:"+username+":"+password);
      String response = receiveMessage();
      if (response.startsWith("SUCCESS")){
        return true;
      }
      System.err.println(response);
      return false;
    }


    /**
     * Wait a day to receive a message.
     */
    public String receiveMessage() throws TimeoutException, InterruptedException{
      setTimeRemaining(24*60*60*1000);
      String result = super.receiveMessage();
      System.err.println("Received message:"+result);
      return result;
    }
    
    public void sendMessage(String str) throws TimeoutException{
      // Setting this time here messes up RECEIVING messages.
      //setTimeRemaining(30000);
      System.err.println("Sending message:"+str);
      super.sendMessage(str);
    }

    
    /**
     * Send a match to the server.<BR>
     * MATCHREQUEST:&lt;gametype&gt;:&lt;matchname&gt;:&lt;alienbot&gt;:&lt;opponent&gt;<BR>
     * @see AlienAgent#processMatchRequestMessage(String)
     * @param match the match to send
     * @throws TimeoutException 
     */
    public void sendMatchRequest(ClientMatch match) throws TimeoutException{
    	String message = match.matchRequest();
    	System.err.println("Sending message "+message);
    	sendMessage(message);
    }
    
    /**
     * Assign a machine to a bot.
     * Initializes currentMachine to the machine sent.<BR>
     * ASSIGNMACHINE:&lt;description&gt;
     * @see AlienAgent#sendAssignMachine(String)
     * @param message the string to parse
     */
    public void processAssignMachineMessage(String message){
    	String description = message.substring("ASSIGNMACHINE:".length());
    	currentMachine = machines.get(description);
    	if (currentMachine==null){
    		System.err.println("Unknown machine received in message "+message);
    		System.exit(0);
    	}
    }
    
    /**
     * Clean the machine that is sent.
     * CLEANMACHINE:&lt;description&gt;
     * @see AlienAgent#sendCleanMachine(String)
     * @param message the string to parse
     */
    public void processCleanMachineMessage(String message){
    	
    	String description = message.substring("CLEANMACHINE:".length());
    	MachineInterface machine = machines.get(description);
    	if (machine==null){
    		System.err.println("Unknown machine received in message "+message);
    		System.exit(0);
    	}
    	machine.clean();
    	runningMachines.remove(machine);
    }
    
    /**
     * Starts a bot on the current machine and connects it to the server IP and port.
     * ASSIGNBOT:&lt;name&gt;:&lt;serverIP&gt;:&lt;port&gt;(always sent after ASSIGNMACHINE)<BR>
     * @see AlienAgent#sendAssignBot(String, InetAddress, int)
     * @param message the string to parse
     */
    public void processAssignBotMessage(String message) throws UnknownHostException{
    	Vector<String> parsed = parseByColons(message);
    	String name = parsed.get(1);
    	InetAddress serverIP = InetAddress.getByName(parsed.get(2));
    	int port = Integer.parseInt(parsed.get(3));
    	BotInterface bot = bots.get(name);
    	currentMachine.start(bot, serverIP, port);
    	runningMachines.add(currentMachine);
    }
    
    /**
     * Note: at present MATCHSTARTED messages are never sent.
     * Receive a message that a match has started.
     * MATCHSTARTED:&lt;matchname&gt;<BR>
     * @see AlienAgent#sendMatchStarted(String)
     * @param message the string to parse
     */
    public void processMatchStartMessage(String message){
      Vector<String> parsed = parseByColons(message);
      String matchNameString = parsed.get(1);
      System.err.println("Match started:"+matchNameString);
    }

    public void processMatchTerminate(String message){
        Vector<String> parsed = parseByColons(message);
        String matchNameString = parsed.get(1);
        System.err.println("Match terminated (message from server):"+matchNameString);
        if (listener!=null){
        	listener.handleMatchTerminated(matchNameString);
        }
      }

    /**
     * Process a message that a match has completed.
     * @see AlienAgent#sendMatchComplete(String)
     * @param message the message sent
     */
    public void processMatchComplete(String message){
        Vector<String> parsed = parseByColons(message);
        String matchNameString = parsed.get(1);
        System.err.println("Match completed:"+matchNameString);
        addCompletedMatch(matchNameString);
        if (listener!=null){
        	listener.handleMatchCompleted(matchNameString);
        }
    }
    
    public void addCompletedMatch(String matchNameString){
    	completedMatchStrings.add(matchNameString);
    }
    
    
    /**
     * Process a message from the server
     * @param message the message sent
     */
    public void processMessage(String message) throws UnknownHostException{
		if (message.startsWith("MATCHSTARTED:")) {
			processMatchStartMessage(message);
		} else if (message.startsWith("MATCHCOMPLETE:")) {
			processMatchComplete(message);
		} else if (message.startsWith("MATCHTERMINATE:")){
			processMatchTerminate(message);
		} else if (message.startsWith("ERROR:")) {
			System.out.println("Error from server");
			System.out.println(message);
			System.exit(0);
		} else if (message.startsWith("ASSIGNMACHINE:")){
			processAssignMachineMessage(message);
		} else if (message.startsWith("ASSIGNBOT:")){
			processAssignBotMessage(message);
		} else if (message.startsWith("CLEANMACHINE:")){
		    processCleanMachineMessage(message);
		} else {
		
			System.err.println("Unrecognized message from server");
			System.err.println(message);
			System.exit(0);
		}
	}
    /**
     * Create a bot
     * @see AlienAgent#processCreateBotMessage(String)
     * @param description the description of the bot
     * @throws TimeoutException 
     */
    public void sendCreateBot(String description) throws TimeoutException{
    	sendMessage("CREATEBOT:"+description);
    }
    
    /**
     * Create a machine<BR>
     * CREATEMACHINE:&lt;description&gt;
     * @see AlienAgent#processCreateMachineMessage(String)
     * @param description a description of the machine
     * @throws TimeoutException 
     */
    public void sendCreateMachine(String description) throws TimeoutException{
    	sendMessage("CREATEMACHINE:"+description);
    }
    
    /**
     * Logout of the server.
     */
    public void sendLogout() throws TimeoutException{
    	sendMessage("LOGOUT");
    }
    
    
    
    
    
    /**
     * 
	 * Add a user.
	 * ADDUSER:&lt;teamname&gt;:&lt;username&gt;:&lt;newpassword&gt;:&lt;email&gt;:&lt;accountType&gt;<BR>
     * @param teamName
     * @param username
     * @param password
     * @param email
     * @param accountType
     */
    public void sendAddUser(String teamName, String username, String password, 
    		String email, String accountType) throws TimeoutException{
    	sendMessage("ADDUSER:"+teamName+":"+username+":"+password+":"+email+":"+accountType);
    }
    
    
    /**
     * Sends a message to shutdown the server.<BR>
     * SHUTDOWN<BR>
     * @see AlienAgent#processShutdownMessage()
     * @throws TimeoutException
     */
    
    public void sendShutdown() throws TimeoutException, InterruptedException{
    	sendMessage("SHUTDOWN");
    	String message = receiveMessage();
    	System.err.println(message);

    }
    
    /**
     * Sends a message to terminate a match.<BR>
     * MATCHTERMINATE:&lt;matchname&gt;<BR>
     * @see AlienAgent#processMatchTerminateMessage(String)
     * @param matchName
     * @throws TimeoutException
     */
    public void sendMatchTerminate(String matchName) throws TimeoutException{
    	System.err.println("Terminating match "+matchName);
		sendMessage("MATCHTERMINATE:"+matchName);
		System.err.println("Terminating match message sent");
	}

	/**
     * 
     * @param account the account to change the password for
     * @param newpassword the new password
     * @throws TimeoutException
     */
    public void sendChangePassword(String account, String newpassword) throws TimeoutException{
    	sendMessage("CHANGEPASSWORD:"+account+":"+newpassword);
    }
    
    public void run(){
    	try{
    	while(true){
    		processMessage(receiveMessage());
    	}
    	} catch (UnknownHostException e){
    		e.printStackTrace(System.err);
    	} catch (TimeoutException to){
    		System.err.println("Note: currently logging out induces a timeout exception.");
    		to.printStackTrace(System.err);    		
    	} catch (InterruptedException ie){
    		System.err.println("Interrupted exception in receive message thread");
    	}
    }

    

}
